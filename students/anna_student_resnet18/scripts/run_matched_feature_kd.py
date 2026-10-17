"""Sequential approved control/KD runs, with durable logs and no automatic restart."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "students/anna_student_resnet18/src")]

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.engine.inference import prepare_validation_files
from src.utils.checkpoint import load_checkpoint
from sixray_student.config import CLASS_NAMES, GRID_SIZE, NUM_BOXES, BBOX_ENCODING
from sixray_student.data import SixRayStudentDataset, detection_collate_fn, load_split
from sixray_student.model import build_student_model
from sixray_student.losses import yolo_loss, feature_distillation_loss
from sixray_student.target_encoder import encode_targets_yolo
from sixray_student.metrics import evaluate_map
from sixray_student.train_utils import train_one_epoch, make_grad_scaler, step_scheduler_after_amp, autocast_context

ARTIFACTS = Path(r"C:\Users\89526\Documents\sixray-artifacts")
INITIAL_RUN = ARTIFACTS / "experiments/feature_kd_20261016T214646677055Z"


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def verify_scheduler():
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = make_grad_scaler(torch.device("cuda"))
    rows = []
    for overflow in (False, True, False):
        optimizer.zero_grad(set_to_none=True)
        before_parameter = parameter.detach().clone()
        before_lr, before_epoch = optimizer.param_groups[0]["lr"], scheduler.last_epoch
        loss = parameter.sum() * (float("inf") if overflow else 1)
        before_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step_scheduler_after_amp(scheduler, before_scale, scaler.get_scale(), 1)
        rows.append({"induced_overflow": overflow, "scheduler_before": before_epoch,
                     "scheduler_after": scheduler.last_epoch, "lr_before": before_lr,
                     "lr_after": optimizer.param_groups[0]["lr"],
                     "parameter_changed": not torch.equal(parameter, before_parameter)})
        if overflow:
            assert scheduler.last_epoch == before_epoch and optimizer.param_groups[0]["lr"] == before_lr
            assert torch.equal(parameter, before_parameter)
    return rows


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    config = json.loads((INITIAL_RUN / "kd/config.json").read_text())
    out = ARTIFACTS / "experiments" / ("matched_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    (ARTIFACTS / "experiments/latest_matched_run.txt").write_text(str(out))
    initial_path = INITIAL_RUN / "initial_student.pth"
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == config["initial_state_sha256"]
    config.update(device=str(device), training_amp=True,
                  gpu=torch.cuda.get_device_name(), initial_state_path=str(initial_path),
                  scheduler_cadence="CosineAnnealingLR T_max=50; one scheduler step per 5250 successful optimizer updates",
                  versions={p: version(p) for p in ("torch", "torchvision", "transformers", "torchmetrics", "pycocotools")},
                  python=sys.version, process_id=os.getpid(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), ROOT / "students/anna_student_resnet18/src/sixray_student/train_utils.py",
                                           ROOT / "students/anna_student_resnet18/src/sixray_student/losses.py",
                                           ROOT / "students/anna_student_resnet18/src/sixray_student/metrics.py",
                                           ROOT / "students/anna_student_resnet18/src/sixray_student/data.py",
                                           ROOT / "students/anna_student_resnet18/src/sixray_student/model.py")})
    print("MATCHED OUTPUT:", out, flush=True)
    for relative in config["source_sha256"]:
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    scheduler_check = verify_scheduler()
    seed_run(config["seed"])
    base = ARTIFACTS / "DatasetAPAI"
    train_indices, val_indices, _ = load_split(config["split_path"])
    prepare_validation_files(base / "subset_clean.zip", base / "subset_clean", train_indices + val_indices)
    train_data = SixRayStudentDataset(base / "subset_clean/train/images", base / "subset_clean/train.json", train_indices)
    val_data = SixRayStudentDataset(base / "subset_clean/train/images", base / "subset_clean/train.json", val_indices)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    processor, teacher, _ = load_checkpoint(config["teacher_path"], device=device, strict=True)
    teacher.requires_grad_(False).eval()

    # Disposable CUDA forward/backward verifies the same batch-2 KD workload as the pilot.
    probe = build_student_model(pretrained_backbone=False).to(device)
    probe.load_state_dict(initial)
    probe.train()
    with torch.random.fork_rng():
        torch.manual_seed(config["adapter_seed"])
        probe_adapter = torch.nn.Conv2d(512, 2048, 1, bias=False).to(device)
    images, targets = next(iter(DataLoader(train_data, batch_size=2, collate_fn=detection_collate_fn)))
    images = images.to(device)
    torch.cuda.reset_peak_memory_stats()
    with autocast_context(device, use_amp=True):
        features = probe.backbone(probe.normalizer(images))
        with torch.no_grad():
            reference = teacher.model.backbone.model(images).feature_maps[-1]
        detection, _ = yolo_loss(probe.head(features), encode_targets_yolo(targets, device=device))
        kd, _ = feature_distillation_loss(features, reference, probe_adapter, targets, background_weight=0.1)
    (detection + kd).backward()
    preflight = {"process_id": os.getpid(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name(), "images_device": str(images.device),
                 "student_features_device": str(features.device), "teacher_features_device": str(reference.device),
                 "student_shape": list(features.shape), "teacher_shape": list(reference.shape),
                 "feature_dtype": str(features.dtype),
                 "memory_probe_scope": "batch-2 CUDA AMP forward/backward; pilot memory includes AdamW steps and occasional gradient diagnostics",
                 "adapter_device": str(probe_adapter.weight.device), "teacher_device": str(next(teacher.parameters()).device),
                 "scheduler_skip_check": scheduler_check,
                 "cuda_probe_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "cuda_probe_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    assert features.is_cuda and reference.is_cuda and images.is_cuda and probe_adapter.weight.is_cuda
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2))
    print("PREFLIGHT:", json.dumps(preflight), flush=True)
    del probe, probe_adapter, images, features, reference, detection, kd
    torch.cuda.empty_cache()

    for name in ("control", "kd"):
        run_dir = out / name
        run_dir.mkdir()
        run_config = {**config, "run": name, "kd_lambda": 1.0 if name == "kd" else 0.0}
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
        with (run_dir / "training.log").open("x", encoding="utf-8", buffering=1) as log:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    seed_run(config["seed"])
                    model = build_student_model(pretrained_backbone=False).to(device)
                    model.load_state_dict(initial)
                    adapter = None
                    parameters = list(model.parameters())
                    if name == "kd":
                        with torch.random.fork_rng():
                            torch.manual_seed(config["adapter_seed"])
                            adapter = torch.nn.Conv2d(512, 2048, 1, bias=False).to(device)
                        parameters += list(adapter.parameters())
                    generator = torch.Generator().manual_seed(config["seed"])
                    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, generator=generator,
                                              num_workers=2, collate_fn=detection_collate_fn)
                    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=2,
                                            generator=torch.Generator().manual_seed(config["seed"]),
                                            collate_fn=detection_collate_fn)
                    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
                    scaler = make_grad_scaler(device)
                    history, best_map, best_epoch = [], -1.0, None
                    started = time.perf_counter()
                    for epoch in range(1, 51):
                        epoch_started = time.perf_counter()
                        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler,
                                                        use_amp=True, log_every_n_batches=1000,
                                                        teacher=teacher if name == "kd" else None, adapter=adapter,
                                                        kd_weight=1.0, background_weight=0.1, scheduler=scheduler)
                        validation = evaluate_map(model, val_loader, device, confidence_threshold=0.001,
                                                  nms_iou_threshold=0.5, max_detections=100,
                                                  use_amp=False, original_coordinates=True)
                        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                                        "train": train_metrics, "validation": validation,
                                        "seconds": time.perf_counter() - epoch_started,
                                        "optimizer_updates": getattr(scheduler, "optimizer_updates", 0)})
                        improved = validation["map"] > best_map
                        if improved:
                            best_map, best_epoch = validation["map"], epoch
                        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                                      "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
                                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                                      "scaler_state_dict": scaler.state_dict(), "data_generator_state": generator.get_state(),
                                      "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
                                      "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                                      "config": run_config, "history": history, "validation": validation,
                                      "best_checkpoint_metric": "map", "best_val_map": best_map, "best_epoch": best_epoch,
                                      "class_names": CLASS_NAMES, "image_size": 640, "grid_size": GRID_SIZE,
                                      "num_classes": len(CLASS_NAMES), "num_boxes": NUM_BOXES, "bbox_encoding": BBOX_ENCODING}
                        if improved:
                            atomic_save(checkpoint, run_dir / "student_best_map.pth")
                        atomic_save(checkpoint, run_dir / "student_last.pth")
                        temporary = run_dir / "history.tmp"
                        temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
                        temporary.replace(run_dir / "history.json")
                        progress = {"run": name, "epoch": epoch, "best_epoch": best_epoch, "best_map": best_map,
                                    "validation": validation, "epoch_seconds": history[-1]["seconds"],
                                    "elapsed_seconds": time.perf_counter() - started}
                        (out / "progress.json").write_text(json.dumps(progress, indent=2))
                        print(json.dumps(progress), flush=True)
                    (run_dir / "complete.json").write_text(json.dumps({"best_epoch": best_epoch, "best_map": best_map,
                                                                       "seconds": time.perf_counter() - started}, indent=2))
            except BaseException:
                failure = traceback.format_exc()
                (run_dir / "failure.txt").write_text(failure)
                if "model" in locals() and "optimizer" in locals():
                    atomic_save({"model_state_dict": model.state_dict(),
                                 "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
                                 "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                                 "scaler_state_dict": scaler.state_dict(), "history": history,
                                 "config": run_config, "failed_epoch": epoch}, run_dir / "student_failure.pth")
                raise
        print(name, "COMPLETE", "best epoch", best_epoch, "mAP", best_map, flush=True)
        del model, optimizer, scheduler, scaler, adapter, parameters, checkpoint, train_loader, val_loader
        torch.cuda.empty_cache()

    original = json.loads((ARTIFACTS / "evaluation_results/validation_20261016T170254281127Z/results.json").read_text())["metrics"]["student"]
    comparison = [{"run": "original_baseline", "best_epoch": 22, **original}]
    for name in ("control", "kd"):
        checkpoint_path = out / name / "student_best_map.pth"
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        comparison.append({"run": name, "best_epoch": best["epoch"], "checkpoint": str(checkpoint_path), **best["validation"]})
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("BOTH RUNS COMPLETE:", out, flush=True)


if __name__ == "__main__":
    main()
