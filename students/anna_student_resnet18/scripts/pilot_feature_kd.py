"""120-step KD pilot and a short timing benchmark; never launches full training."""
from pathlib import Path
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "students/anna_student_resnet18/src")]

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.engine.inference import prepare_validation_files
from src.utils.checkpoint import load_checkpoint
from sixray_student.data import SixRayStudentDataset, detection_collate_fn, load_split
from sixray_student.losses import yolo_loss, feature_distillation_loss
from sixray_student.model import build_student_model
from sixray_student.target_encoder import encode_targets_yolo
from sixray_student.train_utils import make_grad_scaler, autocast_context
from sixray_student.metrics import evaluate_map

ARTIFACTS = Path(r"C:\Users\89526\Documents\sixray-artifacts")
INITIAL_RUN = ARTIFACTS / "experiments/feature_kd_20261016T214646677055Z"
STEPS, WARMUP = 120, 10
GRADIENT_STEPS = {1, 30, 60, 90, 120}


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gradient_norm(gradients):
    return float(torch.stack([g.detach().float().square().sum() for g in gradients]).sum().sqrt())


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    assert torch.cuda.is_available(), "Run this pilot with CUDA-enabled PyTorch."
    config = json.loads((INITIAL_RUN / "kd/config.json").read_text())
    out = ARTIFACTS / "pilots" / ("feature_kd_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    config.update(pilot_optimizer_steps=STEPS, warmup_steps=WARMUP, device=str(device),
                  training_amp=True, gpu=torch.cuda.get_device_name(),
                  versions={p: version(p) for p in ("torch", "torchvision", "transformers", "torchmetrics")},
                  initial_state_path=str(INITIAL_RUN / "initial_student.pth"),
                  pilot_weights_for_final_experiments=False,
                  pilot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("Pilot directory:", out, flush=True)
    base = ARTIFACTS / "DatasetAPAI"
    train_indices, val_indices, _ = load_split(config["split_path"])
    # Reproduce DataLoader's base-seed draw followed by its first shuffled permutation.
    extraction_generator = torch.Generator().manual_seed(config["seed"])
    torch.empty((), dtype=torch.int64).random_(generator=extraction_generator)
    positions = torch.randperm(len(train_indices), generator=extraction_generator)[:416].tolist()
    prepare_validation_files(base / "subset_clean.zip", base / "subset_clean",
                             [train_indices[p] for p in positions])
    train_data = SixRayStudentDataset(base / "subset_clean/train/images", base / "subset_clean/train.json", train_indices)

    def loader():
        return DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                          generator=torch.Generator().manual_seed(config["seed"]),
                          num_workers=config["num_workers"], collate_fn=detection_collate_fn)

    seed_run(config["seed"])
    initial = torch.load(INITIAL_RUN / "initial_student.pth", map_location="cpu", weights_only=True)
    model = build_student_model(pretrained_backbone=False).to(device)
    model.load_state_dict(initial)
    model.train()
    with torch.random.fork_rng():
        torch.manual_seed(config["adapter_seed"])
        adapter = torch.nn.Conv2d(512, 2048, 1, bias=False).to(device)
    adapter_initial = adapter.weight.detach().cpu().clone()
    processor, teacher, _ = load_checkpoint(config["teacher_path"], device=device, strict=True)
    teacher.requires_grad_(False).eval()
    teacher_before = state_hash(teacher)
    parameters = list(model.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = make_grad_scaler(device, use_amp=True)
    iterator = iter(loader())
    records, gradients = [], []
    done, attempts, overflow_count = 0, 0, 0
    nan_gradient_attempts, infinite_gradient_attempts = 0, 0
    torch.cuda.reset_peak_memory_stats()
    while done < STEPS:
        attempts += 1
        torch.cuda.synchronize()
        started = time.perf_counter()
        images, targets = next(iterator)
        images = images.to(device)
        encoded = encode_targets_yolo(targets, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp=True):
            features = model.backbone(model.normalizer(images))
            detection, losses = yolo_loss(model.head(features), encoded)
            with torch.no_grad():
                reference = teacher.model.backbone.model(images).feature_maps[-1]
            kd, kd_values = feature_distillation_loss(features, reference, adapter, targets,
                                                       background_weight=config["background_weight"])
            total = detection + config["kd_lambda"] * kd
        assert torch.isfinite(total), "Nonfinite objective; pilot stopped."
        measured = done + 1 in GRADIENT_STEPS and not any(g["optimizer_step"] == done + 1 for g in gradients)
        if measured:
            backbone = tuple(model.backbone.parameters())
            gd = torch.autograd.grad(detection, backbone, retain_graph=True)
            gk = torch.autograd.grad(kd, backbone, retain_graph=True)
            ga = torch.autograd.grad(kd, tuple(adapter.parameters()), retain_graph=True)
            gradients.append({"optimizer_step": done + 1, "attempt": attempts,
                              "image_ids": [int(t["image_id"]) for t in targets],
                              "objects_per_image": [len(t["boxes"]) for t in targets],
                              "detection_backbone": gradient_norm(gd), "kd_backbone": gradient_norm(gk),
                              "kd_adapter": gradient_norm(ga)})
            del gd, gk, ga
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip_norm"])
        nan_gradient_attempts += int(torch.isnan(norm))
        infinite_gradient_attempts += int(torch.isinf(norm))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        success = scaler.get_scale() >= scale_before
        if measured:
            gradients[-1]["optimizer_step_applied"] = success
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not success:
            overflow_count += 1
            print("AMP overflow: skipped attempt", attempts, "new scale", scaler.get_scale(), flush=True)
            assert attempts < 200, "Too many skipped AMP steps."
            continue
        done += 1
        row = {"optimizer_step": done, "attempt": attempts, **losses, **kd_values,
               "detection_loss": float(detection.detach()), "combined_loss": float(total.detach()),
               "gradient_norm_before_clip": float(norm), "seconds": seconds,
               "timed": done > WARMUP and not measured,
               "positive_images": sum(bool(len(t["boxes"])) for t in targets),
               "image_ids": [int(t["image_id"]) for t in targets]}
        assert np.isfinite(row["gradient_norm_before_clip"]), "Nonfinite gradient on a successful step."
        records.append(row)
        if done % 20 == 0:
            window = records[-20:]
            print(done, {key: round(float(np.mean([r[key] for r in window])), 5)
                         for key in ("detection_loss", "objectness_loss", "bbox_loss", "class_loss", "kd_loss")}, flush=True)
        del features, reference, detection, kd, total

    kd_memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    final_amp_scale = scaler.get_scale()
    updates = {}
    differences = [p.detach().cpu() - initial["backbone." + name] for name, p in model.backbone.named_parameters()]
    updates["backbone_delta_l2"] = gradient_norm(differences)
    updates["adapter_delta_l2"] = gradient_norm([adapter.weight.detach().cpu() - adapter_initial])
    updates["teacher_state_hash_before"] = teacher_before
    updates["teacher_state_hash_after"] = state_hash(teacher)
    updates["teacher_state_unchanged"] = updates["teacher_state_hash_after"] == teacher_before
    updates["teacher_parameters_with_gradients"] = sum(p.grad is not None for p in teacher.parameters())
    updates["teacher_trainable_parameters"] = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    assert updates["teacher_state_unchanged"] and updates["teacher_parameters_with_gradients"] == 0
    assert updates["backbone_delta_l2"] > 0 and updates["adapter_delta_l2"] > 0
    torch.save({"model_state_dict": model.state_dict(), "adapter_state_dict": adapter.state_dict(),
                "optimizer_steps": done, "pilot_only": True, "config": config}, out / "pilot_only_weights.pth")
    (out / "steps.json").write_text(json.dumps(records, indent=2))
    (out / "gradients.json").write_text(json.dumps(gradients, indent=2, allow_nan=False))
    with (out / "steps.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # Timing-only supervised benchmark, in this pilot directory, never final control weights.
    del iterator, optimizer, scaler, model, adapter, teacher, parameters, differences, backbone
    torch.cuda.empty_cache()
    seed_run(config["seed"])
    control = build_student_model(pretrained_backbone=False).to(device)
    control.load_state_dict(initial)
    control.train()
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    control_scaler = make_grad_scaler(device, use_amp=True)
    control_times = []
    control_iterator = iter(loader())
    for index in range(25):
        torch.cuda.synchronize()
        started = time.perf_counter()
        images, targets = next(control_iterator)
        images = images.to(device)
        encoded = encode_targets_yolo(targets, device=device)
        control_optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp=True):
            loss, _ = yolo_loss(control(images), encoded)
        control_scaler.scale(loss).backward()
        control_scaler.unscale_(control_optimizer)
        torch.nn.utils.clip_grad_norm_(control.parameters(), config["grad_clip_norm"])
        scale_before = control_scaler.get_scale()
        control_scaler.step(control_optimizer)
        control_scaler.update()
        torch.cuda.synchronize()
        if index >= 5 and control_scaler.get_scale() >= scale_before:
            control_times.append(time.perf_counter() - started)

    val_data = SixRayStudentDataset(base / "subset_clean/train/images", base / "subset_clean/train.json", val_indices[:100])
    val_loader = DataLoader(val_data, batch_size=2, num_workers=2, collate_fn=detection_collate_fn)
    torch.cuda.synchronize()
    started = time.perf_counter()
    evaluate_map(control, val_loader, device, confidence_threshold=config["score_threshold"],
                 nms_iou_threshold=config["nms_iou"], max_detections=config["max_detections"],
                 use_amp=False, original_coordinates=True)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - started
    kd_seconds = float(np.mean([r["seconds"] for r in records if r["timed"]]))
    control_seconds = float(np.mean(control_times))
    training_steps = len(train_indices) // 2 * config["epochs"]
    validation_estimate = validation_seconds * len(val_indices) / 100 * config["epochs"]
    windows = [{"steps": f"{start + 1}-{start + 20}", **{
        key: float(np.mean([r[key] for r in records[start:start + 20]]))
        for key in ("detection_loss", "objectness_loss", "bbox_loss", "class_loss", "kd_loss", "combined_loss",
                    "kd_foreground", "kd_background", "positive_images")}}
        for start in range(0, STEPS, 20)]
    summary = {"successful_kd_optimizer_steps": done, "attempts": attempts,
               "amp_overflow_skips": overflow_count, "nonfinite_losses": 0,
               "nan_gradient_norm_attempts": nan_gradient_attempts,
               "infinite_gradient_norm_attempts": infinite_gradient_attempts,
               "final_amp_scale": final_amp_scale,
               "combined_gradient_norm_min": min(r["gradient_norm_before_clip"] for r in records),
               "combined_gradient_norm_max": max(r["gradient_norm_before_clip"] for r in records),
               "combined_gradient_norm_median": float(np.median([r["gradient_norm_before_clip"] for r in records])),
               "gradient_check_steps": gradients, "updates": updates, "kd_memory": kd_memory,
               "kd_seconds_per_step": kd_seconds, "kd_timed_steps": sum(r["timed"] for r in records),
               "control_seconds_per_step": control_seconds, "control_timing_only_attempts": 25,
               "validation_100_images_seconds": validation_seconds, "loss_windows": windows,
               "estimated_control_50_epochs_hours": (training_steps * control_seconds + validation_estimate) / 3600,
               "estimated_kd_50_epochs_hours": (training_steps * kd_seconds + validation_estimate) / 3600,
               "runtime_caveat": "Linear projection including per-epoch validation; excludes checkpoint I/O and long-run thermal/data-cache effects."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ("detection_loss", "kd_loss", "combined_loss"):
        axes[0].plot([r["optimizer_step"] for r in records], [r[key] for r in records], alpha=0.6, label=key)
    axes[0].legend()
    axes[0].set_xlabel("Successful optimizer step")
    axes[1].plot([r["optimizer_step"] for r in records], [r["gradient_norm_before_clip"] for r in records])
    axes[1].set(xlabel="Successful optimizer step", ylabel="Combined gradient norm before clipping")
    fig.tight_layout()
    fig.savefig(out / "progress.png")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    print("PILOT COMPLETE. Full experiments were not started. Results:", out, flush=True)


if __name__ == "__main__":
    main()
