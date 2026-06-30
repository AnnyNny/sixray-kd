# Run fine-tuned RT-DETR once and save teacher boxes, labels, scores,
# and approximate class probabilities to JSON.
# The cache is later used inside KD.

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]          # students/anna_student_resnet18
REPO_ROOT = Path(__file__).resolve().parents[3]     # sixray-kd
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))   # allows import from repo src.models.teacher
sys.path.insert(0, str(SRC_DIR))     # allows import sixray_student

from src.models.teacher import load_teacher  # teacher loader

from sixray_student.config import (
    IMAGE_SIZE,
    TRAIN_IMG_DIR,
    TRAIN_JSON,
    TEST_IMG_DIR,
    TEST_JSON,
    SPLIT_PATH,
    CLASS_NAMES,
)


BASE_MODEL_NAME = "PekingU/rtdetr_v2_r50vd"

TEACHER_MODEL_DIR = Path(
    "/content/drive/MyDrive/DatasetAPAI/SIXray_Project/kaggle_checkpoint_rtdetr_best"
)

TEACHER_CONF_THRESHOLD = 0.10

CACHE_DIR = Path("/content/drive/MyDrive/DatasetAPAI/SIXray_Project/teacher_cache")

TRAIN_CACHE_PATH = CACHE_DIR / "rtdetr_train_predictions_conf10.json"
VAL_CACHE_PATH = CACHE_DIR / "rtdetr_val_predictions_conf10.json"
TEST_CACHE_PATH = CACHE_DIR / "rtdetr_test_predictions_conf10.json"


def load_coco_images(annotation_file):
    annotation_file = Path(annotation_file)
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["images"]


def load_split_indices(split_path):
    split_path = Path(split_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return split["train_indices"], split["val_indices"], split["test_indices"]


def get_image_id_value(image_info):
    return int(image_info["id"])


def get_file_name(image_info):
    return image_info["file_name"]


def build_teacher(device):
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}

    processor, model = load_teacher(
        str(TEACHER_MODEL_DIR),
        id2label=id2label,
        label2id=label2id,
        device=device,
        use_data_parallel=False,
    )

    model.eval()
    return processor, model


def convert_teacher_label(label):
    label = int(label)
    if 0 <= label < len(CLASS_NAMES):
        return label
    return None


def clamp_box_to_student_size(box):
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(x1), float(IMAGE_SIZE)))
    y1 = max(0.0, min(float(y1), float(IMAGE_SIZE)))
    x2 = max(0.0, min(float(x2), float(IMAGE_SIZE)))
    y2 = max(0.0, min(float(y2), float(IMAGE_SIZE)))

    return [x1, y1, x2, y2]


def scale_box_to_student_size(box, original_width, original_height):
    x1, y1, x2, y2 = box

    scale_x = IMAGE_SIZE / original_width
    scale_y = IMAGE_SIZE / original_height

    box_scaled = [
        float(x1 * scale_x),
        float(y1 * scale_y),
        float(x2 * scale_x),
        float(y2 * scale_y),
    ]

    return clamp_box_to_student_size(box_scaled)


def get_approx_class_probs(outputs, num_predictions):
    """
    RT-DETR post-processing returns final boxes/scores/labels, but not always
    the exact query index. This func
    takes the top-confidence queries.
    """
    if not hasattr(outputs, "logits"):
        return None

    logits = outputs.logits[0].detach().cpu()
    probs = torch.softmax(logits, dim=-1)

    if probs.shape[-1] < len(CLASS_NAMES):
        return None

    probs = probs[:, : len(CLASS_NAMES)]
    query_scores = probs.max(dim=-1).values
    top_idx = torch.argsort(query_scores, descending=True)[:num_predictions]

    return probs[top_idx]


def predict_one_image(
    image_path,
    processor,
    model,
    device,
    confidence_threshold=TEACHER_CONF_THRESHOLD,
):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [[orig_h, orig_w]],
        dtype=torch.long,
        device=device,
    )

    processed = processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes_original = processed["boxes"].detach().cpu()
    scores = processed["scores"].detach().cpu()
    labels = processed["labels"].detach().cpu()

    class_probs = get_approx_class_probs(outputs, num_predictions=len(scores))

    result = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "class_probs": [],
        "original_size": [orig_w, orig_h],
        "student_image_size": IMAGE_SIZE,
    }

    for i in range(len(scores)):
        score = float(scores[i].item())
        label = convert_teacher_label(labels[i].item())

        if label is None:
            continue

        box_original = boxes_original[i].tolist()
        box_student = scale_box_to_student_size(
            box_original,
            original_width=orig_w,
            original_height=orig_h,
        )

        x1, y1, x2, y2 = box_student

        # Skip invalid boxes after clamping.
        if x2 <= x1 or y2 <= y1:
            continue

        result["boxes"].append(box_student)
        result["labels"].append(label)
        result["scores"].append(score)

        if class_probs is not None and i < len(class_probs):
            probs = class_probs[i].tolist()
            result["class_probs"].append([float(x) for x in probs])
        else:
            probs = [0.0] * len(CLASS_NAMES)
            probs[label] = 1.0
            result["class_probs"].append(probs)

    return result


def cache_split_predictions(
    split_name,
    images,
    selected_indices,
    image_dir,
    output_path,
    processor,
    model,
    device,
):
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    selected_images = [images[i] for i in selected_indices]

    num_images = 0
    num_predictions = 0
    num_images_with_predictions = 0

    for image_info in tqdm(selected_images, desc=f"Caching teacher predictions for {split_name}"):
        image_id = get_image_id_value(image_info)
        file_name = get_file_name(image_info)
        image_path = image_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prediction = predict_one_image(
            image_path=image_path,
            processor=processor,
            model=model,
            device=device,
            confidence_threshold=TEACHER_CONF_THRESHOLD,
        )

        cache[str(image_id)] = prediction

        n = len(prediction["boxes"])
        num_images += 1
        num_predictions += n
        if n > 0:
            num_images_with_predictions += 1

    metadata = {
        "base_model_name": BASE_MODEL_NAME,
        "teacher_model_dir": str(TEACHER_MODEL_DIR),
        "split_name": split_name,
        "confidence_threshold": TEACHER_CONF_THRESHOLD,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "num_images": num_images,
        "num_images_with_predictions": num_images_with_predictions,
        "num_predictions": num_predictions,
        "avg_predictions_per_image": (
            num_predictions / num_images if num_images > 0 else 0.0
        ),
    }

    full_output = {
        "metadata": metadata,
        "predictions": cache,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f)

    print(f"\nSaved {split_name} teacher cache to:")
    print(output_path)
    print(json.dumps(metadata, indent=2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Base model:", BASE_MODEL_NAME)
    print("Teacher dir:", TEACHER_MODEL_DIR)
    print("Teacher dir exists:", TEACHER_MODEL_DIR.exists())
    print("Teacher confidence threshold:", TEACHER_CONF_THRESHOLD)
    print("Class names:", CLASS_NAMES)
    print("SPLIT_PATH:", SPLIT_PATH)

    if device.type != "cuda":
        print("WARNING: CUDA is not available. Teacher caching will be slow.")

    train_images = load_coco_images(TRAIN_JSON)
    test_images = load_coco_images(TEST_JSON)

    train_indices, val_indices, test_indices = load_split_indices(SPLIT_PATH)

    print("Train images in annotation file:", len(train_images))
    print("Test images in annotation file:", len(test_images))
    print("Train split size:", len(train_indices))
    print("Val split size:", len(val_indices))
    print("Test split size:", len(test_indices))

    processor, model = build_teacher(device)

    cache_split_predictions(
        split_name="train",
        images=train_images,
        selected_indices=train_indices,
        image_dir=TRAIN_IMG_DIR,
        output_path=TRAIN_CACHE_PATH,
        processor=processor,
        model=model,
        device=device,
    )

    cache_split_predictions(
        split_name="val",
        images=train_images,
        selected_indices=val_indices,
        image_dir=TRAIN_IMG_DIR,
        output_path=VAL_CACHE_PATH,
        processor=processor,
        model=model,
        device=device,
    )

    cache_split_predictions(
        split_name="test",
        images=test_images,
        selected_indices=test_indices,
        image_dir=TEST_IMG_DIR,
        output_path=TEST_CACHE_PATH,
        processor=processor,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    main()