# run RT-DETR once, saves teacher boxes, labels, scores and class prob to json
# does not train
# cache is used inside KD 

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from sixray_student.config import (
    IMAGE_SIZE,
    TRAIN_IMG_DIR,
    TRAIN_JSON,
    TEST_IMG_DIR,
    TEST_JSON,
    SPLIT_PATH,
    CLASS_NAMES
)

MODEL_NAME = "PekingU/rtdetr_v2_r50vd"
TEACHER_MODEL_DIR = Path(
    "/content/drive/MyDrive/DatasetAPAI/SIXray_Project/kaggle_checkpoint_rtdetr_best"
)
TEACHER_CONF_THRESHOLD = 0.30

CACHE_DIR = Path("/content/drive/MyDrive/DatasetAPAI/SIXray_Project/teacher_cache")

TRAIN_CACHE_PATH = CACHE_DIR / "rtdetr_train_predictions_conf03.json"
VAL_CACHE_PATH = CACHE_DIR / "rtdetr_val_predictions_conf03.json"
TEST_CACHE_PATH = CACHE_DIR / "rtdetr_test_predictions_conf03.json"

def load_coco_images(annotation_file):
    annotation_file = Path(annotation_file)
    with open (annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data["images"]
    return images

def load_split_indices(split_path):
    split_path = Path(split_path)
    with open (split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)

    return split["train_indices"], split["val_indices"], split["test_indices"]

def get_image_id_value(image_info):
    return int(image_info["id"])

def get_file_name(image_info):
    return image_info["file_name"]

def build_teacher(device):
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(TEACHER_MODEL_DIR)
    model = AutoModelForObjectDetection.from_pretrained(
        TEACHER_MODEL_DIR,
        id2label=id2label,
        label2id=label2id,
    )

    model.to(device)
    model.eval()

    return processor, model

def convert_teacher_label(label):
    label = int(label)
    if 0<=label< len(CLASS_NAMES):
            return label
    return None

def scale_box_to_student_size(box, original_width, original_height):
    x1, y1, x2, y2 = box

    scale_x = IMAGE_SIZE / original_width
    scale_y = IMAGE_SIZE / original_height

    return [
        float(x1 * scale_x),
        float(y1 * scale_y),
        float(x2 * scale_x),
        float(y2 * scale_y),
    ]

def predict_one_image(image_path, processor, model, device,confidence_threshold=TEACHER_CONF_THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key:value.to(device) for key, value in inputs.items()}

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

    # [B, num_queries, num_classes] match each postprocessed prediction to the nearest query
    class_probs = None
    if hasattr(outputs, "logits"):
        logits = outputs.logits[0].detach().cpu()
        probs = torch.softmax(logits, dim=-1)

        if probs.shape[-1] >= len(CLASS_NAMES):
            probs = probs[:, : len(CLASS_NAMES)]
            query_scores = probs.max(dim=-1).values
            top_idx = torch.argsort(query_scores, descending=True)[: len(scores)]
            class_probs = probs[top_idx]

        result = {
            "boxes": [],
            "labels" : [],
            "scores" : [],
            "class_probs" : [],
            "original_size" : [orig_w, orig_h],
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
        "model_name": MODEL_NAME,
        "split_name": split_name,
        "confidence_threshold": TEACHER_CONF_THRESHOLD,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "num_images": num_images,
        "num_images_with_predictions": num_images_with_predictions,
        "num_predictions": num_predictions,
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
    print("Model:", MODEL_NAME)
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
