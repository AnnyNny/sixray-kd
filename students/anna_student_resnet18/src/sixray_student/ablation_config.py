"""
Ablation configuration for Anna's ResNet18 YOLO-style student detector.

Use environment variable:

PowerShell:
    $env:SIXRAY_ABLATION="one_box"
    python students\\anna_student_resnet18\\scripts\\train_student.py

Colab:
    %env SIXRAY_ABLATION=one_box
    !python students/anna_student_resnet18/scripts/train_student.py

Default:
    baseline
"""

import os


ABLATION_NAME = os.environ.get("SIXRAY_ABLATION", "baseline").strip().lower()


ABLATIONS = {
    "baseline": {
        "description": "Default student with two box slots per grid cell.",
        "overrides": {},
    },

    "one_box": {
        "description": "Ablation with one box slot per grid cell instead of two.",
        "overrides": {
            "NUM_BOXES": 1,
            "RESUME_TRAINING": False,
        },
    },

    "grid40_layer3": {
        "description": "Ablation with higher resolution 40x40 detection grid using resnet layer3",
        "overrides": {
            "GRID_SIZE" : 40,
            "NUM_BOXES" : 2,
            "BACKBONE_OUTPUT_LAYER" : "layer3",
            "RESUME_TRAINING" : True,
        }
    },
}


if ABLATION_NAME not in ABLATIONS:
    available = ", ".join(sorted(ABLATIONS.keys()))
    raise ValueError(
        f"Unknown SIXRAY_ABLATION='{ABLATION_NAME}'. "
        f"Available ablations: {available}"
    )


ABLATION_DESCRIPTION = ABLATIONS[ABLATION_NAME]["description"]
ABLATION_OVERRIDES = ABLATIONS[ABLATION_NAME]["overrides"]