"""Study-protocol contracts for warm-start, curves, and ablations."""

from __future__ import annotations

from config import (
    ABLATION_CONDITION_A,
    ABLATION_CONDITION_B,
    ABLATION_CONDITION_C,
    ABLATION_CONDITION_D,
)

# CSV columns for learning_curve.csv (order matters).
LEARNING_CURVE_FIELDNAMES = (
    "episode",
    "training_steps",
    "eval_reward",
    "epsilon",
    "shoe_features_on",
)

ABLATION_CONDITIONS = {
    ABLATION_CONDITION_A: {
        "label": "Full from scratch",
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": False,
    },
    ABLATION_CONDITION_B: {
        "label": "Hand-only",
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": True,
    },
    ABLATION_CONDITION_C: {
        "label": "Curriculum",
        "curriculum": True,
        "warmstart": False,
        "force_shoe_off": False,
    },
    ABLATION_CONDITION_D: {
        "label": "Curriculum + warm-start",
        "curriculum": True,
        "warmstart": True,
        "force_shoe_off": False,
    },
}
