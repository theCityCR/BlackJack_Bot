"""Learning-curve CSV logging for neural training runs.

Contract (Wave 1):
    LearningCurveLogger(path) with append(episode, training_steps, eval_reward,
    epsilon, shoe_features_on) writing LEARNING_CURVE_FIELDNAMES from
    agents.study_protocol.

    plot_learning_curves(csv_paths, output_svg) in scripts/plot_learning_curves.py
    using stdlib only (no matplotlib/numpy required).
"""

from __future__ import annotations

import csv
from pathlib import Path

from agents.study_protocol import LEARNING_CURVE_FIELDNAMES


class LearningCurveLogger:
    """Append greedy-eval metrics to a CSV beside training checkpoints."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDNAMES)
                writer.writeheader()

    def append(
        self,
        *,
        episode: int,
        training_steps: int,
        eval_reward: float,
        epsilon: float,
        shoe_features_on: bool,
    ) -> None:
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDNAMES)
            writer.writerow(
                {
                    "episode": episode,
                    "training_steps": training_steps,
                    "eval_reward": eval_reward,
                    "epsilon": epsilon,
                    "shoe_features_on": int(shoe_features_on),
                }
            )
            handle.flush()
