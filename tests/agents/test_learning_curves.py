"""Tests for learning-curve CSV logging and SVG plotting."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from agents.learning_curves import LearningCurveLogger
from agents.study_protocol import LEARNING_CURVE_FIELDNAMES

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_learning_curve_logger_writes_header_and_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "learning_curve.csv"
    logger = LearningCurveLogger(csv_path)

    logger.append(
        episode=1,
        training_steps=100,
        eval_reward=-0.05,
        epsilon=0.5,
        shoe_features_on=True,
    )
    logger.append(
        episode=2,
        training_steps=200,
        eval_reward=-0.02,
        epsilon=0.4,
        shoe_features_on=False,
    )

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(LEARNING_CURVE_FIELDNAMES)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0] == {
        "episode": "1",
        "training_steps": "100",
        "eval_reward": "-0.05",
        "epsilon": "0.5",
        "shoe_features_on": "1",
    }
    assert rows[1]["shoe_features_on"] == "0"


def test_plot_learning_curves_script(tmp_path: Path) -> None:
    csv_path = tmp_path / "run_a.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "episode": 1,
                    "training_steps": 10,
                    "eval_reward": -0.1,
                    "epsilon": 1.0,
                    "shoe_features_on": 1,
                },
                {
                    "episode": 2,
                    "training_steps": 20,
                    "eval_reward": 0.0,
                    "epsilon": 0.9,
                    "shoe_features_on": 1,
                },
            ]
        )

    output_svg = tmp_path / "out.svg"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_learning_curves.py"),
            str(csv_path),
            "--output",
            str(output_svg),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_svg.read_text(encoding="utf-8")
    assert content
    assert "<svg" in content
    assert "polyline" in content
