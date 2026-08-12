"""Tests for scripts/run_penetration_sweep.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_penetration_sweep as sweep


def test_parse_thresholds():
    assert sweep.parse_thresholds("13, 26,52") == [13, 26, 52]


def test_parse_thresholds_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside shoe size"):
        sweep.parse_thresholds("200")


def test_sweep_thresholds_smoke_rows():
    payload = sweep.sweep_thresholds(
        [26, 52],
        episodes=40,
        seed=3,
        rounds_per_shoe=20,
    )
    assert payload["thresholds"] == [26, 52]
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["reshuffle_threshold"] == 26
    assert payload["runs"][1]["reshuffle_threshold"] == 52
    assert "delta_average_reward" in payload["runs"][0]


def test_main_smoke_writes_json(tmp_path: Path, monkeypatch):
    out = tmp_path / "pen.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_penetration_sweep.py",
            "--smoke",
            "--rounds-per-shoe",
            "25",
            "--output",
            str(out),
        ],
    )
    sweep.main()
    payload = json.loads(out.read_text())
    assert payload["smoke"] is True
    assert payload["thresholds"] == [26, 52]
    assert len(payload["runs"]) == 2
