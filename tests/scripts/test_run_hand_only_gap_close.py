"""Smoke test for the hand-only gap-close runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_hand_only_gap_close import run_gap_close


def test_gap_close_smoke(tmp_path, monkeypatch):
    # Redirect results into a temp dir via package_results_path override is hard;
    # use smoke mode which writes under agents/.../gap_close but finishes quickly.
    summary = run_gap_close(seed=0, smoke=True)
    assert summary["hand_only_encoder"] is True
    assert summary["state_size"] == 8
    assert "average_reward" in summary["agent"]
    assert "average_reward" in summary["rule_baseline"]
    assert Path(summary["model_path"]).exists()
    assert Path(summary["learning_curve_path"]).exists()
    results_json = Path(summary["model_path"]).parent / "gap_close_results.json"
    assert results_json.exists()
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
