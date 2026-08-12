"""Tests for scripts/run_variable_betting_eval.py."""

from __future__ import annotations

import json
from pathlib import Path

from agents.spread_rule import SpreadRuleAgent
import scripts.run_variable_betting_eval as vb_eval


def test_run_comparison_summary_keys():
    summary = vb_eval.run_comparison(episodes=40, seed=3, rounds_per_shoe=20)
    assert summary["paired_eval"] is True
    assert summary["episodes"] == 40
    assert summary["rounds_per_shoe"] == 20
    assert "average_reward" in summary["flat_rule"]
    assert "ev_per_unit_wagered" in summary["spread_rule"]
    assert "bet_fraction" in summary["spread_rule"]
    assert "delta_average_reward" in summary


def test_persistent_shoe_eval_uses_raised_bets():
    stats = vb_eval.evaluate_spread_policy(
        SpreadRuleAgent(),
        episodes=200,
        seed=11,
        rounds_per_shoe=100,
    )
    assert stats["average_stake"] > 1.0
    assert any(float(bet) > 1 for bet in stats["bet_fraction"])


def test_compact_run_row_flattens_metrics():
    summary = vb_eval.run_comparison(episodes=40, seed=5, rounds_per_shoe=20)
    row = vb_eval.compact_run_row(summary)
    assert row["seed"] == 5
    assert "flat_average_reward" in row
    assert "delta_average_reward" in row


def test_main_multi_seed_writes_aggregate(tmp_path: Path, monkeypatch):
    out = tmp_path / "multi_seed.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_variable_betting_eval.py",
            "--smoke",
            "--seeds",
            "7,8",
            "--rounds-per-shoe",
            "25",
            "--output",
            str(out),
        ],
    )
    vb_eval.main()
    payload = json.loads(out.read_text())
    assert payload["seeds"] == [7, 8]
    assert payload["smoke"] is True
    assert len(payload["runs"]) == 2
    assert payload["summary"]["n_seeds"] == 2
    assert "spread_average_reward" in payload["summary"]
