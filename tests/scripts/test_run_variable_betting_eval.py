"""Tests for scripts/run_variable_betting_eval.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.spread_rule import SpreadRuleAgent
import scripts.run_variable_betting_eval as vb_eval


def test_run_comparison_summary_keys():
    summary = vb_eval.run_comparison(episodes=40, seed=3, rounds_per_shoe=20)
    assert summary["paired_eval"] is True
    assert summary["episodes"] == 40
    assert summary["rounds_per_shoe"] == 20
    assert summary["reshuffle_threshold"] == 26
    assert summary["dealt_penetration"] == pytest.approx(0.75)
    assert "average_reward" in summary["flat_rule"]
    assert "ev_per_unit_wagered" in summary["spread_rule"]
    assert "bet_fraction" in summary["spread_rule"]
    assert "delta_average_reward" in summary
    assert "bankroll" not in summary["spread_rule"]


def test_run_comparison_custom_reshuffle_threshold():
    deep = vb_eval.run_comparison(
        episodes=80,
        seed=11,
        rounds_per_shoe=40,
        reshuffle_threshold=13,
    )
    shallow = vb_eval.run_comparison(
        episodes=80,
        seed=11,
        rounds_per_shoe=40,
        reshuffle_threshold=52,
    )
    assert deep["reshuffle_threshold"] == 13
    assert shallow["reshuffle_threshold"] == 52
    assert deep["dealt_penetration"] > shallow["dealt_penetration"]


def test_run_comparison_includes_bankroll_when_requested():
    summary = vb_eval.run_comparison(
        episodes=40,
        seed=3,
        rounds_per_shoe=20,
        starting_bankroll=50.0,
        trip_rounds=10,
    )
    assert summary["starting_bankroll"] == 50.0
    assert summary["trip_rounds"] == 10
    spread_bankroll = summary["spread_rule"]["bankroll"]
    assert "path" in spread_bankroll
    assert "trips" in spread_bankroll
    assert spread_bankroll["trips"]["trip_rounds"] == 10
    assert 0.0 <= spread_bankroll["trips"]["risk_of_ruin"] <= 1.0


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
    assert "spread_risk_of_ruin" not in row


def test_compact_run_row_includes_bankroll_metrics():
    summary = vb_eval.run_comparison(
        episodes=40,
        seed=5,
        rounds_per_shoe=20,
        starting_bankroll=80.0,
    )
    row = vb_eval.compact_run_row(summary)
    assert row["starting_bankroll"] == 80.0
    assert row["trip_rounds"] == 20
    assert "spread_risk_of_ruin" in row
    assert "flat_risk_of_ruin" in row


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
    assert "starting_bankroll" not in payload


def test_main_multi_seed_with_bankroll(tmp_path: Path, monkeypatch):
    out = tmp_path / "multi_seed_bankroll.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_variable_betting_eval.py",
            "--smoke",
            "--seeds",
            "7,8",
            "--rounds-per-shoe",
            "25",
            "--bankroll",
            "100",
            "--trip-rounds",
            "25",
            "--output",
            str(out),
        ],
    )
    vb_eval.main()
    payload = json.loads(out.read_text())
    assert payload["starting_bankroll"] == 100.0
    assert payload["trip_rounds"] == 25
    assert "spread_risk_of_ruin" in payload["summary"]
    assert "spread_risk_of_ruin" in payload["runs"][0]


def test_main_rejects_trip_rounds_without_bankroll(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_variable_betting_eval.py", "--smoke", "--trip-rounds", "10"],
    )
    with pytest.raises(SystemExit):
        vb_eval.main()


def test_run_comparison_with_pg_agent():
    from agents.reinforce import ReinforceAgent

    agent = ReinforceAgent()
    summary = vb_eval.run_comparison(
        episodes=30,
        seed=2,
        rounds_per_shoe=15,
        pg_agent=agent,
        pg_agent_name="reinforce",
    )
    assert summary["pg_agent"] == "reinforce"
    assert "average_reward" in summary["pg_policy"]
    assert "delta_pg_minus_flat" in summary
    assert "delta_pg_minus_spread" in summary


def test_main_rejects_pg_agent_without_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_variable_betting_eval.py", "--smoke", "--pg-agent", "a2c"],
    )
    with pytest.raises(SystemExit):
        vb_eval.main()
