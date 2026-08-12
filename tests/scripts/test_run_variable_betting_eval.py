"""Tests for scripts/run_variable_betting_eval.py."""

from __future__ import annotations

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
