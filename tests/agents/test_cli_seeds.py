"""Tests for multi-seed CLI scaffolding helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agents.cli_seeds import (
    add_seed_arguments,
    metric_stats,
    parse_seeds,
    seed_artifact_dir,
    seeds_from_args,
    summarize_ablation_runs,
    summarize_benchmark_runs,
    summarize_gap_close_runs,
    summarize_variable_betting_runs,
)


def test_parse_seeds_defaults_to_single_seed():
    assert parse_seeds(seed=42) == [42]
    assert parse_seeds(seed=7, seeds=None) == [7]


def test_parse_seeds_comma_list_overrides_seed():
    assert parse_seeds(seed=42, seeds="10, 20,30") == [10, 20, 30]


def test_parse_seeds_rejects_empty_and_non_int():
    with pytest.raises(ValueError, match="non-empty"):
        parse_seeds(seeds="")
    with pytest.raises(ValueError, match="non-empty"):
        parse_seeds(seeds="42,")
    with pytest.raises(ValueError, match="integers"):
        parse_seeds(seeds="42,x")


def test_seed_artifact_dir_keeps_legacy_path_for_single_seed(tmp_path: Path):
    assert seed_artifact_dir(tmp_path, 42, multi=False) == tmp_path
    assert seed_artifact_dir(tmp_path, 42, multi=True) == tmp_path / "seed_42"


def test_seeds_from_args_prefers_seeds_flag():
    parser = argparse.ArgumentParser()
    add_seed_arguments(parser)
    args = parser.parse_args(["--seed", "1", "--seeds", "2,3"])
    assert seeds_from_args(args) == [2, 3]


def test_metric_stats_single_value_has_zero_std():
    assert metric_stats([1.5]) == {
        "mean": 1.5,
        "std": 0.0,
        "n": 1,
        "min": 1.5,
        "max": 1.5,
    }


def test_metric_stats_sample_std():
    stats = metric_stats([1.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["std"] == pytest.approx(2**0.5)
    assert stats["n"] == 2
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_metric_stats_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        metric_stats([])


def test_summarize_ablation_runs_groups_by_condition():
    rows = [
        {
            "condition_id": "A_full_scratch",
            "label": "Full from scratch",
            "average_reward": -0.08,
            "win_rate": 0.41,
            "loss_rate": 0.50,
            "draw_rate": 0.09,
        },
        {
            "condition_id": "A_full_scratch",
            "label": "Full from scratch",
            "average_reward": -0.10,
            "win_rate": 0.40,
            "loss_rate": 0.51,
            "draw_rate": 0.09,
        },
        {
            "condition_id": "B_hand_only",
            "label": "Hand-only",
            "average_reward": -0.03,
            "win_rate": 0.42,
            "loss_rate": 0.49,
            "draw_rate": 0.09,
        },
    ]
    summary = summarize_ablation_runs(rows, seeds=[42, 43])
    assert summary["n_seeds"] == 2
    assert summary["seeds"] == [42, 43]
    by_id = {row["condition_id"]: row for row in summary["conditions"]}
    assert by_id["A_full_scratch"]["average_reward"]["mean"] == pytest.approx(-0.09)
    assert by_id["A_full_scratch"]["n_seeds"] == 2
    assert by_id["B_hand_only"]["average_reward"]["mean"] == pytest.approx(-0.03)
    assert by_id["B_hand_only"]["average_reward"]["std"] == 0.0


def test_summarize_gap_close_runs():
    rows = [
        {
            "seed": 42,
            "gap": -0.02,
            "agent_average_reward": -0.03,
            "rule_average_reward": -0.01,
        },
        {
            "seed": 43,
            "gap": -0.04,
            "agent_average_reward": -0.05,
            "rule_average_reward": -0.01,
        },
    ]
    summary = summarize_gap_close_runs(rows, seeds=[42, 43])
    assert summary["gap"]["mean"] == pytest.approx(-0.03)
    assert summary["agent_average_reward"]["n"] == 2
    assert summary["rule_average_reward"]["mean"] == pytest.approx(-0.01)


def test_summarize_benchmark_runs():
    aggregate = [
        {
            "seed": 1,
            "results": [
                {"agent": "Rule-based", "average_reward": -0.01},
                {"agent": "Double DQN", "average_reward": -0.04},
            ],
        },
        {
            "seed": 2,
            "results": [
                {"agent": "Rule-based", "average_reward": -0.02},
                {"agent": "Double DQN", "average_reward": -0.06},
            ],
        },
    ]
    summary = summarize_benchmark_runs(aggregate, seeds=[1, 2])
    by_name = {row["agent"]: row for row in summary["agents"]}
    assert by_name["Rule-based"]["average_reward"]["mean"] == pytest.approx(-0.015)
    assert by_name["Double DQN"]["average_reward"]["mean"] == pytest.approx(-0.05)


def test_summarize_variable_betting_runs():
    rows = [
        {
            "seed": 42,
            "flat_average_reward": 0.0,
            "spread_average_reward": 0.04,
            "spread_ev_per_unit_wagered": 0.02,
            "spread_average_stake": 2.0,
            "delta_average_reward": 0.04,
        },
        {
            "seed": 43,
            "flat_average_reward": -0.02,
            "spread_average_reward": 0.02,
            "spread_ev_per_unit_wagered": 0.01,
            "spread_average_stake": 2.2,
            "delta_average_reward": 0.04,
        },
    ]
    summary = summarize_variable_betting_runs(rows, seeds=[42, 43])
    assert summary["spread_average_reward"]["mean"] == pytest.approx(0.03)
    assert summary["delta_average_reward"]["mean"] == pytest.approx(0.04)
    assert summary["flat_average_reward"]["n"] == 2
