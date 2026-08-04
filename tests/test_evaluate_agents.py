"""Tests for evaluation helpers and CLI safety guards."""

from pathlib import Path

import pytest

from agents.rule_agent.rule_agent import RuleAgent
from evaluate_agents import (
    DOCS_RESULTS_DIR,
    DEFAULT_OUTPUT_DIR,
    evaluate_agent,
    guard_docs_publish,
    write_results,
)


def test_evaluation_is_reproducible():
    first = evaluate_agent("rule", RuleAgent(), episodes=100, seed=7)
    second = evaluate_agent("rule", RuleAgent(), episodes=100, seed=7)

    assert first == second
    assert first["episodes"] == 100
    assert first["win_rate"] + first["loss_rate"] + first["draw_rate"] == 1.0


def test_default_output_dir_is_not_docs_results():
    assert DEFAULT_OUTPUT_DIR.resolve() != DOCS_RESULTS_DIR.resolve()
    assert DEFAULT_OUTPUT_DIR.name == "eval"


def test_guard_docs_publish_blocks_incomplete_overwrite():
    with pytest.raises(SystemExit, match="Refusing to overwrite docs/results"):
        guard_docs_publish(DOCS_RESULTS_DIR, {"Rule-based baseline"})


def test_guard_docs_publish_allows_complete_set():
    guard_docs_publish(
        DOCS_RESULTS_DIR,
        {
            "Rule-based baseline",
            "Double DQN",
            "Dueling Double DQN",
            "Dueling Double DQN + PER",
        },
    )


def test_write_results_creates_artifacts(tmp_path: Path):
    results = [
        {
            "agent": "Rule-based baseline",
            "episodes": 10,
            "average_reward": -0.01,
            "win_rate": 0.4,
            "loss_rate": 0.5,
            "draw_rate": 0.1,
            "training_steps": None,
        }
    ]
    write_results(results, tmp_path, seed=42)

    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark_results.svg").exists()
