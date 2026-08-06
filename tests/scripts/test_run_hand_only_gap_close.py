"""Smoke test for the hand-only gap-close runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.rule import RuleAgent
import scripts.run_hand_only_gap_close as gap_close
from scripts.run_hand_only_gap_close import (
    FULL_RESULTS_DIR,
    SMOKE_RESULTS_DIR,
    guard_full_results_overwrite,
    results_dir_for,
    run_gap_close,
)


def test_results_dir_for_separates_smoke_from_full():
    assert results_dir_for(smoke=True) == SMOKE_RESULTS_DIR
    assert results_dir_for(smoke=False) == FULL_RESULTS_DIR
    assert SMOKE_RESULTS_DIR != FULL_RESULTS_DIR


def test_guard_blocks_overwrite_of_full_summary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gap_close, "FULL_RESULTS_DIR", tmp_path)
    summary = tmp_path / "gap_close_results.json"
    summary.write_text(
        json.dumps(
            {
                "smoke": False,
                "train_episodes": 500_000,
                "agent": {"average_reward": -0.0248},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        guard_full_results_overwrite(tmp_path, force=False)

    guard_full_results_overwrite(tmp_path, force=True)


def test_guard_allows_replacing_smoke_leftovers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gap_close, "FULL_RESULTS_DIR", tmp_path)
    summary = tmp_path / "gap_close_results.json"
    summary.write_text(json.dumps({"smoke": True}) + "\n", encoding="utf-8")
    guard_full_results_overwrite(tmp_path, force=False)


def test_gap_close_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(gap_close, "SMOKE_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(gap_close, "FULL_RESULTS_DIR", tmp_path / "full_unused")

    summary = run_gap_close(seed=0, smoke=True)
    assert summary["hand_only_encoder"] is True
    assert summary["state_size"] == 8
    assert summary["smoke"] is True
    assert "average_reward" in summary["agent"]
    assert "average_reward" in summary["rule_baseline"]
    assert Path(summary["model_path"]).exists()
    assert Path(summary["learning_curve_path"]).exists()
    assert Path(summary["model_path"]).parent == tmp_path
    results_json = tmp_path / "gap_close_results.json"
    assert results_json.exists()
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    assert payload["smoke"] is True


@patch.object(gap_close, "save_torch_checkpoint")
@patch.object(gap_close, "evaluate_greedy")
@patch.object(gap_close, "run_neural_training_loop")
@patch.object(gap_close, "DoubleQNetworkLearningAgent")
@patch.object(gap_close, "BlackjackGame")
@patch.object(gap_close, "set_seed")
def test_gap_close_final_evals_share_seed(
    mock_set_seed,
    mock_game,
    mock_agent_cls,
    mock_run_loop,
    mock_eval_greedy,
    mock_save_checkpoint,
    tmp_path: Path,
):
    """Agent and rule baseline final evals must use the same paired seed."""
    agent = MagicMock()
    agent.input_size = 8
    agent.training_steps = 99
    mock_agent_cls.return_value = agent
    mock_run_loop.return_value = agent
    mock_eval_greedy.return_value = (
        -0.01,
        {"normal_win": 1, "normal_loss": 1, "draw": 0},
    )

    summary = run_gap_close(seed=17, smoke=True, results_dir=tmp_path)

    mock_set_seed.assert_called_once_with(17)
    assert mock_eval_greedy.call_count == 2
    first_call, second_call = mock_eval_greedy.call_args_list
    assert first_call.args[0] is agent
    assert first_call.kwargs == {"seed": 17}
    assert isinstance(second_call.args[0], RuleAgent)
    assert second_call.args[1] == first_call.args[1]
    assert second_call.kwargs == {"seed": 17}
    assert summary["seed"] == 17


@patch.object(gap_close, "save_torch_checkpoint")
@patch.object(gap_close, "evaluate_greedy")
@patch.object(gap_close, "run_neural_training_loop")
@patch.object(gap_close, "DoubleQNetworkLearningAgent")
@patch.object(gap_close, "BlackjackGame")
@patch.object(gap_close, "set_seed")
def test_smoke_run_does_not_touch_full_results_dir(
    mock_set_seed,
    mock_game,
    mock_agent_cls,
    mock_run_loop,
    mock_eval_greedy,
    mock_save_checkpoint,
    tmp_path: Path,
    monkeypatch,
):
    full_dir = tmp_path / "gap_close"
    smoke_dir = tmp_path / "gap_close_smoke"
    full_dir.mkdir()
    (full_dir / "gap_close_results.json").write_text(
        json.dumps({"smoke": False, "train_episodes": 500_000}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gap_close, "FULL_RESULTS_DIR", full_dir)
    monkeypatch.setattr(gap_close, "SMOKE_RESULTS_DIR", smoke_dir)

    agent = MagicMock()
    agent.input_size = 8
    agent.training_steps = 1
    mock_agent_cls.return_value = agent
    mock_eval_greedy.return_value = (-0.1, {"normal_win": 1})

    summary = run_gap_close(seed=0, smoke=True)

    assert Path(summary["model_path"]).parent == smoke_dir
    assert json.loads((full_dir / "gap_close_results.json").read_text())["smoke"] is False


@patch.object(gap_close, "save_torch_checkpoint")
@patch.object(gap_close, "evaluate_greedy")
@patch.object(gap_close, "run_neural_training_loop")
@patch.object(gap_close, "DoubleQNetworkLearningAgent")
@patch.object(gap_close, "BlackjackGame")
@patch.object(gap_close, "set_seed")
def test_main_multi_seed_writes_seed_subdirs(
    mock_set_seed,
    mock_game,
    mock_agent_cls,
    mock_run_loop,
    mock_eval_greedy,
    mock_save_checkpoint,
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(gap_close, "SMOKE_RESULTS_DIR", tmp_path / "smoke")
    monkeypatch.setattr(gap_close, "FULL_RESULTS_DIR", tmp_path / "full")

    agent = MagicMock()
    agent.input_size = 8
    agent.training_steps = 3
    mock_agent_cls.return_value = agent
    mock_eval_greedy.return_value = (-0.02, {"normal_win": 1})

    exit_code = gap_close.main(["--smoke", "--seeds", "3,4"])
    assert exit_code == 0
    assert (tmp_path / "smoke" / "seed_3" / "gap_close_results.json").exists()
    assert (tmp_path / "smoke" / "seed_4" / "gap_close_results.json").exists()
    assert (tmp_path / "smoke" / "multi_seed_gap_close_results.json").exists()
    assert not (tmp_path / "full").exists() or not any((tmp_path / "full").iterdir())
