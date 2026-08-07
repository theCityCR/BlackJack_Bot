"""Tests for the Double DQN ablation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.study_protocol import ABLATION_CONDITIONS
from config import (
    ABLATION_CONDITION_A,
    ABLATION_CONDITION_B,
    ABLATION_CONDITION_C,
    ABLATION_CONDITION_D,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_ablation_double_dqn.py"


def load_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_double_dqn",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ablation = load_ablation_module()

EXPECTED_CONDITION_IDS = {
    ABLATION_CONDITION_A,
    ABLATION_CONDITION_B,
    ABLATION_CONDITION_C,
    ABLATION_CONDITION_D,
}


def test_ablation_conditions_keys_match_expected():
    assert set(ABLATION_CONDITIONS.keys()) == EXPECTED_CONDITION_IDS


def test_ablation_condition_flags():
    assert ABLATION_CONDITIONS[ABLATION_CONDITION_A] == {
        "label": "Full from scratch",
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": False,
    }
    assert ABLATION_CONDITIONS[ABLATION_CONDITION_B] == {
        "label": "Hand-only",
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": True,
    }
    assert ABLATION_CONDITIONS[ABLATION_CONDITION_C] == {
        "label": "Curriculum",
        "curriculum": True,
        "warmstart": False,
        "force_shoe_off": False,
    }
    assert ABLATION_CONDITIONS[ABLATION_CONDITION_D] == {
        "label": "Curriculum + warm-start",
        "curriculum": True,
        "warmstart": True,
        "force_shoe_off": False,
    }


def test_distribution_to_rates():
    rates = ablation.distribution_to_rates(
        {
            "normal_win": 40,
            "blackjack_win": 5,
            "big_win_double_or_split": 5,
            "normal_loss": 40,
            "big_loss_double_or_split": 5,
            "draw": 5,
        }
    )
    assert rates == {
        "win_rate": 0.5,
        "loss_rate": 0.45,
        "draw_rate": 0.05,
    }


def test_write_ablation_results(tmp_path: Path):
    output_path = tmp_path / "ablation_results.json"
    rows = [
        {
            "condition_id": ABLATION_CONDITION_A,
            "label": "Full from scratch",
            "average_reward": 0.1,
            "training_steps": 42,
            "win_rate": 0.4,
            "loss_rate": 0.5,
            "draw_rate": 0.1,
            "episodes": 20,
            "eval_episodes": 100,
            "seed": 42,
            "curriculum": False,
            "warmstart": False,
            "force_shoe_off": False,
            "model_path": "/tmp/model.pt",
            "learning_curve_path": "/tmp/learning_curve.csv",
        }
    ]

    ablation.write_ablation_results(
        rows,
        output_path,
        seeds=[42],
        smoke=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["smoke"] is True
    assert payload["conditions"] == rows


@patch.object(ablation, "save_torch_checkpoint")
@patch.object(ablation, "evaluate_greedy")
@patch.object(ablation, "run_neural_training_loop")
@patch.object(ablation, "DoubleQNetworkLearningAgent")
@patch.object(ablation, "BlackjackGame")
@patch.object(ablation, "set_seed")
def test_run_ablation_condition_smoke_wiring(
    mock_set_seed,
    mock_game,
    mock_agent_cls,
    mock_run_loop,
    mock_eval_greedy,
    mock_save_checkpoint,
    tmp_path: Path,
):
    agent = MagicMock()
    agent.training_steps = 123
    mock_agent_cls.return_value = agent
    mock_run_loop.return_value = agent
    mock_eval_greedy.return_value = (
        0.05,
        {"normal_win": 50, "normal_loss": 40, "draw": 10},
    )

    row = ablation.run_ablation_condition(
        ABLATION_CONDITION_C,
        episodes=999,
        seed=7,
        smoke=True,
        ablation_base=tmp_path,
    )

    mock_set_seed.assert_called_once_with(7)
    mock_run_loop.assert_called_once()
    _, kwargs = mock_run_loop.call_args
    assert kwargs["curriculum"] is True
    assert kwargs["warmstart"] is False
    assert kwargs["force_shoe_off"] is False
    assert kwargs["phase_a_episodes"] == min(5, ablation.SMOKE_EPISODES // 2)
    assert kwargs["warmstart_episodes"] == min(5, ablation.SMOKE_EPISODES)
    assert kwargs["print_interval"] == ablation.SMOKE_PRINT_INTERVAL
    assert kwargs["checkpoint_eval_episodes"] == ablation.SMOKE_CHECKPOINT_EVAL_EPISODES

    mock_save_checkpoint.assert_called_once()
    mock_eval_greedy.assert_called_once_with(
        agent,
        ablation.SMOKE_FINAL_EVAL_EPISODES,
        seed=7,
    )

    assert row["condition_id"] == ABLATION_CONDITION_C
    assert row["episodes"] == ablation.SMOKE_EPISODES
    assert row["eval_episodes"] == ablation.SMOKE_FINAL_EVAL_EPISODES
    assert row["training_steps"] == 123
    assert row["resumed"] is False
    assert (tmp_path / ABLATION_CONDITION_C / "model.pt").as_posix() in row["model_path"]


@patch.object(ablation, "evaluate_greedy")
@patch.object(ablation, "load_torch_checkpoint")
@patch.object(ablation, "run_neural_training_loop")
@patch.object(ablation, "save_torch_checkpoint")
def test_run_ablation_condition_resume_skips_training(
    mock_save_checkpoint,
    mock_run_loop,
    mock_load_checkpoint,
    mock_eval_greedy,
    tmp_path: Path,
):
    condition_dir = tmp_path / ABLATION_CONDITION_B
    condition_dir.mkdir(parents=True)
    model_path = condition_dir / "model.pt"
    model_path.write_bytes(b"stub")

    agent = MagicMock()
    agent.training_steps = 999
    mock_load_checkpoint.return_value = (agent, {})
    mock_eval_greedy.return_value = (
        -0.02,
        {"normal_win": 40, "normal_loss": 50, "draw": 10},
    )

    row = ablation.run_ablation_condition(
        ABLATION_CONDITION_B,
        episodes=200_000,
        seed=42,
        smoke=True,
        ablation_base=tmp_path,
        resume=True,
    )

    mock_run_loop.assert_not_called()
    mock_save_checkpoint.assert_not_called()
    mock_load_checkpoint.assert_called_once()
    assert row["resumed"] is True
    assert row["average_reward"] == -0.02
    assert agent.use_shoe_features is False


@patch.object(ablation, "run_ablation_condition")
def test_main_smoke_writes_json(mock_run_condition, tmp_path: Path):
    mock_run_condition.return_value = {
        "condition_id": ABLATION_CONDITION_A,
        "label": "Full from scratch",
        "average_reward": 0.0,
        "training_steps": 0,
        "win_rate": 0.4,
        "loss_rate": 0.5,
        "draw_rate": 0.1,
        "episodes": ablation.SMOKE_EPISODES,
        "eval_episodes": ablation.SMOKE_FINAL_EVAL_EPISODES,
        "seed": 42,
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": False,
        "model_path": str(tmp_path / "model.pt"),
        "learning_curve_path": str(tmp_path / "learning_curve.csv"),
    }

    output_path = tmp_path / "ablation_results.json"
    exit_code = ablation.main(
        [
            "--smoke",
            "--conditions",
            ABLATION_CONDITION_A,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    mock_run_condition.assert_called_once()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert len(payload["conditions"]) == 1
    assert payload["conditions"][0]["condition_id"] == ABLATION_CONDITION_A


def test_smoke_run_one_condition_writes_json(tmp_path: Path):
    output_path = tmp_path / "ablation_results.json"
    exit_code = ablation.main(
        [
            "--smoke",
            "--conditions",
            ABLATION_CONDITION_A,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert len(payload["conditions"]) == 1

    row = payload["conditions"][0]
    assert row["condition_id"] == ABLATION_CONDITION_A
    assert row["episodes"] == ablation.SMOKE_EPISODES
    assert Path(row["model_path"]).is_file()
    assert Path(row["learning_curve_path"]).is_file()


@patch.object(ablation, "run_ablation_condition")
def test_main_multi_seed_uses_seed_subdirs(mock_run_condition, tmp_path: Path, monkeypatch):
    mock_run_condition.return_value = {
        "condition_id": ABLATION_CONDITION_A,
        "label": "Full from scratch",
        "average_reward": 0.0,
        "training_steps": 0,
        "win_rate": 0.4,
        "loss_rate": 0.5,
        "draw_rate": 0.1,
        "episodes": ablation.SMOKE_EPISODES,
        "eval_episodes": ablation.SMOKE_FINAL_EVAL_EPISODES,
        "seed": 0,
        "curriculum": False,
        "warmstart": False,
        "force_shoe_off": False,
        "model_path": str(tmp_path / "model.pt"),
        "learning_curve_path": str(tmp_path / "learning_curve.csv"),
    }
    monkeypatch.setattr(ablation, "ablation_base_dir", lambda: tmp_path)

    exit_code = ablation.main(
        [
            "--smoke",
            "--conditions",
            ABLATION_CONDITION_A,
            "--seeds",
            "11,12",
        ]
    )

    assert exit_code == 0
    assert mock_run_condition.call_count == 2
    first_base = mock_run_condition.call_args_list[0].kwargs["ablation_base"]
    second_base = mock_run_condition.call_args_list[1].kwargs["ablation_base"]
    assert first_base == tmp_path / "seed_11"
    assert second_base == tmp_path / "seed_12"
    assert (tmp_path / "seed_11" / "ablation_results.json").exists()
    aggregate_path = tmp_path / "multi_seed_ablation_results.json"
    assert aggregate_path.exists()
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert payload["summary"]["n_seeds"] == 2
    assert payload["summary"]["conditions"][0]["condition_id"] == ABLATION_CONDITION_A
    assert "average_reward" in payload["summary"]["conditions"][0]
    assert "per_seed" in payload
