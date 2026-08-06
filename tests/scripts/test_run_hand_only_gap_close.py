"""Smoke test for the hand-only gap-close runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.rule import RuleAgent
import scripts.run_hand_only_gap_close as gap_close
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
    monkeypatch,
):
    """Agent and rule baseline final evals must use the same paired seed."""
    monkeypatch.setattr(gap_close, "RESULTS_DIR", tmp_path)

    agent = MagicMock()
    agent.input_size = 8
    agent.training_steps = 99
    mock_agent_cls.return_value = agent
    mock_run_loop.return_value = agent
    mock_eval_greedy.return_value = (
        -0.01,
        {"normal_win": 1, "normal_loss": 1, "draw": 0},
    )

    summary = run_gap_close(seed=17, smoke=True)

    mock_set_seed.assert_called_once_with(17)
    assert mock_eval_greedy.call_count == 2
    first_call, second_call = mock_eval_greedy.call_args_list
    assert first_call.args[0] is agent
    assert first_call.kwargs == {"seed": 17}
    assert isinstance(second_call.args[0], RuleAgent)
    assert second_call.args[1] == first_call.args[1]
    assert second_call.kwargs == {"seed": 17}
    assert summary["seed"] == 17
