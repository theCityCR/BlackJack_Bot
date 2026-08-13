"""Tests for bet+play policy-gradient agents (REINFORCE / A2C / PPO)."""

from __future__ import annotations

from pathlib import Path

import torch

from agents.a2c import A2CAgent
from agents.common import (
    ACTION_LIST,
    SHOE_FEATURE_COUNT,
    encode_shoe,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from agents.counting import full_shoe_count_vector
from agents.pg_warmstart import warmstart_from_spread_rule
from agents.policy_base import (
    BET_ACTION_COUNT,
    bet_index_to_stake,
    stake_to_bet_index,
)
from agents.ppo import PPOAgent
from agents.reinforce import ReinforceAgent
from conftest import make_state
from game import Action, BlackjackGame, ShoeObservation


def _full_shoe() -> ShoeObservation:
    counts = full_shoe_count_vector(2)
    return ShoeObservation(count_vector=counts, cards_remaining=sum(counts))


def test_encode_shoe_shape_and_normalization():
    shoe = _full_shoe()
    features = encode_shoe(shoe)
    assert features.shape == (SHOE_FEATURE_COUNT,)
    assert features[0].item() == 1.0
    assert torch.isclose(features[1:].sum(), torch.tensor(1.0))


def test_bet_index_stake_roundtrip():
    for index in range(BET_ACTION_COUNT):
        stake = bet_index_to_stake(index)
        assert stake_to_bet_index(stake) == index


def test_reinforce_masks_illegal_play_actions():
    agent = ReinforceAgent()
    with torch.no_grad():
        for param in agent.model.parameters():
            param.zero_()
        agent.model.play_policy.bias[ACTION_LIST.index(Action.SPLIT)] = 100.0
        agent.model.play_policy.bias[ACTION_LIST.index(Action.STAND)] = 10.0

    state = make_state(player_value=12, can_double=False, can_split=False)
    action = agent.choose_action(
        state, [Action.HIT, Action.STAND], greedy=True
    )
    assert action == Action.STAND


def test_trajectory_assigns_bet_and_play_returns():
    agent = ReinforceAgent()
    game = BlackjackGame()
    trajectory = agent.collect_trajectory(game)
    assert trajectory.bet.return_ == trajectory.round_reward
    for step in trajectory.play:
        assert isinstance(step.return_, float)


def test_reinforce_train_one_episode_updates():
    agent = ReinforceAgent()
    before = agent.training_steps
    game = BlackjackGame()
    reward = agent.train_one_episode(game)
    assert isinstance(reward, float)
    assert agent.training_steps == before + 1


def test_a2c_train_one_episode_updates():
    agent = A2CAgent()
    before = agent.training_steps
    agent.train_one_episode(BlackjackGame())
    assert agent.training_steps == before + 1


def test_ppo_rollout_then_update():
    agent = PPOAgent(rollout_episodes=2, ppo_epochs=1, minibatch_size=8)
    game = BlackjackGame()
    agent.train_one_episode(game)
    assert agent._episodes_in_rollout == 1
    assert agent.training_steps == 0
    agent.train_one_episode(game)
    assert agent._episodes_in_rollout == 0
    assert agent.training_steps == 1


def test_play_episode_greedy_sets_last_bet():
    agent = A2CAgent()
    reward = agent.play_episode(BlackjackGame())
    assert isinstance(reward, float)
    assert 1.0 <= agent.last_bet <= 8.0


def test_policy_checkpoint_roundtrip(tmp_path: Path):
    agent = ReinforceAgent()
    agent.train_one_episode(BlackjackGame())
    path = tmp_path / "reinforce.pt"
    save_policy_checkpoint(agent, path)
    loaded, payload = load_policy_checkpoint(ReinforceAgent, path)
    assert payload["kind"] == "bet_play_pg"
    assert loaded.training_steps == agent.training_steps
    for a, b in zip(agent.model.parameters(), loaded.model.parameters()):
        assert torch.allclose(a, b)


def test_warmstart_from_spread_rule_runs():
    agent = A2CAgent()
    before = agent.training_steps
    warmstart_from_spread_rule(
        agent, BlackjackGame(), num_episodes=3, batch_size=1
    )
    assert agent.training_steps == before + 3


def test_freeze_play_uses_rule_chart_and_skips_play_updates():
    agent = A2CAgent(freeze_play=True, teacher_bet_ce_coef=0.1)
    assert agent.use_rule_play is True
    assert all(not p.requires_grad for p in agent.model.play_policy.parameters())
    assert any(p.requires_grad for p in agent.model.bet_policy.parameters())

    game = BlackjackGame()
    trajectory = agent.collect_trajectory(game)
    assert trajectory.play == []
    assert 0 <= trajectory.bet.teacher_bet_index < BET_ACTION_COUNT

    before = {
        name: param.detach().clone()
        for name, param in agent.model.named_parameters()
        if name.startswith("play_")
    }
    agent.update_from_trajectory(trajectory)
    after = dict(agent.model.named_parameters())
    for name, prior in before.items():
        assert torch.allclose(prior, after[name])


def test_bet_focus_cli_defaults():
    from agents.train_pg_cli import build_pg_arg_parser, resolve_pg_train_settings
    from config import (
        PG_BET_FOCUS_ARTIFACT_SUBDIR,
        PG_BET_FOCUS_BET_ENTROPY_COEF,
        PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES,
        PG_BET_FOCUS_FINAL_EVAL_EPISODES,
        PG_BET_FOCUS_PRINT_INTERVAL,
        PG_BET_FOCUS_TEACHER_BET_CE_COEF,
        PG_BET_FOCUS_TRAINING_EPISODES,
        PG_BET_FOCUS_WARMSTART_EPISODES,
    )

    parser = build_pg_arg_parser("test")
    args = parser.parse_args(["--bet-focus"])
    settings = resolve_pg_train_settings(args)
    assert settings["episodes"] == PG_BET_FOCUS_TRAINING_EPISODES
    assert settings["warmstart_episodes"] == PG_BET_FOCUS_WARMSTART_EPISODES
    assert settings["print_interval"] == PG_BET_FOCUS_PRINT_INTERVAL
    assert (
        settings["checkpoint_eval_episodes"]
        == PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES
    )
    assert settings["final_eval_episodes"] == PG_BET_FOCUS_FINAL_EVAL_EPISODES
    assert settings["artifact_subdir"] == PG_BET_FOCUS_ARTIFACT_SUBDIR
    assert settings["agent_kwargs"]["freeze_play"] is True
    assert (
        settings["agent_kwargs"]["bet_entropy_coef"] == PG_BET_FOCUS_BET_ENTROPY_COEF
    )
    assert (
        settings["agent_kwargs"]["teacher_bet_ce_coef"]
        == PG_BET_FOCUS_TEACHER_BET_CE_COEF
    )


def test_warmstart_batches_optimizer_steps():
    agent = A2CAgent(freeze_play=True)
    before = agent.training_steps
    warmstart_from_spread_rule(
        agent, BlackjackGame(), num_episodes=10, batch_size=4
    )
    # 10 episodes / batch 4 => 3 optimizer steps (4+4+2).
    assert agent.training_steps == before + 3


def test_freeze_play_checkpoint_restores_rule_play(tmp_path: Path):
    agent = A2CAgent(freeze_play=True, teacher_bet_ce_coef=0.05)
    agent.train_one_episode(BlackjackGame())
    path = tmp_path / "a2c_freeze.pt"
    save_policy_checkpoint(
        agent,
        path,
        extra={
            "freeze_play": True,
            "use_rule_play": True,
            "teacher_bet_ce_coef": 0.05,
        },
    )
    loaded, payload = load_policy_checkpoint(A2CAgent, path)
    assert payload["freeze_play"] is True
    assert loaded.freeze_play is True
    assert loaded.use_rule_play is True
    assert loaded.teacher_bet_ce_coef == 0.05
