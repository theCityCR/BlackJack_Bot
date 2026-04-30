import pytest

from agents.rule_agent import RuleAgent
from game import Action, BlackjackGame, GameState


def test_rule_agent_hits_below_threshold():
    agent = RuleAgent(stand_threshold=17)

    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
    )

    assert agent.choose_action(state) == Action.HIT


def test_rule_agent_stands_at_threshold():
    agent = RuleAgent(stand_threshold=17)

    state = GameState(
        player_value=17,
        dealer_upcard=10,
        usable_ace=False,
    )

    assert agent.choose_action(state) == Action.STAND


def test_rule_agent_stands_above_threshold():
    agent = RuleAgent(stand_threshold=17)

    state = GameState(
        player_value=20,
        dealer_upcard=6,
        usable_ace=False,
    )

    assert agent.choose_action(state) == Action.STAND


def test_rule_agent_custom_threshold():
    agent = RuleAgent(stand_threshold=19)

    low_state = GameState(
        player_value=18,
        dealer_upcard=10,
        usable_ace=False,
    )

    high_state = GameState(
        player_value=19,
        dealer_upcard=10,
        usable_ace=False,
    )

    assert agent.choose_action(low_state) == Action.HIT
    assert agent.choose_action(high_state) == Action.STAND


def test_play_episode_returns_valid_reward():
    game = BlackjackGame()
    agent = RuleAgent(stand_threshold=17)

    reward = agent.play_episode(game)

    assert reward in {-1, 0, 1}
    assert game.done is True


def test_play_episode_can_run_multiple_times():
    game = BlackjackGame()
    agent = RuleAgent(stand_threshold=17)

    rewards = [agent.play_episode(game) for _ in range(10)]

    assert all(reward in {-1, 0, 1} for reward in rewards)
    assert game.done is True