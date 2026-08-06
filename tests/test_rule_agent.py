"""Tests for the 2-deck S17 DAS RuleAgent chart."""

from __future__ import annotations

import pytest

from agents.rule import RuleAgent
from conftest import make_state
from game import Action, BlackjackGame


@pytest.fixture
def agent() -> RuleAgent:
    return RuleAgent()


# =========================
# Hard hit / stand
# =========================


def test_hard_16_hits_against_dealer_10(agent):
    assert agent.choose_action(make_state(16, 10)) == Action.HIT


def test_hard_16_stands_against_dealer_6(agent):
    assert agent.choose_action(make_state(16, 6)) == Action.STAND


def test_hard_12_hits_against_dealer_2(agent):
    assert agent.choose_action(make_state(12, 2)) == Action.HIT


def test_hard_12_stands_against_dealer_4(agent):
    assert agent.choose_action(make_state(12, 4)) == Action.STAND


def test_hard_17_stands(agent):
    assert agent.choose_action(make_state(17, 10)) == Action.STAND


# =========================
# Soft hit / stand
# =========================


def test_soft_17_hits_when_double_not_available(agent):
    assert agent.choose_action(make_state(17, 10, usable_ace=True)) == Action.HIT


def test_soft_18_stands_against_dealer_6_without_double(agent):
    state = make_state(18, 6, usable_ace=True, can_double=False)
    assert agent.choose_action(state) == Action.STAND


def test_soft_18_hits_against_dealer_10(agent):
    assert agent.choose_action(make_state(18, 10, usable_ace=True)) == Action.HIT


def test_soft_19_stands_against_dealer_10(agent):
    assert agent.choose_action(make_state(19, 10, usable_ace=True)) == Action.STAND


# =========================
# Doubles (including 2-deck cells)
# =========================


def test_hard_11_doubles_when_allowed(agent):
    assert agent.choose_action(make_state(11, 10, can_double=True)) == Action.DOUBLE


def test_hard_11_doubles_against_ace(agent):
    assert agent.choose_action(make_state(11, 1, can_double=True)) == Action.DOUBLE


def test_hard_10_doubles_against_dealer_9(agent):
    assert agent.choose_action(make_state(10, 9, can_double=True)) == Action.DOUBLE


def test_hard_10_does_not_double_against_dealer_10(agent):
    assert agent.choose_action(make_state(10, 10, can_double=True)) == Action.HIT


def test_hard_10_does_not_double_against_ace(agent):
    """Regression: Ace is upcard 1; must not treat dealer<=9 as covering Ace."""
    assert agent.choose_action(make_state(10, 1, can_double=True)) == Action.HIT


@pytest.mark.parametrize("dealer", [2, 3, 4, 5, 6])
def test_hard_9_doubles_against_dealer_2_to_6(agent, dealer):
    assert agent.choose_action(make_state(9, dealer, can_double=True)) == Action.DOUBLE


@pytest.mark.parametrize("dealer", [2, 3, 4, 5, 6])
def test_soft_18_doubles_against_dealer_2_to_6(agent, dealer):
    state = make_state(18, dealer, usable_ace=True, can_double=True)
    assert agent.choose_action(state) == Action.DOUBLE


def test_soft_19_doubles_against_dealer_6(agent):
    state = make_state(19, 6, usable_ace=True, can_double=True)
    assert agent.choose_action(state) == Action.DOUBLE


def test_soft_19_stands_against_dealer_5(agent):
    state = make_state(19, 5, usable_ace=True, can_double=True)
    assert agent.choose_action(state) == Action.STAND


@pytest.mark.parametrize(
    ("value", "dealer"),
    [
        (13, 5),
        (13, 6),
        (15, 4),
        (17, 3),
    ],
)
def test_soft_doubles_selected_cells(agent, value, dealer):
    state = make_state(value, dealer, usable_ace=True, can_double=True)
    assert agent.choose_action(state) == Action.DOUBLE


# =========================
# Splits
# =========================


def test_splits_aces(agent):
    state = make_state(12, 10, usable_ace=True, can_split=True)
    assert agent.choose_action(state) == Action.SPLIT


def test_splits_8s(agent):
    state = make_state(16, 10, can_split=True)
    assert agent.choose_action(state) == Action.SPLIT


def test_does_not_split_10s(agent):
    state = make_state(20, 6, can_split=True)
    assert agent.choose_action(state) == Action.STAND


def test_does_not_split_5s_prefers_double(agent):
    state = make_state(10, 6, can_double=True, can_split=True)
    assert agent.choose_action(state) == Action.DOUBLE


def test_splits_9s_against_dealer_9(agent):
    state = make_state(18, 9, can_split=True)
    assert agent.choose_action(state) == Action.SPLIT


def test_does_not_split_9s_against_dealer_7(agent):
    state = make_state(18, 7, can_split=True)
    assert agent.choose_action(state) == Action.STAND


@pytest.mark.parametrize(
    ("value", "dealer", "expected"),
    [
        (4, 5, Action.SPLIT),  # 2s
        (4, 8, Action.HIT),
        (6, 7, Action.SPLIT),  # 3s
        (8, 5, Action.SPLIT),  # 4s
        (8, 4, Action.HIT),
        (12, 7, Action.SPLIT),  # 6s
        (14, 7, Action.SPLIT),  # 7s
    ],
)
def test_pair_matrix_spots(agent, value, dealer, expected):
    state = make_state(value, dealer, can_split=True, can_double=False)
    assert agent.choose_action(state) == expected


# =========================
# Full episode smoke
# =========================


def test_play_episode_returns_numeric_reward():
    game = BlackjackGame()
    reward = RuleAgent().play_episode(game)
    assert isinstance(reward, (int, float))
    assert game.done is True


def test_play_episode_can_run_multiple_times():
    game = BlackjackGame()
    agent = RuleAgent()
    rewards = [agent.play_episode(game) for _ in range(100)]
    assert all(isinstance(reward, (int, float)) for reward in rewards)
    assert game.done is True


def test_play_episode_reward_can_include_blackjack_or_double_or_split_values():
    game = BlackjackGame()
    agent = RuleAgent()
    rewards = [agent.play_episode(game) for _ in range(500)]
    assert all(isinstance(reward, (int, float)) for reward in rewards)
