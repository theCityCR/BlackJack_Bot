"""
test_game.py

Tests for game.py and integration with cards.py.
"""

import pytest

from cards import Deck
from game import BlackjackGame, Action, GameState
from config import REWARD_WIN, REWARD_LOSS, REWARD_DRAW


class FixedDeck(Deck):
    """
    Deterministic deck for testing.

    Cards are drawn in the order given.
    """

    def __init__(self, cards):
        self.cards = cards
        self.index = 0

    def draw_card(self):
        if self.index >= len(self.cards):
            raise IndexError("No more cards in FixedDeck.")

        card = self.cards[self.index]
        self.index += 1
        return card

    def reset(self):
        pass


# =========================
# Reset / Initial State
# =========================

def test_reset_creates_player_and_dealer_hands():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)

    state = game.reset()

    assert game.player_hand.cards == [10, 7]
    assert game.dealer_hand.cards == [9, 6]
    assert isinstance(state, GameState)
    assert state.player_value == 17
    assert state.dealer_upcard == 9
    assert state.usable_ace is False


def test_get_state_returns_visible_state_only():
    deck = FixedDeck([1, 7, 10, 6])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.player_value == 18
    assert state.dealer_upcard == 10
    assert state.usable_ace is True


def test_state_as_tuple():
    state = GameState(player_value=18, dealer_upcard=10, usable_ace=True)

    assert state.as_tuple() == (18, 10, True)


# =========================
# Hit Behavior
# =========================

def test_hit_adds_card_and_game_continues_if_not_bust():
    deck = FixedDeck([10, 2, 9, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [10, 2, 5]
    assert next_state.player_value == 17
    assert reward == 0
    assert done is False


def test_hit_busts_player_and_ends_game():
    deck = FixedDeck([10, 9, 8, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [10, 9, 5]
    assert next_state is None
    assert reward == REWARD_LOSS
    assert done is True
    assert game.done is True


def test_hit_with_usable_ace_adjusts_value_correctly():
    deck = FixedDeck([1, 7, 10, 6, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [1, 7, 5]
    assert next_state.player_value == 13
    assert next_state.usable_ace is False
    assert reward == 0
    assert done is False


# =========================
# Stand / Dealer Behavior
# =========================

def test_stand_dealer_draws_until_17_or_more():
    deck = FixedDeck([10, 8, 9, 2, 5, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [9, 2, 5, 3]
    assert game.dealer_hand.value() == 19
    assert next_state is None
    assert done is True


def test_stand_player_wins_when_dealer_busts():
    deck = FixedDeck([10, 8, 10, 6, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.player_hand.value() == 18
    assert game.dealer_hand.value() == 26
    assert reward == REWARD_WIN
    assert next_state is None
    assert done is True


def test_stand_player_wins_with_higher_value():
    deck = FixedDeck([10, 9, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.player_hand.value() == 19
    assert game.dealer_hand.value() == 17
    assert reward == REWARD_WIN
    assert next_state is None
    assert done is True


def test_stand_player_loses_with_lower_value():
    deck = FixedDeck([10, 6, 10, 8])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.player_hand.value() == 16
    assert game.dealer_hand.value() == 18
    assert reward == REWARD_LOSS
    assert next_state is None
    assert done is True


def test_stand_draw_when_values_equal():
    deck = FixedDeck([10, 8, 10, 8])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.player_hand.value() == 18
    assert game.dealer_hand.value() == 18
    assert reward == REWARD_DRAW
    assert next_state is None
    assert done is True


# =========================
# Available Actions
# =========================

def test_available_actions_before_game_done():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)
    game.reset()

    assert game.available_actions() == [Action.HIT, Action.STAND]


def test_available_actions_after_game_done():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.available_actions() == []


# =========================
# Error Handling
# =========================

def test_get_state_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.get_state()


def test_step_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_step_after_game_done_raises_error():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_invalid_action_raises_error():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)
    game.reset()

    with pytest.raises(ValueError):
        game.step("invalid")

def test_dealer_stands_on_soft_17():
    deck = FixedDeck([10, 8, 1, 6])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.dealer_hand.cards == [1, 6]
    assert game.dealer_hand.value() == 17


def test_dealer_hits_soft_16():
    deck = FixedDeck([10, 8, 1, 5, 2])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.dealer_hand.cards == [1, 5, 2]
    assert game.dealer_hand.value() == 18


def test_blackjack_like_21_is_not_auto_win_on_reset():
    deck = FixedDeck([1, 10, 10, 9])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.player_value == 21
    assert game.done is False


def test_player_can_stand_on_21_and_win():
    deck = FixedDeck([1, 10, 10, 9])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert reward == REWARD_WIN
    assert next_state is None
    assert done is True


def test_reset_after_finished_game_starts_new_game():
    deck = FixedDeck([10, 7, 10, 7, 5, 5, 9, 8])
    game = BlackjackGame(deck)

    game.reset()
    game.step(Action.STAND)

    state = game.reset()

    assert game.done is False
    assert game.player_hand.cards == [5, 5]
    assert game.dealer_hand.cards == [9, 8]
    assert state.player_value == 10
    assert state.dealer_upcard == 9


def test_hit_after_stand_is_not_allowed():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_stand_after_bust_is_not_allowed():
    deck = FixedDeck([10, 9, 10, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.HIT)

    with pytest.raises(RuntimeError):
        game.step(Action.STAND)