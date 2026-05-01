"""
test_game.py

Tests for the updated BlackjackGame environment.

These tests cover:
- old hit/stand behavior
- the new GameState fields for Q-learning
- double-down behavior through per-hand bets
- split behavior through multiple hands
- split-ace restrictions
- legal/illegal action handling
"""

import pytest

from cards import Deck
from config import REWARD_DRAW, REWARD_LOSS, REWARD_WIN
from game import Action, BlackjackGame, GameState


class FixedDeck(Deck):
    """
    Deterministic deck for testing.

    Cards are drawn from left to right.
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
# GameState / Reset
# =========================


def test_game_state_as_tuple_includes_q_learning_context():
    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=True,
    )

    assert state.as_tuple() == (16, 10, False, True, False, True)


def test_reset_creates_initial_hands_and_state():
    deck = FixedDeck([10, 6, 9, 7])
    game = BlackjackGame(deck)

    state = game.reset()

    assert game.player_hands[0].cards == [10, 6]
    assert game.player_hand.cards == [10, 6]
    assert game.dealer_hand.cards == [9, 7]
    assert game.active_hand_index == 0
    assert game.hand_bets == [1]
    assert game.is_split_hand == [False]
    assert game.is_split_aces_hand == [False]

    assert state.player_value == 16
    assert state.dealer_upcard == 9
    assert state.usable_ace is False
    assert state.can_double is True
    assert state.can_split is False
    assert state.is_split_hand is False


def test_get_state_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.get_state()


# =========================
# Available actions
# =========================


def test_available_actions_initial_non_pair_has_hit_stand_double_only():
    deck = FixedDeck([10, 6, 9, 7])
    game = BlackjackGame(deck)
    game.reset()

    assert game.available_actions() == [Action.HIT, Action.STAND, Action.DOUBLE]


def test_available_actions_initial_pair_includes_split():
    deck = FixedDeck([8, 8, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    assert game.available_actions() == [
        Action.HIT,
        Action.STAND,
        Action.DOUBLE,
        Action.SPLIT,
    ]


def test_double_not_available_after_hit():
    deck = FixedDeck([10, 2, 9, 7, 4])
    game = BlackjackGame(deck)
    game.reset()

    state, reward, done = game.step(Action.HIT)

    assert state.player_value == 16
    assert state.can_double is False
    assert Action.DOUBLE not in game.available_actions()
    assert reward == 0
    assert done is False


def test_available_actions_after_done_is_empty():
    deck = FixedDeck([10, 8, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.available_actions() == []


# =========================
# Hit / Stand old behavior
# =========================


def test_hit_adds_card_and_continues_if_not_bust():
    deck = FixedDeck([10, 2, 9, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [10, 2, 5]
    assert next_state.player_value == 17
    assert next_state.can_double is False
    assert reward == 0
    assert done is False
    assert game.done is False


def test_hit_busts_only_hand_and_ends_without_dealer_drawing():
    # Dealer starts at 15. If the dealer incorrectly plays after the player busts,
    # this test would need another card and fail with IndexError.
    deck = FixedDeck([10, 9, 8, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [10, 9, 5]
    assert game.dealer_hand.cards == [8, 7]
    assert next_state is None
    assert reward == REWARD_LOSS
    assert done is True
    assert game.done is True


def test_stand_dealer_draws_until_17_or_more():
    deck = FixedDeck([10, 8, 9, 2, 5, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [9, 2, 5, 3]
    assert game.dealer_hand.value() == 19
    assert next_state is None
    assert reward == REWARD_LOSS
    assert done is True


def test_stand_player_wins_when_dealer_busts():
    deck = FixedDeck([10, 8, 10, 6, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.value() == 26
    assert next_state is None
    assert reward == REWARD_WIN
    assert done is True


def test_stand_draw_when_values_equal():
    deck = FixedDeck([10, 8, 10, 8])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert reward == REWARD_DRAW
    assert done is True


# =========================
# Double behavior
# =========================


def test_double_draws_one_card_sets_bet_to_two_and_wins_double_reward():
    # Player: 5,6 then doubles and draws 10 -> 21.
    # Dealer: 10,6 then draws 10 -> bust.
    deck = FixedDeck([5, 6, 10, 6, 10, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [5, 6, 10]
    assert game.hand_bets == [2]
    assert game.dealer_hand.cards == [10, 6, 10]
    assert next_state is None
    assert reward == 2 * REWARD_WIN
    assert done is True


def test_double_loss_returns_negative_two():
    # Player: 5,6 then doubles and draws 2 -> 13.
    # Dealer: 10,8 -> 18.
    deck = FixedDeck([5, 6, 10, 8, 2])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [5, 6, 2]
    assert game.hand_bets == [2]
    assert next_state is None
    assert reward == 2 * REWARD_LOSS
    assert done is True


def test_illegal_double_after_hit_raises_error():
    deck = FixedDeck([10, 2, 9, 7, 4])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.HIT)

    with pytest.raises(ValueError):
        game.step(Action.DOUBLE)


# =========================
# Split behavior
# =========================


def test_split_creates_two_split_hands_and_preserves_bets():
    # Player 8,8 splits into [8,3] and [8,2].
    deck = FixedDeck([8, 8, 10, 7, 3, 2])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[8, 3], [8, 2]]
    assert game.hand_bets == [1, 1]
    assert game.is_split_hand == [True, True]
    assert game.is_split_aces_hand == [False, False]
    assert game.active_hand_index == 0

    assert next_state.player_value == 11
    assert next_state.can_double is True
    assert next_state.can_split is False
    assert next_state.is_split_hand is True
    assert reward == 0
    assert done is False


def test_stand_after_split_moves_to_second_hand_without_dealer_playing():
    deck = FixedDeck([8, 8, 10, 6, 3, 2])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert game.active_hand_index == 1
    assert game.dealer_hand.cards == [10, 6]
    assert next_state.player_value == 10
    assert next_state.is_split_hand is True
    assert reward == 0
    assert done is False


def test_split_total_reward_sums_both_hands_after_dealer_plays_once():
    # Hands after split: [8,10] = 18 and [8,9] = 17.
    # Dealer: 10,6 then draws 10 -> bust.
    deck = FixedDeck([8, 8, 10, 6, 10, 9, 10])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    next_state, reward, done = game.step(Action.STAND)

    assert reward == 0
    assert done is False
    assert next_state.player_value == 17

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [10, 6, 10]
    assert next_state is None
    assert reward == 2
    assert done is True


def test_hit_bust_after_split_moves_to_next_hand_not_game_over():
    # First split hand [10,9] hits 5 and busts.
    # Second split hand [10,8] should become active.
    deck = FixedDeck([10, 10, 9, 7, 9, 8, 5])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [10, 9, 5]
    assert game.active_hand_index == 1
    assert next_state.player_value == 18
    assert reward == 0
    assert done is False


def test_double_after_split_counts_only_that_hand_as_double_bet():
    # Split: [8,3] and [8,2].
    # Double first hand: [8,3,10] = 21, bet 2.
    # Stand second hand: [8,2] = 10, bet 1.
    # Dealer busts, so total reward = +2 + +1 = +3.
    deck = FixedDeck([8, 8, 10, 6, 3, 2, 10, 10])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [8, 3, 10]
    assert game.hand_bets == [2, 1]
    assert game.active_hand_index == 1
    assert reward == 0
    assert done is False
    assert next_state.player_value == 10

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert reward == 3
    assert done is True


def test_split_not_available_for_non_pair():
    deck = FixedDeck([10, 9, 8, 7])
    game = BlackjackGame(deck)
    game.reset()

    assert Action.SPLIT not in game.available_actions()

    with pytest.raises(ValueError):
        game.step(Action.SPLIT)


# =========================
# Split aces behavior
# =========================


def test_split_aces_auto_moves_to_second_ace_hand_and_disallows_hit_double_split():
    # Current config has ALLOW_HIT_SPLIT_ACES = False.
    # Split aces become [A,10] and [A,9]. First hand is auto-finished.
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[1, 10], [1, 9]]
    assert game.active_hand_index == 1
    assert game.is_split_hand == [True, True]
    assert game.is_split_aces_hand == [True, True]

    assert next_state.player_value == 20
    assert next_state.can_double is False
    assert next_state.can_split is False
    assert next_state.is_split_hand is True
    assert game.available_actions() == [Action.STAND]
    assert reward == 0
    assert done is False


def test_standing_second_split_ace_hand_finishes_round():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert reward == 2
    assert done is True


# =========================
# Error handling
# =========================


def test_step_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_step_after_game_done_raises_error():
    deck = FixedDeck([10, 8, 10, 7])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_invalid_action_type_raises_error():
    deck = FixedDeck([10, 8, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    with pytest.raises(ValueError):
        game.step("hit")

# =========================
# Additional edge-case coverage
# =========================


def test_game_state_repr_includes_new_q_learning_fields():
    state = GameState(
        player_value=12,
        dealer_upcard=6,
        usable_ace=False,
        can_double=True,
        can_split=True,
        is_split_hand=False,
    )

    text = repr(state)

    assert "player_value=12" in text
    assert "dealer_upcard=6" in text
    assert "usable_ace=False" in text
    assert "can_double=True" in text
    assert "can_split=True" in text
    assert "is_split_hand=False" in text


def test_current_hand_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.current_hand()


def test_split_keeps_metadata_lists_aligned_after_multiple_splits():
    # Initial [8, 8] splits into [8, 8] and [8, 4].
    # Then the first split hand [8, 8] is resplit into [8, 2] and [8, 3].
    deck = FixedDeck([8, 8, 10, 7, 8, 4, 2, 3])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[8, 2], [8, 3], [8, 4]]
    assert game.hand_bets == [1, 1, 1]
    assert game.is_split_hand == [True, True, True]
    assert game.is_split_aces_hand == [False, False, False]
    assert len(game.player_hands) == len(game.hand_bets)
    assert len(game.player_hands) == len(game.is_split_hand)
    assert len(game.player_hands) == len(game.is_split_aces_hand)
    assert game.active_hand_index == 0


def test_all_split_hands_bust_so_dealer_does_not_draw():
    # Dealer starts at 15. If the dealer incorrectly plays after both hands bust,
    # this test would need another card and fail with IndexError.
    deck = FixedDeck([10, 10, 8, 7, 9, 9, 5, 5])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [10, 9, 5]
    assert next_state.player_value == 19
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[1].cards == [10, 9, 5]
    assert game.dealer_hand.cards == [8, 7]
    assert next_state is None
    assert reward == 2 * REWARD_LOSS
    assert done is True


def test_dealer_plays_if_at_least_one_split_hand_is_not_bust():
    # First split hand busts. Second split hand stands on 18.
    # Dealer starts at 15 and should draw 2 to reach 17.
    deck = FixedDeck([10, 10, 8, 7, 9, 8, 5, 2])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [10, 9, 5]
    assert next_state.player_value == 18
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [8, 7, 2]
    assert next_state is None
    assert reward == 0  # first hand loses, second hand wins
    assert done is True


def test_split_aces_rejects_hit_double_and_split_steps():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    assert game.available_actions() == [Action.STAND]

    with pytest.raises(ValueError):
        game.step(Action.HIT)

    with pytest.raises(ValueError):
        game.step(Action.DOUBLE)

    with pytest.raises(ValueError):
        game.step(Action.SPLIT)


def test_double_that_busts_after_split_uses_doubled_loss_later():
    # Split: [9,3] and [9,8].
    # Double first hand: [9,3,10] busts, bet 2 => -2.
    # Stand second hand: dealer 16 draws 10 and busts, normal bet => +1.
    # Total reward = -1.
    deck = FixedDeck([9, 9, 10, 6, 3, 8, 10, 10])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [9, 3, 10]
    assert game.hand_bets == [2, 1]
    assert next_state.player_value == 17
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [10, 6, 10]
    assert next_state is None
    assert reward == -1
    assert done is True


def test_state_distinguishes_initial_hand_from_split_hand_with_same_value():
    # Initial state: [8,8] has value 16 and can split, but is not a split hand.
    # After split: first hand [8,8] still has value 16 and can split, but is a split hand.
    deck = FixedDeck([8, 8, 10, 7, 8, 4])
    game = BlackjackGame(deck)

    initial_state = game.reset()
    split_state, reward, done = game.step(Action.SPLIT)

    assert initial_state.player_value == split_state.player_value == 16
    assert initial_state.dealer_upcard == split_state.dealer_upcard == 10
    assert initial_state.can_split is True
    assert split_state.can_split is True
    assert initial_state.is_split_hand is False
    assert split_state.is_split_hand is True
    assert initial_state.as_tuple() != split_state.as_tuple()
    assert reward == 0
    assert done is False


def test_reset_after_split_round_clears_all_split_metadata():
    deck = FixedDeck([8, 8, 10, 7, 3, 4, 10, 6, 9, 8])
    game = BlackjackGame(deck)

    game.reset()
    game.step(Action.SPLIT)
    game.step(Action.STAND)
    game.step(Action.STAND)

    state = game.reset()

    assert game.player_hands[0].cards == [10, 6]
    assert game.dealer_hand.cards == [9, 8]
    assert game.active_hand_index == 0
    assert game.hand_bets == [1]
    assert game.is_split_hand == [False]
    assert game.is_split_aces_hand == [False]
    assert state.is_split_hand is False
    assert state.can_split is False


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


def test_blackjack_like_21_is_not_auto_finished_on_reset():
    deck = FixedDeck([1, 10, 10, 9])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.player_value == 21
    assert game.done is False
    assert game.available_actions() == [Action.HIT, Action.STAND, Action.DOUBLE]


def test_natural_21_can_stand_and_win_normally():
    deck = FixedDeck([1, 10, 10, 9])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert reward == REWARD_WIN
    assert done is True
