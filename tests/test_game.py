"""
test_game.py

Tests for game.py and integration with cards.py.

These tests cover the original hit/stand behavior plus the newer double and
split behavior.
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

    assert game.player_hands[0].cards == [10, 7]
    assert game.player_hand.cards == [10, 7]
    assert game.dealer_hand.cards == [9, 6]
    assert isinstance(state, GameState)
    assert state.player_value == 17
    assert state.dealer_upcard == 9
    assert state.usable_ace is False
    assert state.can_double is True
    assert state.can_split is False
    assert state.active_hand_index == 0
    assert state.num_hands == 1


def test_get_state_returns_visible_state_only():
    deck = FixedDeck([1, 7, 10, 6])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.player_value == 18
    assert state.dealer_upcard == 10
    assert state.usable_ace is True


def test_state_as_tuple():
    state = GameState(
        player_value=18,
        dealer_upcard=10,
        usable_ace=True,
        can_double=True,
        can_split=False,
        active_hand_index=0,
        num_hands=1,
    )

    assert state.as_tuple() == (18, 10, True, True, False, 0, 1)


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
    assert next_state.can_double is False
    assert next_state.can_split is False
    assert reward == 0
    assert done is False


def test_hit_busts_single_hand_and_ends_game():
    deck = FixedDeck([10, 9, 8, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [10, 9, 5]
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


# =========================
# Double Behavior
# =========================


def test_double_draws_one_card_ends_single_hand_and_doubles_win_reward():
    # Player: 10, 1 -> 11. Dealer: 10, 7 -> 17. Double card: 10 -> 21.
    deck = FixedDeck([10, 1, 10, 7, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [10, 1, 10]
    assert game.player_hands[0].value() == 21
    assert game.hand_bets == [2]
    assert reward == 2 * REWARD_WIN
    assert next_state is None
    assert done is True


def test_double_draws_one_card_ends_single_hand_and_doubles_loss_reward():
    # Player: 10, 6 -> 16. Dealer: 10, 7 -> 17. Double card: 10 -> bust.
    deck = FixedDeck([10, 6, 10, 7, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [10, 6, 10]
    assert game.player_hands[0].is_bust() is True
    assert game.hand_bets == [2]
    assert reward == 2 * REWARD_LOSS
    assert next_state is None
    assert done is True


def test_double_is_not_available_after_hit():
    deck = FixedDeck([5, 4, 10, 7, 2])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.HIT)

    assert Action.DOUBLE not in game.available_actions()
    with pytest.raises(ValueError):
        game.step(Action.DOUBLE)


# =========================
# Split Behavior
# =========================


def test_split_creates_two_hands_and_continues_on_first_hand():
    # Player: 8, 8. Dealer: 10, 7. Split draw cards: 3 and 2.
    deck = FixedDeck([8, 8, 10, 7, 3, 2])
    game = BlackjackGame(deck)
    state = game.reset()

    assert state.can_split is True
    assert Action.SPLIT in game.available_actions()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[8, 3], [8, 2]]
    assert game.hand_bets == [1, 1]
    assert game.active_hand_index == 0
    assert next_state.player_value == 11
    assert next_state.active_hand_index == 0
    assert next_state.num_hands == 2
    assert reward == 0
    assert done is False


def test_stand_after_split_moves_to_next_hand_without_dealer_playing_yet():
    deck = FixedDeck([8, 8, 10, 7, 3, 2])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert game.active_hand_index == 1
    assert next_state.player_value == 10
    assert next_state.active_hand_index == 1
    assert reward == 0
    assert done is False
    assert game.dealer_hand.cards == [10, 7]


def test_split_total_reward_sums_both_hands():
    # Split 8s into 11 and 20. Dealer has 17.
    # Hand 1 loses: -1. Hand 2 wins: +1. Total: 0.
    deck = FixedDeck([8, 8, 10, 7, 3, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)
    assert reward == 0
    assert done is False
    assert next_state.active_hand_index == 1

    next_state, reward, done = game.step(Action.STAND)

    assert reward == 0
    assert next_state is None
    assert done is True
    assert game.done is True


def test_hit_bust_after_split_moves_to_next_hand_not_game_over():
    # Split 8s into [8, 10] and [8, 2]. Hit first hand with 10 -> bust.
    deck = FixedDeck([8, 8, 10, 7, 10, 2, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [8, 10, 10]
    assert game.player_hands[0].is_bust() is True
    assert game.active_hand_index == 1
    assert next_state.active_hand_index == 1
    assert reward == 0
    assert done is False


def test_double_after_split_finishes_current_hand_and_later_counts_double_reward():
    # Split 8s into [8, 3] and [8, 2].
    # Double first hand: draw 10 -> 21, bet 2.
    # Stand second hand: 10 loses to dealer 17.
    # Total = 2 * win + 1 * loss = 1.
    deck = FixedDeck([8, 8, 10, 7, 3, 2, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [8, 3, 10]
    assert game.hand_bets == [2, 1]
    assert game.active_hand_index == 1
    assert next_state.active_hand_index == 1
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert reward == 1
    assert next_state is None
    assert done is True


def test_split_aces_get_one_card_each_and_first_hand_auto_finishes():
    # Current config has ALLOW_HIT_SPLIT_ACES = False.
    # Split aces into [A, 10] and [A, 9]. First hand auto-finishes.
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[1, 10], [1, 9]]
    assert game.active_hand_index == 1
    assert next_state.active_hand_index == 1
    assert next_state.player_value == 20
    assert Action.HIT not in game.available_actions()
    assert Action.DOUBLE not in game.available_actions()
    assert Action.SPLIT not in game.available_actions()
    assert game.available_actions() == [Action.STAND]
    assert reward == 0
    assert done is False


def test_split_not_available_for_non_pair():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)
    state = game.reset()

    assert state.can_split is False
    assert Action.SPLIT not in game.available_actions()

    with pytest.raises(ValueError):
        game.step(Action.SPLIT)


# =========================
# Available Actions
# =========================


def test_available_actions_initial_two_card_non_pair():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)
    game.reset()

    assert game.available_actions() == [Action.HIT, Action.STAND, Action.DOUBLE]


def test_available_actions_initial_pair():
    deck = FixedDeck([8, 8, 9, 6])
    game = BlackjackGame(deck)
    game.reset()

    assert game.available_actions() == [
        Action.HIT,
        Action.STAND,
        Action.DOUBLE,
        Action.SPLIT,
    ]


def test_available_actions_after_game_done():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.available_actions() == []


# =========================
# Other Existing Behaviors
# =========================


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
    assert game.player_hands[0].cards == [5, 5]
    assert game.dealer_hand.cards == [9, 8]
    assert state.player_value == 10
    assert state.dealer_upcard == 9
    assert state.can_split is True


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


# =========================
# Additional Edge Cases
# =========================


def test_available_actions_before_reset_raises_error():
    game = BlackjackGame()

    with pytest.raises(RuntimeError):
        game.available_actions()


def test_player_hand_property_before_reset_is_none():
    game = BlackjackGame()

    assert game.player_hand is None


def test_current_hand_after_game_done_raises_error():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.current_hand()


def test_player_hand_property_after_game_done_returns_last_hand():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    assert game.player_hand.cards == [10, 7]


def test_state_repr_contains_new_fields():
    state = GameState(
        player_value=12,
        dealer_upcard=6,
        usable_ace=False,
        can_double=True,
        can_split=True,
        active_hand_index=1,
        num_hands=2,
    )

    text = repr(state)

    assert "player_value=12" in text
    assert "dealer_upcard=6" in text
    assert "usable_ace=False" in text
    assert "can_double=True" in text
    assert "can_split=True" in text
    assert "active_hand_index=1" in text
    assert "num_hands=2" in text


def test_get_state_after_split_reports_second_hand_after_first_stands():
    deck = FixedDeck([8, 8, 10, 7, 3, 4])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert reward == 0
    assert done is False
    assert next_state.as_tuple() == (12, 10, False, True, False, 1, 2)


def test_hit_after_split_first_hand_continues_on_same_hand_if_not_bust():
    deck = FixedDeck([8, 8, 10, 7, 2, 3, 4])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.HIT)

    assert game.active_hand_index == 0
    assert game.player_hands[0].cards == [8, 2, 4]
    assert next_state.active_hand_index == 0
    assert next_state.player_value == 14
    assert reward == 0
    assert done is False


def test_double_bust_after_split_moves_to_next_hand_without_dealer_playing():
    # Split 9s into [9, 2] and [9, 3]. Double first hand with 10 -> 21,
    # not bust, actually. Use [9, 9] split draw [8, 3], then double 8+9+10 = 27.
    deck = FixedDeck([9, 9, 10, 7, 8, 3, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.player_hands[0].cards == [9, 8, 10]
    assert game.player_hands[0].is_bust() is True
    assert game.hand_bets == [2, 1]
    assert game.active_hand_index == 1
    assert game.dealer_hand.cards == [10, 7]
    assert next_state.active_hand_index == 1
    assert reward == 0
    assert done is False


def test_all_split_hands_bust_dealer_does_not_draw_more_cards():
    # Both hands bust. Dealer starts at 12, but should not draw because all
    # player hands are already bust.
    deck = FixedDeck([10, 10, 10, 2, 9, 8, 5, 6])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.HIT)
    assert done is False
    assert game.active_hand_index == 1

    next_state, reward, done = game.step(Action.HIT)

    assert game.dealer_hand.cards == [10, 2]
    assert reward == 2 * REWARD_LOSS
    assert next_state is None
    assert done is True


def test_dealer_plays_once_after_last_split_hand_finishes():
    deck = FixedDeck([8, 8, 10, 2, 2, 3, 5])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    game.step(Action.STAND)
    next_state, reward, done = game.step(Action.STAND)

    assert game.dealer_hand.cards == [10, 2, 5]
    assert game.dealer_hand.value() == 17
    assert next_state is None
    assert done is True


def test_split_metadata_and_bets_reset_on_new_game():
    deck = FixedDeck([8, 8, 10, 7, 3, 2, 10, 10, 7, 10, 6])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)
    game.step(Action.DOUBLE)
    game.step(Action.STAND)

    state = game.reset()

    assert game.player_hands[0].cards == [10, 7]
    assert game.dealer_hand.cards == [10, 6]
    assert game.hand_bets == [1]
    assert game.split_aces == [False]
    assert game.active_hand_index == 0
    assert state.num_hands == 1


def test_resplit_available_when_pair_created_after_split():
    # Initial 8,8. Split draws another 8 to first hand and 3 to second hand.
    deck = FixedDeck([8, 8, 10, 7, 8, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert game.player_hands[0].cards == [8, 8]
    assert next_state.can_split is True
    assert Action.SPLIT in game.available_actions()
    assert reward == 0
    assert done is False


def test_split_not_available_when_max_player_hands_reached():
    # Build up to 4 hands, each time the active hand remains splittable.
    deck = FixedDeck([
        8, 8, 10, 7,  # initial player/dealer
        8, 8,          # first split -> 2 hands, active hand still [8, 8]
        8, 8,          # second split -> 3 hands, active hand still [8, 8]
        8, 8,          # third split -> 4 hands, active hand still [8, 8]
    ])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.SPLIT)
    game.step(Action.SPLIT)
    game.step(Action.SPLIT)

    assert len(game.player_hands) == 4
    assert game.get_state().can_split is False
    assert Action.SPLIT not in game.available_actions()


def test_split_aces_second_hand_stand_finishes_round():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert reward == 2 * REWARD_WIN


def test_cannot_hit_split_aces_when_rule_disallows_it():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    with pytest.raises(ValueError):
        game.step(Action.HIT)


def test_cannot_double_split_aces_when_rule_disallows_hit_split_aces():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    with pytest.raises(ValueError):
        game.step(Action.DOUBLE)


def test_cannot_resplit_split_aces_when_rule_disallows_hit_split_aces():
    # Second split-ace hand is [A, A], but split aces cannot be acted on
    # except by standing under the current rule settings.
    deck = FixedDeck([1, 1, 10, 7, 10, 1])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    assert game.player_hands[1].cards == [1, 1]
    assert Action.SPLIT not in game.available_actions()

    with pytest.raises(ValueError):
        game.step(Action.SPLIT)


def test_manual_total_reward_helper_before_dealer_busts():
    deck = FixedDeck([10, 9, 10, 7])
    game = BlackjackGame(deck)
    game.reset()

    assert game._compare_hands() == REWARD_WIN


def test_compare_bust_hand_to_dealer_returns_loss_even_before_dealer_plays():
    deck = FixedDeck([10, 9, 10, 7, 5])
    game = BlackjackGame(deck)
    game.reset()
    game.player_hands[0].hit()

    assert game.player_hands[0].is_bust() is True
    assert game._compare_hands() == REWARD_LOSS
