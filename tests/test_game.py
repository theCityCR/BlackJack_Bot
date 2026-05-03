"""
test_game.py

Pytest tests for the BlackjackGame environment.
"""

import pytest

from cards import Deck
from config import REWARD_DRAW, REWARD_LOSS, REWARD_WIN
from game import BLACKJACK_PAYOUT, Action, BlackjackGame, GameState


class FixedDeck(Deck):
    """
    Deterministic deck for testing.

    Cards are drawn from left to right.
    reset() is intentionally a no-op.
    """

    def __init__(self, cards):
        self.cards = list(cards)
        self.index = 0

    def draw_card(self):
        if self.index >= len(self.cards):
            raise IndexError("No more cards in FixedDeck.")

        card = self.cards[self.index]
        self.index += 1
        return card

    def reset(self):
        pass

    def cards_remaining(self):
        return len(self.cards) - self.index

    def get_count_vector(self):
        remaining_cards = self.cards[self.index:]
        counts = [0] * 10

        for card in remaining_cards:
            if card == 1:
                counts[0] += 1
            elif 2 <= card <= 9:
                counts[card - 1] += 1
            elif card == 10:
                counts[9] += 1
            else:
                raise ValueError(f"Invalid card value: {card}")

        return tuple(counts)


# =========================
# Reset / GameState
# =========================


def test_reset_creates_initial_state_with_q_learning_context():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)

    state = game.reset()

    assert isinstance(state, GameState)
    assert game.player_hands[0].cards == [10, 7]
    assert game.dealer_hand.cards == [9, 6]
    assert game.active_hand_index == 0
    assert game.done is False
    assert game.round_reward is None
    assert game.hand_rewards == [None]

    assert state.player_value == 17
    assert state.dealer_upcard == 9
    assert state.usable_ace is False
    assert state.can_double is True
    assert state.can_split is False
    assert state.is_split_hand is False
    assert state.active_hand_index == 0
    assert state.num_hands == 1
    assert state.count_vector == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_state_as_tuple_includes_split_context_and_count_vector():
    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=True,
        active_hand_index=1,
        num_hands=3,
        count_vector=(4, 4, 4, 4, 4, 4, 4, 4, 4, 16),
    )

    assert state.as_tuple() == (
        16,
        10,
        False,
        True,
        False,
        True,
        1,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        16,
    )


def test_game_state_has_default_count_vector_for_old_tests():
    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    assert state.count_vector == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert len(state.as_tuple()) == 18


def test_get_state_after_round_done_raises_error():
    deck = FixedDeck([10, 8, 10, 8])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.get_state()


# =========================
# Count vector behavior
# =========================


def test_reset_state_includes_remaining_count_vector():
    deck = FixedDeck([10, 7, 9, 6, 5, 4, 3])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.count_vector == (
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    )


def test_hit_updates_count_vector():
    deck = FixedDeck([10, 2, 9, 7, 5, 4, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert done is False
    assert reward == 0
    assert next_state.count_vector == (
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_split_updates_count_vector_after_dealing_to_split_hands():
    deck = FixedDeck([8, 8, 10, 6, 3, 4, 5, 2])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert done is False
    assert reward == 0
    assert [hand.cards for hand in game.player_hands] == [[8, 3], [8, 4]]
    assert next_state.count_vector == (
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    )


def test_double_consumes_one_card_and_terminal_state_has_no_next_state():
    deck = FixedDeck([9, 2, 10, 7, 5, 4, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert next_state is None
    assert done is True
    assert game.player_hand.cards == [9, 2, 5]
    assert game.deck.get_count_vector() == (
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_game_reset_does_not_reshuffle_finite_deck_when_above_threshold():
    deck = Deck(num_decks=2, shuffle=False, reshuffle_threshold=26)
    game = BlackjackGame(deck)

    state_1 = game.reset()
    cards_after_first_reset = deck.cards_remaining()

    game.step(Action.STAND)

    state_2 = game.reset()
    cards_after_second_reset = deck.cards_remaining()

    assert state_1 is not None
    assert state_2 is not None
    assert cards_after_second_reset < cards_after_first_reset
    assert cards_after_second_reset != 52 * deck.num_decks


def test_game_reset_reshuffles_finite_deck_when_at_or_below_threshold():
    deck = Deck(num_decks=2, shuffle=False, reshuffle_threshold=100)
    game = BlackjackGame(deck)

    game.reset()
    assert deck.cards_remaining() == 52 * deck.num_decks - 4

    game.step(Action.STAND)

    game.reset()

    # At the start of the second reset, the shoe is at or below threshold,
    # so it force-resets to full shoe, then deals 4 cards.
    assert deck.cards_remaining() == 52 * deck.num_decks - 4


# =========================
# Immediate dealer blackjack
# =========================


def test_dealer_blackjack_ends_round_immediately_before_player_acts():
    deck = FixedDeck([10, 9, 1, 10])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state is None
    assert game.done is True
    assert game.initial_dealer_blackjack is True
    assert game.available_actions() == []
    assert game.hand_rewards == [REWARD_LOSS]
    assert game.round_reward == REWARD_LOSS
    assert game.active_hand_index == 1

    with pytest.raises(RuntimeError):
        game.step(Action.HIT)


def test_dealer_and_player_blackjack_immediate_push():
    deck = FixedDeck([1, 10, 1, 10])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state is None
    assert game.done is True
    assert game.hand_rewards == [REWARD_DRAW]
    assert game.round_reward == REWARD_DRAW


# =========================
# Hit / Stand / Dealer
# =========================


def test_hit_adds_card_and_continues_when_not_bust():
    deck = FixedDeck([10, 2, 9, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hand.cards == [10, 2, 5]
    assert next_state.player_value == 17
    assert next_state.can_double is False
    assert reward == 0
    assert done is False
    assert game.hand_rewards == [None]


def test_hit_bust_assigns_that_hand_reward_immediately():
    deck = FixedDeck([10, 9, 8, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert next_state is None
    assert done is True
    assert reward == REWARD_LOSS
    assert game.hand_rewards == [REWARD_LOSS]
    assert game.round_reward == REWARD_LOSS
    assert game.dealer_hand.cards == [8, 7]


def test_stand_dealer_draws_until_17_or_more():
    deck = FixedDeck([10, 8, 9, 2, 5, 3])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert game.dealer_hand.cards == [9, 2, 5, 3]
    assert game.dealer_hand.value() == 19
    assert reward == REWARD_LOSS
    assert game.hand_rewards == [REWARD_LOSS]


def test_dealer_stands_on_soft_17():
    deck = FixedDeck([10, 8, 1, 6])
    game = BlackjackGame(deck)
    game.reset()

    game.step(Action.STAND)

    assert game.dealer_hand.cards == [1, 6]
    assert game.dealer_hand.value() == 17
    assert game.dealer_hand.usable_ace() is True


# =========================
# Natural blackjack
# =========================


def test_player_natural_blackjack_allows_only_stand():
    deck = FixedDeck([1, 10, 9, 7])
    game = BlackjackGame(deck)

    state = game.reset()

    assert state.player_value == 21
    assert state.can_double is False
    assert state.can_split is False
    assert game.available_actions() == [Action.STAND]


@pytest.mark.parametrize("illegal_action", [Action.HIT, Action.DOUBLE, Action.SPLIT])
def test_player_natural_blackjack_rejects_non_stand_actions(illegal_action):
    deck = FixedDeck([1, 10, 9, 7])
    game = BlackjackGame(deck)
    game.reset()

    with pytest.raises(ValueError):
        game.step(illegal_action)


def test_player_natural_blackjack_pays_three_to_two():
    deck = FixedDeck([1, 10, 9, 7])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert reward == BLACKJACK_PAYOUT
    assert game.hand_rewards == [BLACKJACK_PAYOUT]
    assert game.round_reward == BLACKJACK_PAYOUT
    assert game.dealer_hand.cards == [9, 7]


def test_non_natural_21_pays_normal_win_not_blackjack_payout():
    deck = FixedDeck([10, 6, 10, 7, 5])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert next_state.player_value == 21
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert reward == REWARD_WIN
    assert reward != BLACKJACK_PAYOUT
    assert game.hand_rewards == [REWARD_WIN]


# =========================
# Double
# =========================


def test_double_draws_one_card_finishes_hand_and_doubles_win_reward():
    deck = FixedDeck([9, 2, 10, 7, 10])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert next_state is None
    assert done is True
    assert game.player_hand.cards == [9, 2, 10]
    assert game.hand_bets == [2]
    assert reward == 2 * REWARD_WIN
    assert game.hand_rewards == [2 * REWARD_WIN]
    assert game.round_reward == 2 * REWARD_WIN


def test_double_is_not_available_after_hit():
    deck = FixedDeck([5, 5, 10, 7, 2])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.HIT)

    assert done is False
    assert next_state.can_double is False
    assert Action.DOUBLE not in game.available_actions()

    with pytest.raises(ValueError):
        game.step(Action.DOUBLE)


# =========================
# Split and per-hand rewards
# =========================


def test_split_creates_two_hands_and_updates_state_context():
    deck = FixedDeck([8, 8, 10, 6, 3, 4])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[8, 3], [8, 4]]
    assert game.hand_bets == [1, 1]
    assert game.hand_rewards == [None, None]
    assert game.split_aces == [False, False]
    assert game.active_hand_index == 0

    assert next_state.is_split_hand is True
    assert next_state.player_value == 11
    assert next_state.active_hand_index == 0
    assert next_state.num_hands == 2
    assert reward == 0
    assert done is False


def test_stand_after_split_moves_to_next_hand_without_scoring_yet():
    deck = FixedDeck([8, 8, 10, 6, 3, 4])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.STAND)

    assert game.active_hand_index == 1
    assert game.dealer_hand.cards == [10, 6]
    assert game.hand_rewards == [None, None]
    assert next_state.is_split_hand is True
    assert next_state.player_value == 12
    assert next_state.active_hand_index == 1
    assert next_state.num_hands == 2
    assert reward == 0
    assert done is False


def test_split_final_reward_is_stored_per_hand_and_round_reward_is_sum():
    deck = FixedDeck([8, 8, 10, 6, 10, 9, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)
    game.step(Action.STAND)

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert game.hand_rewards == [REWARD_WIN, REWARD_WIN]
    assert game.round_reward == 2 * REWARD_WIN
    assert reward == game.round_reward


def test_busted_split_hand_gets_own_loss_before_round_ends():
    deck = FixedDeck([10, 10, 9, 8, 2, 3, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    next_state, reward, done = game.step(Action.HIT)

    assert game.player_hands[0].cards == [10, 2, 10]
    assert game.player_hands[0].is_bust() is True
    assert game.hand_rewards == [REWARD_LOSS, None]
    assert game.active_hand_index == 1
    assert next_state.is_split_hand is True
    assert reward == 0
    assert done is False


def test_double_after_split_uses_only_that_hand_bet_in_final_hand_rewards():
    deck = FixedDeck([8, 8, 10, 7, 3, 2, 10])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.SPLIT)

    assert Action.DOUBLE in game.available_actions()

    next_state, reward, done = game.step(Action.DOUBLE)

    assert game.hand_bets == [2, 1]
    assert game.hand_rewards == [None, None]
    assert game.active_hand_index == 1
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert game.hand_rewards == [2 * REWARD_WIN, REWARD_LOSS]
    assert game.round_reward == 1
    assert reward == 1


def test_split_ace_ten_is_not_natural_blackjack_payout():
    deck = FixedDeck([1, 1, 10, 7, 10, 9])
    game = BlackjackGame(deck)
    game.reset()

    next_state, reward, done = game.step(Action.SPLIT)

    assert [hand.cards for hand in game.player_hands] == [[1, 10], [1, 9]]
    assert game.split_aces == [True, True]
    assert game.active_hand_index == 1
    assert next_state.is_split_hand is True
    assert game.available_actions() == [Action.STAND]
    assert reward == 0
    assert done is False

    next_state, reward, done = game.step(Action.STAND)

    assert next_state is None
    assert done is True
    assert game.hand_rewards == [REWARD_WIN, REWARD_WIN]
    assert game.round_reward == 2 * REWARD_WIN
    assert reward != BLACKJACK_PAYOUT + REWARD_WIN


def test_split_is_not_available_for_non_pair():
    deck = FixedDeck([8, 7, 10, 6])
    game = BlackjackGame(deck)
    game.reset()

    assert Action.SPLIT not in game.available_actions()

    with pytest.raises(ValueError):
        game.step(Action.SPLIT)


# =========================
# Errors / reset
# =========================


def test_available_actions_after_game_done_is_empty():
    deck = FixedDeck([10, 7, 10, 7])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    assert game.available_actions() == []


def test_reset_after_finished_game_clears_rewards_and_metadata():
    deck = FixedDeck([10, 7, 10, 7, 5, 5, 9, 8])
    game = BlackjackGame(deck)

    game.reset()
    game.step(Action.STAND)
    state = game.reset()

    assert game.done is False
    assert game.round_reward is None
    assert game.hand_rewards == [None]
    assert game.player_hands[0].cards == [5, 5]
    assert game.dealer_hand.cards == [9, 8]
    assert game.active_hand_index == 0
    assert game.hand_bets == [1]
    assert game.split_aces == [False]
    assert state.player_value == 10
    assert state.can_split is True


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


def test_invalid_action_raises_value_error():
    deck = FixedDeck([10, 7, 9, 6])
    game = BlackjackGame(deck)
    game.reset()

    with pytest.raises(ValueError):
        game.step("invalid")