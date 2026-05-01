"""
test_game.py

Pytest tests for the updated BlackjackGame environment.

Covers:
- reset/state fields
- hit/stand/double/split behavior
- dealer soft-17 behavior
- natural blackjack payout
- immediate dealer blackjack resolution
- per-hand final rewards after splitting

Run from the project root with:
    pytest tests/test_game.py -q

Assumes your project files are named:
    cards.py
    config.py
    game.py
"""

import pytest

from cards import Deck
from config import REWARD_DRAW, REWARD_LOSS, REWARD_WIN
from game import BLACKJACK_PAYOUT, Action, BlackjackGame, GameState


class FixedDeck(Deck):
    """
    Deterministic deck for testing.

    Cards are drawn from left to right. reset() is intentionally a no-op so a
    test can call game.reset() without rewinding the fixed card sequence.
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


# =========================
# Reset / GameState
# =========================


def test_reset_creates_initial_state_with_q_learning_context():
    # Player: 10,7. Dealer: 9,6.
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


def test_state_as_tuple_includes_split_context():
    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=True,
        active_hand_index=1,
        num_hands=3,
    )

    assert state.as_tuple() == (16, 10, False, True, False, True, 1, 3)


def test_get_state_after_round_done_raises_error():
    deck = FixedDeck([10, 8, 10, 8])
    game = BlackjackGame(deck)
    game.reset()
    game.step(Action.STAND)

    with pytest.raises(RuntimeError):
        game.get_state()


# =========================
# Immediate dealer blackjack
# =========================


def test_dealer_blackjack_ends_round_immediately_before_player_acts():
    # Player: 10,9 = 19. Dealer: A,10 = blackjack.
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
    # Player: A,10 = blackjack. Dealer: A,10 = blackjack.
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
    # Player: 10,2 hits 5 -> 17. Dealer: 9,7.
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
    # Player: 10,9 hits 5 -> bust. Dealer should not draw.
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
    # Player: 18. Dealer: 9,2 draws 5 then 3 -> 19.
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
    # Dealer A,6 is soft 17 and should not draw.
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
    # Player blackjack. Dealer 9,7.
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
    # Dealer should not draw because player natural blackjack already has known result.
    assert game.dealer_hand.cards == [9, 7]


def test_non_natural_21_pays_normal_win_not_blackjack_payout():
    # Player 10,6 hits 5 -> 21, then stands vs dealer 17.
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
    # Player 9,2 doubles, draws 10 -> 21. Dealer 10,7 = 17.
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
    # Player 8,8 split into [8,3], [8,4]. Dealer 10,6.
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
    # Player split 8,8 into [8,10] and [8,9]. Dealer 10,6 draws 10 and busts.
    # Both hands win: per-hand rewards [1, 1], round reward 2.
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
    # Split 10,10 into [10,2] and [10,3]. First hand hits 10 and busts.
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
    # Split 8,8 into [8,3] and [8,2].
    # First hand doubles and draws 10 -> 21, bet 2, wins +2 vs dealer 17.
    # Second hand stands on 10 and loses -1 vs dealer 17.
    # Per-hand rewards [2, -1], round reward +1.
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
    # Split aces into [A,10] and [A,9]. Both are forced to stand.
    # [A,10] after split is normal 21, not natural blackjack.
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
