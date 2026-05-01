"""
game.py

Blackjack game environment.

Supported player actions:
- Hit
- Stand
- Double
- Split

Dealer follows a fixed policy: hit until 17+.
The deck abstraction is currently infinite.
"""

from enum import Enum
from typing import List, Optional, Tuple

from cards import Deck, Hand
from config import (
    ALLOW_DOUBLE_AFTER_SPLIT,
    ALLOW_HIT_SPLIT_ACES,
    ALLOW_RESPLIT,
    DEALER_STAND_THRESHOLD,
    DOUBLE_REWARD_MULTIPLIER,
    MAX_PLAYER_HANDS,
    REWARD_DRAW,
    REWARD_LOSS,
    REWARD_WIN,
)


class Action(Enum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3


class GameState:
    """
    Public game state given to an agent.

    The state describes the current active hand, not every player hand.
    The extra boolean fields reduce state aliasing for Q-learning.
    """

    def __init__(
        self,
        player_value: int,
        dealer_upcard: int,
        usable_ace: bool,
        can_double: bool,
        can_split: bool,
        is_split_hand: bool,
    ):
        self.player_value = player_value
        self.dealer_upcard = dealer_upcard
        self.usable_ace = usable_ace
        self.can_double = can_double
        self.can_split = can_split
        self.is_split_hand = is_split_hand

    def as_tuple(self) -> Tuple[int, int, bool, bool, bool, bool]:
        """
        Useful as a Q-table key.
        """
        return (
            self.player_value,
            self.dealer_upcard,
            self.usable_ace,
            self.can_double,
            self.can_split,
            self.is_split_hand,
        )

    def __repr__(self):
        return (
            "GameState("
            f"player_value={self.player_value}, "
            f"dealer_upcard={self.dealer_upcard}, "
            f"usable_ace={self.usable_ace}, "
            f"can_double={self.can_double}, "
            f"can_split={self.can_split}, "
            f"is_split_hand={self.is_split_hand})"
        )


class BlackjackGame:
    """
    Blackjack environment.

    Agent interaction pattern:

        game = BlackjackGame()
        state = game.reset()

        done = False
        while not done:
            available_actions = game.available_actions()
            action = agent.choose_action(state, available_actions)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    Notes:
    - Double gives one card, immediately finishes the current hand, and doubles
      that hand's final reward.
    - Split turns the current hand into two hands. The game then plays each hand
      from left to right before the dealer plays.
    """

    def __init__(self, deck: Optional[Deck] = None):
        self.deck = deck if deck is not None else Deck()
        self.player_hands: List[Hand] = []
        self.dealer_hand: Optional[Hand] = None
        self.active_hand_index = 0
        self.hand_bets: List[int] = []
        self.is_split_hand: List[bool] = []
        self.is_split_aces_hand: List[bool] = []
        self.done = False

    @property
    def player_hand(self) -> Optional[Hand]:
        """
        Backward-compatible access to the current active hand.
        """
        if not self.player_hands:
            return None
        if self.active_hand_index >= len(self.player_hands):
            return self.player_hands[-1]
        return self.player_hands[self.active_hand_index]

    def reset(self) -> GameState:
        """
        Start a new round.
        """
        self.deck.reset()

        self.player_hands = [Hand(self.deck)]
        self.dealer_hand = Hand(self.deck)
        self.active_hand_index = 0
        self.hand_bets = [1]
        self.is_split_hand = [False]
        self.is_split_aces_hand = [False]
        self.done = False

        return self.get_state()

    def get_state(self) -> GameState:
        """
        Return the state visible to the agent.
        """
        if not self.player_hands or self.dealer_hand is None:
            raise RuntimeError("Game has not been reset yet.")

        hand = self.current_hand()

        return GameState(
            player_value=hand.value(),
            dealer_upcard=self.dealer_hand.cards[0],
            usable_ace=hand.usable_ace(),
            can_double=self._can_double_current_hand(),
            can_split=self._can_split_current_hand(),
            is_split_hand=self.is_split_hand[self.active_hand_index],
        )

    def current_hand(self) -> Hand:
        """
        Return the current active player hand.
        """
        if not self.player_hands:
            raise RuntimeError("Game has not been reset yet.")

        if self.active_hand_index >= len(self.player_hands):
            raise RuntimeError("There is no active hand.")

        return self.player_hands[self.active_hand_index]

    def step(self, action: Action) -> Tuple[Optional[GameState], int, bool]:
        """
        Apply one player action.

        Returns:
            next_state, reward, done

        If done is True, next_state is None.
        """
        if self.done:
            raise RuntimeError("Game is already done. Call reset() to start a new game.")

        if not self.player_hands or self.dealer_hand is None:
            raise RuntimeError("Game has not been reset yet.")

        if not isinstance(action, Action):
            raise ValueError(f"Unknown action: {action}")

        if action not in self.available_actions():
            raise ValueError(f"Action is not currently available: {action}")

        if action == Action.HIT:
            return self._player_hit()

        if action == Action.STAND:
            return self._player_stand()

        if action == Action.DOUBLE:
            return self._player_double()

        if action == Action.SPLIT:
            return self._player_split()

        raise ValueError(f"Unknown action: {action}")

    def _player_hit(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player hits the current hand.
        """
        hand = self.current_hand()
        hand.hit()

        if hand.is_bust():
            return self._finish_current_hand()

        return self.get_state(), 0, False

    def _player_stand(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player stands on the current hand.
        """
        return self._finish_current_hand()

    def _player_double(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player doubles the current hand.
        """
        hand = self.current_hand()
        hand.hit()
        self.hand_bets[self.active_hand_index] *= DOUBLE_REWARD_MULTIPLIER

        return self._finish_current_hand()

    def _player_split(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player splits the current hand into two hands.
        """
        hand = self.current_hand()
        first_hand, second_hand = hand.split()

        split_aces = hand.cards[0] == 1
        current_bet = self.hand_bets[self.active_hand_index]

        self.player_hands[self.active_hand_index] = first_hand
        self.player_hands.insert(self.active_hand_index + 1, second_hand)

        self.hand_bets[self.active_hand_index] = current_bet
        self.hand_bets.insert(self.active_hand_index + 1, current_bet)

        self.is_split_hand[self.active_hand_index] = True
        self.is_split_hand.insert(self.active_hand_index + 1, True)

        self.is_split_aces_hand[self.active_hand_index] = split_aces
        self.is_split_aces_hand.insert(self.active_hand_index + 1, split_aces)

        # Many blackjack tables allow only one card per split ace.
        # If that rule is active, immediately finish the first split-ace hand.
        if split_aces and not ALLOW_HIT_SPLIT_ACES:
            return self._finish_current_hand()

        return self.get_state(), 0, False

    def _finish_current_hand(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Finish the current hand and advance to the next hand.

        If all player hands are finished, the dealer plays and the episode ends.
        If every player hand is already bust, the dealer does not need to draw.
        """
        self.active_hand_index += 1

        if self.active_hand_index < len(self.player_hands):
            return self.get_state(), 0, False

        if not self._all_player_hands_bust():
            self._play_dealer_turn()

        reward = self._total_reward()
        self.done = True

        return None, reward, True

    def _all_player_hands_bust(self) -> bool:
        """
        Whether every player hand is bust.
        """
        return all(hand.is_bust() for hand in self.player_hands)

    def _play_dealer_turn(self):
        """
        Dealer hits until reaching stand threshold.
        """
        while self.dealer_hand.value() < DEALER_STAND_THRESHOLD:
            self.dealer_hand.hit()

    def _total_reward(self) -> int:
        """
        Compare every player hand against the dealer and add the rewards.
        """
        total = 0

        for hand, bet in zip(self.player_hands, self.hand_bets):
            total += bet * self._compare_hand_to_dealer(hand)

        return total

    def _compare_hand_to_dealer(self, hand: Hand) -> int:
        """
        Compare one player hand to the dealer hand.
        """
        player_value = hand.value()
        dealer_value = self.dealer_hand.value()

        if hand.is_bust():
            return REWARD_LOSS

        if self.dealer_hand.is_bust():
            return REWARD_WIN

        if player_value > dealer_value:
            return REWARD_WIN

        if player_value < dealer_value:
            return REWARD_LOSS

        return REWARD_DRAW

    def _compare_hands(self) -> int:
        """
        Backward-compatible final comparison helper.
        """
        return self._total_reward()

    def _can_double_current_hand(self) -> bool:
        """
        Whether the current hand can double.
        """
        if self.done or not self.player_hands:
            return False

        hand = self.current_hand()

        if len(hand.cards) != 2:
            return False

        if self.is_split_aces_hand[self.active_hand_index] and not ALLOW_HIT_SPLIT_ACES:
            return False

        if self.is_split_hand[self.active_hand_index] and not ALLOW_DOUBLE_AFTER_SPLIT:
            return False

        return True

    def _can_split_current_hand(self) -> bool:
        """
        Whether the current hand can split.
        """
        if self.done or not self.player_hands:
            return False

        if len(self.player_hands) >= MAX_PLAYER_HANDS:
            return False

        if self.is_split_hand[self.active_hand_index] and not ALLOW_RESPLIT:
            return False

        if self.is_split_aces_hand[self.active_hand_index] and not ALLOW_HIT_SPLIT_ACES:
            return False

        return self.current_hand().can_split()

    def available_actions(self):
        """
        Return currently legal actions.
        """
        if self.done:
            return []

        if self.is_split_aces_hand[self.active_hand_index] and not ALLOW_HIT_SPLIT_ACES:
            return [Action.STAND]

        actions = [Action.HIT, Action.STAND]

        if self._can_double_current_hand():
            actions.append(Action.DOUBLE)

        if self._can_split_current_hand():
            actions.append(Action.SPLIT)

        return actions

    def render(self):
        """
        Print current game state for debugging.
        """
        if not self.player_hands or self.dealer_hand is None:
            print("Game has not started.")
            return

        for index, hand in enumerate(self.player_hands):
            marker = " <- active" if not self.done and index == self.active_hand_index else ""
            bet = self.hand_bets[index]
            split_text = ", split" if self.is_split_hand[index] else ""
            split_aces_text = ", split aces" if self.is_split_aces_hand[index] else ""
            print(
                f"Player hand {index + 1}: {hand}, "
                f"bet={bet}{split_text}{split_aces_text}{marker}"
            )

        if self.done:
            print(f"Dealer: {self.dealer_hand}")
        else:
            print(f"Dealer upcard: {self.dealer_hand.cards[0]}")
