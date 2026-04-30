"""
game.py

Blackjack game environment.

Simple version:
- Player can only hit or stand
- Dealer follows fixed policy: hit until 17+
- Infinite deck through Deck abstraction
"""

from enum import Enum
from typing import Optional, Tuple

from cards import Deck, Hand
from config import (
    DEALER_STAND_THRESHOLD,
    REWARD_WIN,
    REWARD_LOSS,
    REWARD_DRAW,
)


class Action(Enum):
    HIT = 0
    STAND = 1


class GameState:
    """
    Public game state given to an agent.
    """

    def __init__(self, player_value: int, dealer_upcard: int, usable_ace: bool):
        self.player_value = player_value
        self.dealer_upcard = dealer_upcard
        self.usable_ace = usable_ace

    def as_tuple(self) -> Tuple[int, int, bool]:
        """
        Useful as a Q-table key.
        """
        return (self.player_value, self.dealer_upcard, self.usable_ace)

    def __repr__(self):
        return (
            f"GameState("
            f"player_value={self.player_value}, "
            f"dealer_upcard={self.dealer_upcard}, "
            f"usable_ace={self.usable_ace})"
        )


class BlackjackGame:
    """
    Blackjack environment.

    Agent interaction pattern:

        game = BlackjackGame()
        state = game.reset()

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
    """

    def __init__(self, deck: Optional[Deck] = None):
        self.deck = deck if deck is not None else Deck()
        self.player_hand: Optional[Hand] = None
        self.dealer_hand: Optional[Hand] = None
        self.done = False

    def reset(self) -> GameState:
        """
        Start a new round.
        """
        self.deck.reset()

        self.player_hand = Hand(self.deck)
        self.dealer_hand = Hand(self.deck)
        self.done = False

        return self.get_state()

    def get_state(self) -> GameState:
        """
        Return the state visible to the agent.
        """
        if self.player_hand is None or self.dealer_hand is None:
            raise RuntimeError("Game has not been reset yet.")

        return GameState(
            player_value=self.player_hand.value(),
            dealer_upcard=self.dealer_hand.cards[0],
            usable_ace=self.player_hand.usable_ace(),
        )

    def step(self, action: Action) -> Tuple[Optional[GameState], int, bool]:
        """
        Apply one player action.

        Returns:
            next_state, reward, done

        If done is True, next_state is None.
        """
        if self.done:
            raise RuntimeError("Game is already done. Call reset() to start a new game.")

        if self.player_hand is None or self.dealer_hand is None:
            raise RuntimeError("Game has not been reset yet.")

        if action == Action.HIT:
            return self._player_hit()

        if action == Action.STAND:
            return self._player_stand()

        raise ValueError(f"Unknown action: {action}")

    def _player_hit(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player hits.
        """
        self.player_hand.hit()

        if self.player_hand.is_bust():
            self.done = True
            return None, REWARD_LOSS, True

        return self.get_state(), 0, False

    def _player_stand(self) -> Tuple[Optional[GameState], int, bool]:
        """
        Player stands, then dealer plays.
        """
        self._play_dealer_turn()

        reward = self._compare_hands()
        self.done = True

        return None, reward, True

    def _play_dealer_turn(self):
        """
        Dealer hits until reaching stand threshold.
        """
        while self.dealer_hand.value() < DEALER_STAND_THRESHOLD:
            self.dealer_hand.hit()

    def _compare_hands(self) -> int:
        """
        Compare final player and dealer hands.
        """
        player_value = self.player_hand.value()
        dealer_value = self.dealer_hand.value()

        if self.dealer_hand.is_bust():
            return REWARD_WIN

        if player_value > dealer_value:
            return REWARD_WIN

        if player_value < dealer_value:
            return REWARD_LOSS

        return REWARD_DRAW

    def available_actions(self):
        """
        Return possible actions.

        For this simple version, both actions are always available
        until the game is done.
        """
        if self.done:
            return []

        return [Action.HIT, Action.STAND]

    def render(self):
        """
        Print current game state for debugging.
        """
        if self.player_hand is None or self.dealer_hand is None:
            print("Game has not started.")
            return

        print(f"Player: {self.player_hand}")

        if self.done:
            print(f"Dealer: {self.dealer_hand}")
        else:
            print(f"Dealer upcard: {self.dealer_hand.cards[0]}")