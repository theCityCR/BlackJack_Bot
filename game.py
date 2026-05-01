"""
game.py

Blackjack game environment.

Supported player actions:
- Hit
- Stand
- Double
- Split

Design notes:
- The environment exposes one active player hand at a time.
- Splitting creates multiple player hands, which are played left to right.
- Dealer plays once, after all player hands are finished.
- Per-hand metadata is stored with the hand in PlayerHandState instead of in
  parallel lists. This makes split/double logic easier to reason about.
- Each hand stores its own final reward. The round reward is the sum of all
  final hand rewards.
"""

from dataclasses import dataclass
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


BLACKJACK_PAYOUT = 1.5


class Action(Enum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3


@dataclass(frozen=True)
class GameState:
    """
    Public game state given to an agent.

    This describes the current active hand, plus enough split context to avoid
    merging states that can have different future values.
    """

    player_value: int
    dealer_upcard: int
    usable_ace: bool
    can_double: bool
    can_split: bool
    is_split_hand: bool
    active_hand_index: int
    num_hands: int

    def as_tuple(self) -> Tuple[int, int, bool, bool, bool, bool, int, int]:
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
            self.active_hand_index,
            self.num_hands,
        )


@dataclass
class PlayerHandState:
    """
    One player hand plus the metadata needed to score and restrict it.
    """

    hand: Hand
    bet_multiplier: float = 1.0
    is_split_hand: bool = False
    is_split_aces_hand: bool = False
    is_finished: bool = False
    final_reward: Optional[float] = None


class BlackjackGame:
    """
    Blackjack environment.

    Agent interaction pattern:

        game = BlackjackGame()
        state = game.reset()

        done = state is None
        while not done:
            available_actions = game.available_actions()
            action = agent.choose_action(state, available_actions)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    Rules currently modeled:
    - Dealer stands on soft 17 because dealer draws only while value < 17.
    - Dealer blackjack is checked immediately on the initial deal. If dealer has
      blackjack, the round ends before the player can hit, double, or split.
    - Double gives exactly one card and then finishes that hand.
    - Split creates two hands and plays them left to right.
    - Double after split is controlled by ALLOW_DOUBLE_AFTER_SPLIT.
    - Split aces receive one card each when ALLOW_HIT_SPLIT_ACES is False.
    - Natural blackjack pays 3:2 only on the original unsplit hand.
    - Each player hand stores its own final_reward; round_reward is the sum.
    """

    def __init__(self, deck: Optional[Deck] = None):
        self.deck = deck if deck is not None else Deck()
        self.hand_states: List[PlayerHandState] = []
        self.dealer_hand: Optional[Hand] = None
        self.active_hand_index = 0
        self.done = False
        self.round_reward: Optional[float] = None
        self.initial_dealer_blackjack = False

    # =========================
    # Backward-compatible views
    # =========================

    @property
    def player_hands(self) -> List[Hand]:
        """
        Return all player Hand objects.

        New code should use hand_states when it needs metadata such as bet size,
        whether a hand came from a split, or that hand's final reward.
        """
        return [state.hand for state in self.hand_states]

    @property
    def player_hand(self) -> Optional[Hand]:
        """
        Return the current active hand, for compatibility with older tests/code.
        """
        if not self.hand_states:
            return None

        if self.active_hand_index >= len(self.hand_states):
            return self.hand_states[-1].hand

        return self.hand_states[self.active_hand_index].hand

    @property
    def hand_bets(self) -> List[float]:
        """
        Compatibility view of bet multipliers.
        """
        return [state.bet_multiplier for state in self.hand_states]

    @property
    def hand_rewards(self) -> List[Optional[float]]:
        """
        Final reward for each player hand.

        Values are None until the hand's final result is known. After the round
        is done, every hand has a final reward and round_reward is their sum.
        """
        return [state.final_reward for state in self.hand_states]

    @property
    def is_split_hand(self) -> List[bool]:
        """
        Compatibility view of split-hand flags.
        """
        return [state.is_split_hand for state in self.hand_states]

    @property
    def is_split_aces_hand(self) -> List[bool]:
        """
        Compatibility view of split-aces flags.
        """
        return [state.is_split_aces_hand for state in self.hand_states]

    # Older intermediate versions used this name.
    @property
    def split_aces(self) -> List[bool]:
        return self.is_split_aces_hand

    # =========================
    # Public API
    # =========================

    def reset(self) -> Optional[GameState]:
        """
        Start a new round.

        If the dealer has a natural blackjack on the initial deal, the round is
        resolved immediately before the player acts. In that case, this method
        returns None and the final rewards are available through hand_rewards
        and round_reward.
        """
        self.deck.reset()

        self.hand_states = [PlayerHandState(hand=Hand(self.deck))]
        self.dealer_hand = Hand(self.deck)
        self.active_hand_index = 0
        self.done = False
        self.round_reward = None
        self.initial_dealer_blackjack = False

        if self._dealer_has_blackjack():
            self.initial_dealer_blackjack = True
            self._finish_round(dealer_already_resolved=True)
            return None

        return self.get_state()

    def get_state(self) -> GameState:
        """
        Return the state visible to the agent.
        """
        self._require_started()

        if self.done:
            raise RuntimeError("Round is already done; there is no active state.")

        hand_state = self.current_hand_state()
        hand = hand_state.hand

        return GameState(
            player_value=hand.value(),
            dealer_upcard=self.dealer_hand.cards[0],
            usable_ace=hand.usable_ace(),
            can_double=self._can_double(hand_state),
            can_split=self._can_split(hand_state),
            is_split_hand=hand_state.is_split_hand,
            active_hand_index=self.active_hand_index,
            num_hands=len(self.hand_states),
        )

    def available_actions(self) -> List[Action]:
        """
        Return currently legal actions.
        """
        if self.done:
            return []

        self._require_started()

        hand_state = self.current_hand_state()
        hand = hand_state.hand

        # Natural blackjack and 21 are effectively decision-complete.
        if self._is_natural_blackjack(hand_state) or hand.value() == 21:
            return [Action.STAND]

        # Split aces usually receive only one card and then must stand.
        if hand_state.is_split_aces_hand and not ALLOW_HIT_SPLIT_ACES:
            return [Action.STAND]

        actions = [Action.HIT, Action.STAND]

        if self._can_double(hand_state):
            actions.append(Action.DOUBLE)

        if self._can_split(hand_state):
            actions.append(Action.SPLIT)

        return actions

    def step(self, action: Action) -> Tuple[Optional[GameState], float, bool]:
        """
        Apply one player action.

        Returns:
            next_state, reward, done

        If done is True, next_state is None. The reward returned by step() is
        still the total round reward at terminal time for compatibility. For
        better split-hand training, use hand_rewards after the round ends.
        """
        if self.done:
            raise RuntimeError("Game is already done. Call reset() to start a new game.")

        self._require_started()
        self._validate_action(action)

        if action == Action.HIT:
            return self._hit()

        if action == Action.STAND:
            return self._stand()

        if action == Action.DOUBLE:
            return self._double()

        if action == Action.SPLIT:
            return self._split()

        raise ValueError(f"Unknown action: {action}")

    def render(self):
        """
        Print current game state for debugging.
        """
        if not self.hand_states or self.dealer_hand is None:
            print("Game has not started.")
            return

        for index, hand_state in enumerate(self.hand_states):
            marker = " <- active" if not self.done and index == self.active_hand_index else ""
            split_text = ", split" if hand_state.is_split_hand else ""
            ace_text = ", split aces" if hand_state.is_split_aces_hand else ""
            finished_text = ", finished" if hand_state.is_finished else ""
            reward_text = "" if hand_state.final_reward is None else f", reward={hand_state.final_reward}"

            print(
                f"Player hand {index + 1}: {hand_state.hand}, "
                f"bet={hand_state.bet_multiplier}"
                f"{split_text}{ace_text}{finished_text}{reward_text}{marker}"
            )

        if self.done:
            print(f"Dealer: {self.dealer_hand}")
            print(f"Round reward: {self.round_reward}")
        else:
            print(f"Dealer upcard: {self.dealer_hand.cards[0]}")

    # =========================
    # Current-hand helpers
    # =========================

    def current_hand_state(self) -> PlayerHandState:
        """
        Return the current active player hand state.
        """
        self._require_started()

        if self.active_hand_index >= len(self.hand_states):
            raise RuntimeError("There is no active hand.")

        return self.hand_states[self.active_hand_index]

    def current_hand(self) -> Hand:
        """
        Return the current active player hand.
        """
        return self.current_hand_state().hand

    # =========================
    # Action handlers
    # =========================

    def _hit(self) -> Tuple[Optional[GameState], float, bool]:
        """
        Player hits the current hand.
        """
        hand_state = self.current_hand_state()
        hand_state.hand.hit()

        if hand_state.hand.is_bust():
            return self._finish_current_hand()

        return self.get_state(), 0.0, False

    def _stand(self) -> Tuple[Optional[GameState], float, bool]:
        """
        Player stands on the current hand.
        """
        return self._finish_current_hand()

    def _double(self) -> Tuple[Optional[GameState], float, bool]:
        """
        Player doubles the current hand.
        """
        hand_state = self.current_hand_state()
        hand_state.hand.hit()
        hand_state.bet_multiplier *= DOUBLE_REWARD_MULTIPLIER

        return self._finish_current_hand()

    def _split(self) -> Tuple[Optional[GameState], float, bool]:
        """
        Player splits the current hand into two hands.
        """
        original_state = self.current_hand_state()
        original_hand = original_state.hand
        split_aces = original_hand.cards[0] == 1

        first_hand, second_hand = original_hand.split()

        first_state = PlayerHandState(
            hand=first_hand,
            bet_multiplier=original_state.bet_multiplier,
            is_split_hand=True,
            is_split_aces_hand=split_aces,
        )
        second_state = PlayerHandState(
            hand=second_hand,
            bet_multiplier=original_state.bet_multiplier,
            is_split_hand=True,
            is_split_aces_hand=split_aces,
        )

        self.hand_states[self.active_hand_index] = first_state
        self.hand_states.insert(self.active_hand_index + 1, second_state)

        # If split aces cannot be hit, the first split-ace hand is immediately
        # complete. The second split-ace hand will also only be allowed to stand.
        if split_aces and not ALLOW_HIT_SPLIT_ACES:
            return self._finish_current_hand()

        return self.get_state(), 0.0, False

    # =========================
    # Round flow
    # =========================

    def _finish_current_hand(self) -> Tuple[Optional[GameState], float, bool]:
        """
        Finish the current hand and either move to the next hand or end the round.
        """
        hand_state = self.current_hand_state()
        hand_state.is_finished = True

        # Busts are known immediately and do not depend on the dealer result.
        if hand_state.hand.is_bust():
            hand_state.final_reward = hand_state.bet_multiplier * REWARD_LOSS

        self.active_hand_index += 1

        if self.active_hand_index < len(self.hand_states):
            return self.get_state(), 0.0, False

        return self._finish_round()

    def _finish_round(self, dealer_already_resolved: bool = False) -> Tuple[None, float, bool]:
        """
        Finish the full round after all player hands are complete.

        Each hand receives its own final_reward. The returned reward is still
        the total round reward for backward compatibility and accounting.
        """
        if not dealer_already_resolved and self._dealer_should_play():
            self._play_dealer_turn()

        self._assign_final_rewards()
        self.round_reward = self._total_reward()
        self.done = True
        self.active_hand_index = len(self.hand_states)

        return None, self.round_reward, True

    # =========================
    # Dealer logic
    # =========================

    def _dealer_should_play(self) -> bool:
        """
        Return whether the dealer needs to draw before scoring.
        """
        if self._all_player_hands_bust():
            return False

        if self._dealer_has_blackjack():
            return False

        # If all remaining player hands are already natural blackjacks and the
        # dealer does not have blackjack, dealer drawing cannot change scoring.
        if self._all_non_bust_hands_are_natural_blackjacks():
            return False

        return True

    def _play_dealer_turn(self):
        """
        Dealer hits until reaching stand threshold.
        """
        while self._dealer_should_hit():
            self.dealer_hand.hit()

    def _dealer_should_hit(self) -> bool:
        """
        Dealer policy.

        With the current rule, dealer stands on soft 17 because value() == 17.
        """
        return self.dealer_hand.value() < DEALER_STAND_THRESHOLD

    # =========================
    # Scoring
    # =========================

    def _assign_final_rewards(self):
        """
        Store the final reward on each hand.

        Bust rewards may already have been assigned when the hand busted. All
        other hand rewards are assigned once the dealer result is known.
        """
        for hand_state in self.hand_states:
            if hand_state.final_reward is None:
                hand_state.final_reward = hand_state.bet_multiplier * self._base_reward(hand_state)

            hand_state.is_finished = True

    def _total_reward(self) -> float:
        """
        Return the total round reward across all player hands.
        """
        total = 0.0

        for hand_state in self.hand_states:
            if hand_state.final_reward is not None:
                total += hand_state.final_reward
            else:
                total += hand_state.bet_multiplier * self._base_reward(hand_state)

        return total

    def _base_reward(self, hand_state: PlayerHandState) -> float:
        """
        Reward for one hand before applying its bet multiplier.
        """
        hand = hand_state.hand

        if hand.is_bust():
            return REWARD_LOSS

        player_blackjack = self._is_natural_blackjack(hand_state)
        dealer_blackjack = self._dealer_has_blackjack()

        if player_blackjack and dealer_blackjack:
            return REWARD_DRAW

        if player_blackjack:
            return BLACKJACK_PAYOUT * REWARD_WIN

        if dealer_blackjack:
            return REWARD_LOSS

        if self.dealer_hand.is_bust():
            return REWARD_WIN

        player_value = hand.value()
        dealer_value = self.dealer_hand.value()

        if player_value > dealer_value:
            return REWARD_WIN

        if player_value < dealer_value:
            return REWARD_LOSS

        return REWARD_DRAW

    def _compare_hands(self) -> float:
        """
        Backward-compatible final comparison helper.

        With split support, this returns the total reward across all player hands.
        """
        return self._total_reward()

    def _is_natural_blackjack(self, hand_state: PlayerHandState) -> bool:
        """
        Return whether this hand is an original, unsplit natural blackjack.
        """
        return not hand_state.is_split_hand and hand_state.hand.is_blackjack()

    def _dealer_has_blackjack(self) -> bool:
        """
        Return whether the dealer has a two-card blackjack.
        """
        return self.dealer_hand is not None and self.dealer_hand.is_blackjack()

    def _all_player_hands_bust(self) -> bool:
        """
        Return whether every player hand is bust.
        """
        return all(hand_state.hand.is_bust() for hand_state in self.hand_states)

    def _all_non_bust_hands_are_natural_blackjacks(self) -> bool:
        """
        Return whether every non-bust player hand is a natural blackjack.
        """
        found_non_bust_hand = False

        for hand_state in self.hand_states:
            if hand_state.hand.is_bust():
                continue

            found_non_bust_hand = True

            if not self._is_natural_blackjack(hand_state):
                return False

        return found_non_bust_hand

    # =========================
    # Legality helpers
    # =========================

    def _validate_action(self, action: Action):
        """
        Ensure the requested action is legal in the current state.
        """
        if not isinstance(action, Action):
            raise ValueError(f"Unknown action: {action}")

        if action not in self.available_actions():
            raise ValueError(f"Action is not currently available: {action}")

    def _can_double(self, hand_state: PlayerHandState) -> bool:
        """
        Return whether the given hand can double.
        """
        if self.done:
            return False

        if len(hand_state.hand.cards) != 2:
            return False

        if hand_state.hand.value() == 21:
            return False

        if self._is_natural_blackjack(hand_state):
            return False

        if hand_state.is_split_aces_hand and not ALLOW_HIT_SPLIT_ACES:
            return False

        if hand_state.is_split_hand and not ALLOW_DOUBLE_AFTER_SPLIT:
            return False

        return True

    def _can_split(self, hand_state: PlayerHandState) -> bool:
        """
        Return whether the given hand can split.
        """
        if self.done:
            return False

        if self._is_natural_blackjack(hand_state):
            return False

        if hand_state.hand.value() == 21:
            return False

        if len(self.hand_states) >= MAX_PLAYER_HANDS:
            return False

        if hand_state.is_split_hand and not ALLOW_RESPLIT:
            return False

        if hand_state.is_split_aces_hand and not ALLOW_HIT_SPLIT_ACES:
            return False

        return hand_state.hand.can_split()

    # =========================
    # Validation helpers
    # =========================

    def _require_started(self):
        """
        Ensure reset() has been called.
        """
        if not self.hand_states or self.dealer_hand is None:
            raise RuntimeError("Game has not been reset yet.")
