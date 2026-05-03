"""
cards.py

Card, Deck, and Hand abstractions for Blackjack.
"""

import random
from typing import Iterable, List, Tuple

from config import (
    NUM_DECKS,
    RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
)


SINGLE_DECK_CARDS = (
    [1] * 4 +
    [2] * 4 +
    [3] * 4 +
    [4] * 4 +
    [5] * 4 +
    [6] * 4 +
    [7] * 4 +
    [8] * 4 +
    [9] * 4 +
    [10] * 16
)


class Deck:
    """
    Finite blackjack deck.

    Ace is represented as 1.
    10, J, Q, and K are all represented as 10.

    Important:
    reset() does not automatically reshuffle every round.
    It only reshuffles when the remaining deck is below the configured
    reshuffle threshold.
    """

    def __init__(
        self,
        num_decks: int = NUM_DECKS,
        reshuffle_threshold: int = RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
        shuffle: bool = True,
    ):
        if num_decks <= 0:
            raise ValueError("num_decks must be positive")

        if reshuffle_threshold < 0:
            raise ValueError("reshuffle_threshold cannot be negative")

        self.num_decks = num_decks
        self.reshuffle_threshold = reshuffle_threshold
        self.shuffle = shuffle
        self.cards: List[int] = []

        self.force_reset()

    def force_reset(self):
        """
        Rebuild and optionally shuffle the full shoe.
        """
        self.cards = list(SINGLE_DECK_CARDS) * self.num_decks

        if self.shuffle:
            random.shuffle(self.cards)

    def reset(self):
        """
        Prepare the deck for a new round.

        This intentionally does not reshuffle every round.
        It reshuffles only when the remaining shoe is too low.
        """
        if self.cards_remaining() < self.reshuffle_threshold:
            self.force_reset()

    def draw_card(self) -> int:
        """
        Draw one card from the finite deck.
        """
        if not self.cards:
            raise RuntimeError("Cannot draw from an empty deck.")

        return self.cards.pop()

    def cards_remaining(self) -> int:
        """
        Return how many cards remain in the shoe.
        """
        return len(self.cards)

    def get_count_vector(self) -> Tuple[int, ...]:
        """
        Return remaining card counts in this order:

        [A, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        10 includes 10, J, Q, and K.
        """
        counts = [0] * 10

        for card in self.cards:
            if card == 1:
                counts[0] += 1
            elif 2 <= card <= 9:
                counts[card - 1] += 1
            elif card == 10:
                counts[9] += 1
            else:
                raise ValueError(f"Invalid card value in deck: {card}")

        return tuple(counts)

    def count_remaining(self, card_value: int) -> int:
        """
        Return the number of remaining cards with the given value.
        """
        if card_value not in range(1, 11):
            raise ValueError("card_value must be between 1 and 10")

        return self.get_count_vector()[card_value - 1]

    def __repr__(self):
        return (
            f"Deck(num_decks={self.num_decks}, "
            f"cards_remaining={self.cards_remaining()})"
        )


class Hand:
    """
    Represents a blackjack hand.

    By default, a new hand is dealt 2 cards.
    """

    def __init__(self, deck: Deck, deal_initial: bool = True):
        self.deck = deck
        self.cards: List[int] = []

        if deal_initial:
            self.hit()
            self.hit()

    @classmethod
    def from_cards(cls, deck: Deck, cards: Iterable[int]) -> "Hand":
        """
        Create a hand from existing cards without drawing from the deck.
        """
        hand = cls(deck, deal_initial=False)
        hand.cards = list(cards)
        return hand

    def hit(self):
        """
        Draw a card from the deck into this hand.
        """
        self.cards.append(self.deck.draw_card())

    def usable_ace(self) -> bool:
        """
        Whether hand has a usable ace counted as 11.
        """
        return 1 in self.cards and sum(self.cards) + 10 <= 21

    def value(self) -> int:
        """
        Compute blackjack hand value.
        """
        total = sum(self.cards)

        if self.usable_ace():
            return total + 10

        return total

    def is_bust(self) -> bool:
        """
        Whether this hand is bust.
        """
        return self.value() > 21

    def can_split(self) -> bool:
        """
        Whether this hand can be split.
        """
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def split(self) -> Tuple["Hand", "Hand"]:
        """
        Split this two-card hand into two hands.

        Each new hand receives one original card and one newly dealt card.
        """
        if not self.can_split():
            raise ValueError(f"Cannot split hand: {self.cards}")

        first_card, second_card = self.cards

        first_hand = Hand.from_cards(self.deck, [first_card])
        second_hand = Hand.from_cards(self.deck, [second_card])

        first_hand.hit()
        second_hand.hit()

        return first_hand, second_hand

    def is_blackjack(self) -> bool:
        """
        Whether this hand is a natural blackjack.
        """
        return len(self.cards) == 2 and self.value() == 21

    def __repr__(self):
        return f"Hand(cards={self.cards}, value={self.value()})"