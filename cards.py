"""
cards.py

Card, Deck, and Hand abstractions for Blackjack.

Deck behavior:
- Build a standard finite shoe from one or more decks.
- Keep using that shoe between rounds until it reaches a cut-card threshold.
- The game knows the remaining shoe through Deck.cards.
- The agent receives the remaining card counts through get_count_vector().
"""

import random
from typing import Iterable, List, Tuple

from config import NUM_DECKS, RESHUFFLE_WHEN_CARDS_REMAINING_BELOW

CARDS_PER_DECK = 52

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


def shoe_size(num_decks: int = NUM_DECKS) -> int:
    """Total cards in a full ``num_decks`` shoe."""
    if num_decks <= 0:
        raise ValueError("num_decks must be positive")
    return CARDS_PER_DECK * num_decks


def dealt_penetration(
    reshuffle_threshold: int, num_decks: int = NUM_DECKS
) -> float:
    """Fraction of the shoe dealt before the cut (higher = deeper penetration)."""
    size = shoe_size(num_decks)
    if not 0 <= reshuffle_threshold <= size:
        raise ValueError("reshuffle_threshold must be within the shoe size")
    return (size - reshuffle_threshold) / size


class Deck:
    """
    Finite blackjack shoe made from standard 52-card decks.

    Ace is represented as 1.
    10, J, Q, and K are all represented as 10.

    The shoe persists across rounds and is rebuilt when no more than
    ``reshuffle_threshold`` cards remain. This preserves real card frequencies
    and makes the count vector useful across consecutive rounds.
    """

    def __init__(
        self,
        num_decks: int = NUM_DECKS,
        shuffle: bool = True,
        reshuffle_threshold: int = RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
    ):
        if num_decks <= 0:
            raise ValueError("num_decks must be positive")
        if not 0 <= reshuffle_threshold <= shoe_size(num_decks):
            raise ValueError("reshuffle_threshold must be within the shoe size")

        self.num_decks = num_decks
        self.shuffle = shuffle
        self.reshuffle_threshold = reshuffle_threshold
        self.cards: List[int] = []

        self.force_reset()

    def force_reset(self):
        """
        Build a complete new shoe.
        """
        self.cards = list(SINGLE_DECK_CARDS) * self.num_decks

        if self.shuffle:
            random.shuffle(self.cards)

    def reset(self):
        """
        Prepare the deck for a new round.

        Keep the current shoe unless it has passed the cut card.
        """
        if self.cards_remaining() <= self.reshuffle_threshold:
            self.force_reset()

    def draw_card(self) -> int:
        """
        Draw one card from the random finite shoe.
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
