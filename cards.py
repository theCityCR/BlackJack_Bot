"""
cards.py

Card, Deck, and Hand abstractions for Blackjack.
"""

import random
from typing import Iterable, List, Tuple

from config import CARD_VALUES


# =========================
# Deck
# =========================

class Deck:
    """
    Deck abstraction.

    Currently infinite.
    Later: can switch to finite without changing API.
    """

    def __init__(self):
        pass  # nothing needed for infinite deck

    def draw_card(self) -> int:
        """
        Draw a card from the deck.
        """
        return random.choice(CARD_VALUES)

    def reset(self):
        """
        Reset deck (no-op for infinite deck).
        Exists for future finite deck support.
        """
        pass


# =========================
# Hand
# =========================

class Hand:
    """
    Represents a blackjack hand.

    By default, a new hand is dealt 2 cards.

    For split support, pass deal_initial=False and either:
    - add cards manually, or
    - use Hand.from_cards(...)
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
        Create a hand from existing cards without dealing 2 new cards.

        Useful when splitting a two-card hand into two one-card hands.
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
        Whether hand has a usable ace (counts as 11 safely).
        """
        return 1 in self.cards and sum(self.cards) + 10 <= 21

    def value(self) -> int:
        """
        Compute hand value.
        """
        total = sum(self.cards)
        if self.usable_ace():
            return total + 10
        return total

    def is_bust(self) -> bool:
        """
        Check if hand is bust.
        """
        return self.value() > 21

    def can_split(self) -> bool:
        """
        Whether this hand is eligible to split.

        In this simple card model, 10, J, Q, and K are all represented as 10,
        so any pair of 10-value cards counts as splittable.
        """
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def split(self) -> Tuple["Hand", "Hand"]:
        """
        Split this two-card hand into two hands.

        Each new hand receives one of the original cards plus one new card.
        Raises ValueError if the hand cannot be split.
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

        Note: after split is added, game.py may choose not to treat split hands
        as natural blackjacks, depending on the rules you want.
        """
        return len(self.cards) == 2 and self.value() == 21

    def __repr__(self):
        return f"Hand(cards={self.cards}, value={self.value()})"
