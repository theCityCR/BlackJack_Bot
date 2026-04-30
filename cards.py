"""
cards.py

Card, Deck, and Hand abstractions for Blackjack.
"""

import random
from typing import List

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
    """

    def __init__(self, deck: Deck):
        self.deck = deck
        self.cards: List[int] = []

        # initial 2 cards
        self.hit()
        self.hit()

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

    def __repr__(self):
        return f"Hand(cards={self.cards}, value={self.value()})"