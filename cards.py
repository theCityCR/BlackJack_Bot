"""
cards.py

Card, Deck, and Hand abstractions for Blackjack.

Deck behavior:
- At the start of each round, generate a fresh random shoe of 26 cards.
- Each card is sampled from the standard blackjack deck distribution.
- The game knows the full 26-card shoe through Deck.cards.
- The agent receives the remaining card counts through get_count_vector().
"""

import random
from typing import Iterable, List, Tuple


RANDOM_SHOE_SIZE = 26

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
    Random 26-card blackjack shoe.

    Ace is represented as 1.
    10, J, Q, and K are all represented as 10.

    Important:
    This is not a real half-deck.
    Each of the 26 cards is sampled independently from the standard blackjack
    deck distribution.

    This means:
    - The shoe is finite during the round.
    - The agent can use the remaining-card count vector.
    - A new random 26-card shoe is generated every round.
    """

    def __init__(self, num_cards: int = RANDOM_SHOE_SIZE, shuffle: bool = True):
        if num_cards <= 0:
            raise ValueError("num_cards must be positive")

        self.num_cards = num_cards
        self.shuffle = shuffle
        self.cards: List[int] = []

        self.force_reset()

    def force_reset(self):
        """
        Generate a new random shoe.
        """
        self.cards = [
            random.choice(SINGLE_DECK_CARDS)
            for _ in range(self.num_cards)
        ]

        if self.shuffle:
            random.shuffle(self.cards)

    def reset(self):
        """
        Prepare the deck for a new round.

        For this deck type, every round gets a fresh random 26-card shoe.
        """
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
            f"Deck(num_cards={self.num_cards}, "
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