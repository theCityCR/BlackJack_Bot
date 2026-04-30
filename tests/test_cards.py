import pytest

from cards import Deck, Hand


class FixedDeck:
    """
    Fake deck for deterministic tests.
    Draws cards from left to right.
    """

    def __init__(self, cards):
        self.cards = cards
        self.index = 0

    def draw_card(self):
        card = self.cards[self.index]
        self.index += 1
        return card


def test_deck_draw_card_returns_valid_value():
    deck = Deck()

    for _ in range(100):
        card = deck.draw_card()
        assert card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_hand_starts_with_two_cards():
    deck = FixedDeck([10, 6])
    hand = Hand(deck)

    assert hand.cards == [10, 6]
    assert hand.value() == 16


def test_hit_adds_one_card():
    deck = FixedDeck([10, 6, 5])
    hand = Hand(deck)

    hand.hit()

    assert hand.cards == [10, 6, 5]
    assert hand.value() == 21


def test_usable_ace_true():
    deck = FixedDeck([1, 7])
    hand = Hand(deck)

    assert hand.usable_ace() is True
    assert hand.value() == 18


def test_usable_ace_false_when_it_would_bust():
    deck = FixedDeck([1, 10, 5])
    hand = Hand(deck)
    hand.hit()

    assert hand.usable_ace() is False
    assert hand.value() == 16


def test_bust_true():
    deck = FixedDeck([10, 9, 5])
    hand = Hand(deck)
    hand.hit()

    assert hand.value() == 24
    assert hand.is_bust() is True


def test_bust_false():
    deck = FixedDeck([10, 9])
    hand = Hand(deck)

    assert hand.value() == 19
    assert hand.is_bust() is False


def test_multiple_aces():
    deck = FixedDeck([1, 1, 9])
    hand = Hand(deck)
    hand.hit()

    assert hand.cards == [1, 1, 9]
    assert hand.usable_ace() is True
    assert hand.value() == 21


def test_repr_contains_cards_and_value():
    deck = FixedDeck([10, 6])
    hand = Hand(deck)

    text = repr(hand)

    assert "Hand" in text
    assert "cards=[10, 6]" in text
    assert "value=16" in text