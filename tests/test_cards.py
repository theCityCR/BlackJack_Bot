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


# =========================
# Existing behavior
# =========================


def test_deck_draw_card_returns_valid_value():
    deck = Deck()

    initial_cards_remaining = deck.cards_remaining()

    for _ in range(initial_cards_remaining):
        card = deck.draw_card()
        assert card in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    assert deck.cards_remaining() == 0


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


# =========================
# New constructor behavior
# =========================


def test_hand_can_be_created_without_initial_deal():
    deck = FixedDeck([10, 6])
    hand = Hand(deck, deal_initial=False)

    assert hand.cards == []
    assert hand.value() == 0
    assert deck.index == 0


def test_hand_without_initial_deal_can_still_hit():
    deck = FixedDeck([10, 6])
    hand = Hand(deck, deal_initial=False)

    hand.hit()

    assert hand.cards == [10]
    assert hand.value() == 10
    assert deck.index == 1


def test_from_cards_creates_hand_without_drawing_from_deck():
    deck = FixedDeck([5, 6])
    hand = Hand.from_cards(deck, [10, 1])

    assert hand.cards == [10, 1]
    assert hand.value() == 21
    assert deck.index == 0


def test_from_cards_copies_input_iterable():
    deck = FixedDeck([5])
    cards = [8, 8]
    hand = Hand.from_cards(deck, cards)

    cards.append(10)

    assert hand.cards == [8, 8]


# =========================
# New split behavior
# =========================


def test_can_split_true_for_matching_pair():
    deck = FixedDeck([8, 8])
    hand = Hand(deck)

    assert hand.can_split() is True


def test_can_split_true_for_pair_of_aces():
    deck = FixedDeck([1, 1])
    hand = Hand(deck)

    assert hand.can_split() is True


def test_can_split_true_for_pair_of_tens():
    deck = FixedDeck([10, 10])
    hand = Hand(deck)

    assert hand.can_split() is True


def test_can_split_false_for_non_matching_two_card_hand():
    deck = FixedDeck([10, 9])
    hand = Hand(deck)

    assert hand.can_split() is False


def test_can_split_false_when_hand_has_more_than_two_cards():
    deck = FixedDeck([8, 8, 3])
    hand = Hand(deck)
    hand.hit()

    assert hand.cards == [8, 8, 3]
    assert hand.can_split() is False


def test_split_creates_two_new_hands_and_deals_one_card_to_each():
    # Initial hand: [8, 8]
    # After split, first hand gets 3 and second hand gets 10.
    deck = FixedDeck([8, 8, 3, 10])
    hand = Hand(deck)

    first_hand, second_hand = hand.split()

    assert first_hand.cards == [8, 3]
    assert second_hand.cards == [8, 10]
    assert first_hand.value() == 11
    assert second_hand.value() == 18
    assert deck.index == 4


def test_split_hands_share_same_deck():
    deck = FixedDeck([7, 7, 2, 3, 4])
    hand = Hand(deck)

    first_hand, second_hand = hand.split()
    first_hand.hit()

    assert first_hand.deck is deck
    assert second_hand.deck is deck
    assert first_hand.cards == [7, 2, 4]
    assert second_hand.cards == [7, 3]


def test_split_raises_value_error_for_non_pair():
    deck = FixedDeck([10, 9])
    hand = Hand(deck)

    with pytest.raises(ValueError):
        hand.split()


def test_split_raises_value_error_after_hit_even_if_first_two_cards_match():
    deck = FixedDeck([8, 8, 2])
    hand = Hand(deck)
    hand.hit()

    with pytest.raises(ValueError):
        hand.split()


# =========================
# New blackjack behavior
# =========================


def test_is_blackjack_true_for_ace_and_ten():
    deck = FixedDeck([1, 10])
    hand = Hand(deck)

    assert hand.is_blackjack() is True


def test_is_blackjack_true_for_ten_and_ace():
    deck = FixedDeck([10, 1])
    hand = Hand(deck)

    assert hand.is_blackjack() is True


def test_is_blackjack_false_for_three_card_twenty_one():
    deck = FixedDeck([1, 5, 5])
    hand = Hand(deck)
    hand.hit()

    assert hand.value() == 21
    assert hand.is_blackjack() is False


def test_is_blackjack_false_for_two_card_non_twenty_one():
    deck = FixedDeck([10, 9])
    hand = Hand(deck)

    assert hand.is_blackjack() is False


# =========================
# Finite deck behavior
# =========================


def test_finite_deck_starts_with_52_cards_by_default():
    deck = Deck(shuffle=False)

    assert deck.cards_remaining() == 52


def test_finite_deck_has_correct_single_deck_count_vector():
    deck = Deck(shuffle=False)

    assert deck.get_count_vector() == (
        4,   # A
        4,   # 2
        4,   # 3
        4,   # 4
        4,   # 5
        4,   # 6
        4,   # 7
        4,   # 8
        4,   # 9
        16,  # 10/J/Q/K
    )


def test_drawing_card_reduces_cards_remaining():
    deck = Deck(shuffle=False)

    old_remaining = deck.cards_remaining()
    card = deck.draw_card()

    assert card in range(1, 11)
    assert deck.cards_remaining() == old_remaining - 1


def test_drawing_card_updates_count_vector():
    deck = Deck(shuffle=False)

    # With shuffle=False, cards are popped from the end.
    # The end of SINGLE_DECK_CARDS is a 10.
    card = deck.draw_card()

    assert card == 10
    assert deck.get_count_vector() == (
        4, 4, 4, 4, 4, 4, 4, 4, 4, 15
    )


def test_count_remaining_returns_count_for_specific_card_value():
    deck = Deck(shuffle=False)

    assert deck.count_remaining(1) == 4
    assert deck.count_remaining(10) == 16

    deck.draw_card()

    assert deck.count_remaining(10) == 15


def test_multiple_deck_shoe_has_scaled_counts():
    deck = Deck(num_decks=6, shuffle=False)

    assert deck.cards_remaining() == 312
    assert deck.get_count_vector() == (
        24, 24, 24, 24, 24, 24, 24, 24, 24, 96
    )


def test_reset_does_not_reshuffle_when_enough_cards_remain():
    deck = Deck(shuffle=False, reshuffle_threshold=15)

    for _ in range(10):
        deck.draw_card()

    assert deck.cards_remaining() == 42

    deck.reset()

    assert deck.cards_remaining() == 42


def test_reset_reshuffles_when_below_threshold():
    deck = Deck(shuffle=False, reshuffle_threshold=15)

    while deck.cards_remaining() >= 15:
        deck.draw_card()

    assert deck.cards_remaining() == 14

    deck.reset()

    assert deck.cards_remaining() == 52
    assert deck.get_count_vector() == (
        4, 4, 4, 4, 4, 4, 4, 4, 4, 16
    )


def test_force_reset_always_restores_full_deck():
    deck = Deck(shuffle=False, reshuffle_threshold=15)

    for _ in range(20):
        deck.draw_card()

    assert deck.cards_remaining() == 32

    deck.force_reset()

    assert deck.cards_remaining() == 52


def test_draw_from_empty_deck_raises_runtime_error():
    deck = Deck(shuffle=False)

    for _ in range(52):
        deck.draw_card()

    assert deck.cards_remaining() == 0

    with pytest.raises(RuntimeError):
        deck.draw_card()


def test_invalid_num_decks_raises_value_error():
    with pytest.raises(ValueError):
        Deck(num_decks=0)


def test_invalid_reshuffle_threshold_raises_value_error():
    with pytest.raises(ValueError):
        Deck(reshuffle_threshold=-1)


def test_invalid_count_remaining_card_value_raises_value_error():
    deck = Deck()

    with pytest.raises(ValueError):
        deck.count_remaining(0)

    with pytest.raises(ValueError):
        deck.count_remaining(11)


def test_force_reset_shuffles_when_enabled():
    deck = Deck(shuffle=True)

    original = list(deck.cards)

    deck.force_reset()
    new = deck.cards

    # Very small probability of equality, acceptable for test
    assert original != new


def test_multiple_resets_do_not_reshuffle_until_threshold():
    deck = Deck(shuffle=False, reshuffle_threshold=10)

    # Remove some cards, but stay above threshold
    for _ in range(20):
        deck.draw_card()

    remaining_before = deck.cards_remaining()

    # Call reset multiple times
    for _ in range(5):
        deck.reset()

    assert deck.cards_remaining() == remaining_before