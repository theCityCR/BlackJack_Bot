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


def expected_count_vector(num_decks):
    return tuple([4 * num_decks] * 9 + [16 * num_decks])


@pytest.mark.parametrize("num_decks", [1, 2, 4, 6])
def test_finite_deck_total_cards_scales_with_num_decks(num_decks):
    deck = Deck(num_decks=num_decks, shuffle=False)

    assert deck.cards_remaining() == 52 * num_decks


@pytest.mark.parametrize("num_decks", [1, 2, 4, 6])
def test_finite_deck_count_vector_scales_with_num_decks(num_decks):
    deck = Deck(num_decks=num_decks, shuffle=False)

    assert deck.get_count_vector() == expected_count_vector(num_decks)


def test_deck_draw_exhausts_all_cards_with_valid_values():
    deck = Deck()

    initial_cards_remaining = deck.cards_remaining()

    for _ in range(initial_cards_remaining):
        card = deck.draw_card()
        assert card in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    assert deck.cards_remaining() == 0


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_drawing_card_reduces_cards_remaining(num_decks):
    deck = Deck(num_decks=num_decks, shuffle=False)

    old_remaining = deck.cards_remaining()
    card = deck.draw_card()

    assert card in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert deck.cards_remaining() == old_remaining - 1


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_drawing_card_updates_count_vector(num_decks):
    deck = Deck(num_decks=num_decks, shuffle=False)

    before = deck.get_count_vector()
    card = deck.draw_card()
    after = deck.get_count_vector()

    card_index = card - 1
    expected = list(before)
    expected[card_index] -= 1

    assert after == tuple(expected)


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_count_remaining_returns_count_for_specific_card_value(num_decks):
    deck = Deck(num_decks=num_decks, shuffle=False)

    assert deck.count_remaining(1) == 4 * num_decks
    assert deck.count_remaining(10) == 16 * num_decks

    card = deck.draw_card()

    assert deck.count_remaining(card) == expected_count_vector(num_decks)[card - 1] - 1


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_reset_does_not_reshuffle_when_enough_cards_remain(num_decks):
    threshold = 26
    deck = Deck(
        num_decks=num_decks,
        shuffle=False,
        reshuffle_threshold=threshold,
    )

    for _ in range(10):
        deck.draw_card()

    remaining_before = deck.cards_remaining()

    deck.reset()

    assert deck.cards_remaining() == remaining_before


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_reset_reshuffles_when_at_or_below_threshold(num_decks):
    threshold = 26
    deck = Deck(
        num_decks=num_decks,
        shuffle=False,
        reshuffle_threshold=threshold,
    )

    while deck.cards_remaining() > threshold:
        deck.draw_card()

    assert deck.cards_remaining() == threshold

    deck.reset()

    assert deck.cards_remaining() == 52 * num_decks
    assert deck.get_count_vector() == expected_count_vector(num_decks)


@pytest.mark.parametrize("num_decks", [1, 2, 4])
def test_force_reset_always_restores_full_deck(num_decks):
    deck = Deck(
        num_decks=num_decks,
        shuffle=False,
        reshuffle_threshold=26,
    )

    for _ in range(20):
        deck.draw_card()

    assert deck.cards_remaining() == 52 * num_decks - 20

    deck.force_reset()

    assert deck.cards_remaining() == 52 * num_decks
    assert deck.get_count_vector() == expected_count_vector(num_decks)


def test_count_vector_sum_matches_remaining_cards():
    deck = Deck(num_decks=3, shuffle=False)

    for _ in range(17):
        deck.draw_card()

    assert sum(deck.get_count_vector()) == deck.cards_remaining()


def test_draw_from_empty_deck_raises_runtime_error():
    deck = Deck(num_decks=1, shuffle=False)

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