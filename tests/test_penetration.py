"""Tests for shoe size / dealt-penetration helpers."""

from __future__ import annotations

import pytest

from cards import dealt_penetration, shoe_size
from config import NUM_DECKS, RESHUFFLE_WHEN_CARDS_REMAINING_BELOW
from game import BlackjackGame


def test_shoe_size_matches_decks():
    assert shoe_size(1) == 52
    assert shoe_size(2) == 104
    assert shoe_size(NUM_DECKS) == 52 * NUM_DECKS


def test_default_cut_is_about_75_percent_dealt():
    pen = dealt_penetration(RESHUFFLE_WHEN_CARDS_REMAINING_BELOW, NUM_DECKS)
    assert pen == pytest.approx(0.75)


def test_deeper_cut_raises_dealt_penetration():
    assert dealt_penetration(13, 2) > dealt_penetration(26, 2)
    assert dealt_penetration(52, 2) == pytest.approx(0.5)


def test_blackjack_game_accepts_reshuffle_threshold():
    game = BlackjackGame(reshuffle_threshold=13)
    assert game.reshuffle_threshold == 13
    assert game.deck.reshuffle_threshold == 13


def test_blackjack_game_rejects_deck_and_threshold_together():
    from cards import Deck

    with pytest.raises(ValueError, match="pass deck or shoe kwargs"):
        BlackjackGame(Deck(), reshuffle_threshold=13)
