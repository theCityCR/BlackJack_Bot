"""Shared pytest helpers for Blackjack agent/env tests."""

from __future__ import annotations

from game import GameState

def make_state(
    player_value=16,
    dealer_upcard=10,
    usable_ace=False,
    can_double=True,
    can_split=False,
    is_split_hand=False,
    active_hand_index=0,
    num_hands=1,
    count_vector=(4, 4, 4, 4, 4, 4, 4, 4, 4, 16),
) -> GameState:
    return GameState(
        player_value=player_value,
        dealer_upcard=dealer_upcard,
        usable_ace=usable_ace,
        can_double=can_double,
        can_split=can_split,
        is_split_hand=is_split_hand,
        active_hand_index=active_hand_index,
        num_hands=num_hands,
        count_vector=count_vector,
    )
