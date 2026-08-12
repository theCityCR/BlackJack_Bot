"""Scripted true-count → stake schedules for variable betting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agents.counting import true_count_from_shoe
from config import BET_MAX, BET_MIN, NUM_DECKS
from game import ShoeObservation


@dataclass(frozen=True)
class TrueCountBetSchedule:
    """Map floor(true count) to stake units in ``[bet_min, bet_max]``.

    Default ramp (units): TC≤0→1, 1→2, 2→4, 3→6, ≥4→8.
    """

    bet_min: int = BET_MIN
    bet_max: int = BET_MAX
    num_decks: int = NUM_DECKS
    # Floor TC → stake. Applied in ascending threshold order.
    schedule: tuple[tuple[int, int], ...] = (
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
    )

    def __post_init__(self) -> None:
        if self.bet_min <= 0 or self.bet_max < self.bet_min:
            raise ValueError("bet_min/bet_max must satisfy 0 < bet_min <= bet_max")
        for _, stake in self.schedule:
            if stake < self.bet_min or stake > self.bet_max:
                raise ValueError("schedule stakes must lie within bet_min..bet_max")

    def choose_bet(self, shoe: ShoeObservation) -> float:
        floor_tc = math.floor(
            true_count_from_shoe(shoe, num_decks=self.num_decks)
        )
        stake = self.bet_min
        for threshold, units in self.schedule:
            if floor_tc >= threshold:
                stake = units
        return float(stake)


@dataclass(frozen=True)
class FlatBetSchedule:
    """Always bet a fixed stake (default flat unit)."""

    bet: float = 1.0

    def choose_bet(self, shoe: ShoeObservation) -> float:
        del shoe  # unused; flat stake ignores the shoe
        if self.bet <= 0:
            raise ValueError("bet must be positive")
        return float(self.bet)
