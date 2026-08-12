"""Bankroll path and trip-level risk-of-ruin metrics for variable betting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BankrollPathResult:
    """One continuous bankroll walk over round P&L."""

    starting_bankroll: float
    ending_bankroll: float
    min_bankroll: float
    max_bankroll: float
    max_drawdown: float
    ruined: bool
    rounds_played: int
    rounds_to_ruin: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "starting_bankroll": self.starting_bankroll,
            "ending_bankroll": self.ending_bankroll,
            "min_bankroll": self.min_bankroll,
            "max_bankroll": self.max_bankroll,
            "max_drawdown": self.max_drawdown,
            "ruined": self.ruined,
            "rounds_played": self.rounds_played,
            "rounds_to_ruin": self.rounds_to_ruin,
        }


@dataclass(frozen=True)
class BankrollTripSummary:
    """Independent trips that each restart at ``starting_bankroll``."""

    starting_bankroll: float
    trip_rounds: int
    n_trips: int
    ruined_trips: int
    risk_of_ruin: float
    mean_ending_bankroll: float
    mean_max_drawdown: float
    mean_rounds_played: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "starting_bankroll": self.starting_bankroll,
            "trip_rounds": self.trip_rounds,
            "n_trips": self.n_trips,
            "ruined_trips": self.ruined_trips,
            "risk_of_ruin": self.risk_of_ruin,
            "mean_ending_bankroll": self.mean_ending_bankroll,
            "mean_max_drawdown": self.mean_max_drawdown,
            "mean_rounds_played": self.mean_rounds_played,
        }


def simulate_bankroll_path(
    rewards: Sequence[float],
    stakes: Sequence[float],
    *,
    starting_bankroll: float,
) -> BankrollPathResult:
    """Walk bankroll; ruin when cash cannot cover the next scheduled stake.

    After a round settles, bankroll updates by that round's net units. Before
    the next round, if ``bankroll < stake``, the path stops as ruined.
    """
    if starting_bankroll <= 0:
        raise ValueError("starting_bankroll must be positive")
    if len(rewards) != len(stakes):
        raise ValueError("rewards and stakes must have the same length")

    bankroll = float(starting_bankroll)
    peak = bankroll
    min_bankroll = bankroll
    max_bankroll = bankroll
    max_drawdown = 0.0
    rounds_played = 0
    rounds_to_ruin: int | None = None

    for reward, stake in zip(rewards, stakes):
        stake_f = float(stake)
        if stake_f <= 0:
            raise ValueError("stakes must be positive")
        if bankroll < stake_f:
            rounds_to_ruin = rounds_played
            return BankrollPathResult(
                starting_bankroll=float(starting_bankroll),
                ending_bankroll=bankroll,
                min_bankroll=min_bankroll,
                max_bankroll=max_bankroll,
                max_drawdown=max_drawdown,
                ruined=True,
                rounds_played=rounds_played,
                rounds_to_ruin=rounds_to_ruin,
            )

        bankroll += float(reward)
        rounds_played += 1
        if bankroll < min_bankroll:
            min_bankroll = bankroll
        if bankroll > max_bankroll:
            max_bankroll = bankroll
        if bankroll > peak:
            peak = bankroll
        drawdown = peak - bankroll
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if bankroll <= 0:
            rounds_to_ruin = rounds_played
            return BankrollPathResult(
                starting_bankroll=float(starting_bankroll),
                ending_bankroll=bankroll,
                min_bankroll=min_bankroll,
                max_bankroll=max_bankroll,
                max_drawdown=max_drawdown,
                ruined=True,
                rounds_played=rounds_played,
                rounds_to_ruin=rounds_to_ruin,
            )

    return BankrollPathResult(
        starting_bankroll=float(starting_bankroll),
        ending_bankroll=bankroll,
        min_bankroll=min_bankroll,
        max_bankroll=max_bankroll,
        max_drawdown=max_drawdown,
        ruined=False,
        rounds_played=rounds_played,
        rounds_to_ruin=None,
    )


def summarize_bankroll_trips(
    rewards: Sequence[float],
    stakes: Sequence[float],
    *,
    starting_bankroll: float,
    trip_rounds: int,
) -> BankrollTripSummary:
    """Split the round stream into fixed-length trips and estimate risk of ruin."""
    if trip_rounds <= 0:
        raise ValueError("trip_rounds must be positive")
    if len(rewards) != len(stakes):
        raise ValueError("rewards and stakes must have the same length")
    if not rewards:
        raise ValueError("rewards must be non-empty")

    n = len(rewards)
    n_trips = (n + trip_rounds - 1) // trip_rounds
    ruined = 0
    ending_sum = 0.0
    drawdown_sum = 0.0
    rounds_sum = 0

    for trip_i in range(n_trips):
        start = trip_i * trip_rounds
        end = min(start + trip_rounds, n)
        path = simulate_bankroll_path(
            rewards[start:end],
            stakes[start:end],
            starting_bankroll=starting_bankroll,
        )
        if path.ruined:
            ruined += 1
        ending_sum += path.ending_bankroll
        drawdown_sum += path.max_drawdown
        rounds_sum += path.rounds_played

    return BankrollTripSummary(
        starting_bankroll=float(starting_bankroll),
        trip_rounds=int(trip_rounds),
        n_trips=n_trips,
        ruined_trips=ruined,
        risk_of_ruin=ruined / n_trips,
        mean_ending_bankroll=ending_sum / n_trips,
        mean_max_drawdown=drawdown_sum / n_trips,
        mean_rounds_played=rounds_sum / n_trips,
    )


def bankroll_report(
    rewards: Sequence[float],
    stakes: Sequence[float],
    *,
    starting_bankroll: float,
    trip_rounds: int,
) -> dict[str, Any]:
    """Full-path stats plus trip-level risk-of-ruin for portfolio demos."""
    path = simulate_bankroll_path(
        rewards, stakes, starting_bankroll=starting_bankroll
    )
    trips = summarize_bankroll_trips(
        rewards,
        stakes,
        starting_bankroll=starting_bankroll,
        trip_rounds=trip_rounds,
    )
    return {
        "path": path.as_dict(),
        "trips": trips.as_dict(),
    }
