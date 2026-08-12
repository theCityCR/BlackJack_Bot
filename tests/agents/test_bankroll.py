"""Tests for agents/bankroll.py."""

from __future__ import annotations

import pytest

from agents.bankroll import (
    bankroll_report,
    simulate_bankroll_path,
    summarize_bankroll_trips,
)


def test_path_survives_when_bankroll_covers_stakes():
    path = simulate_bankroll_path(
        rewards=[1.0, -1.0, 2.0],
        stakes=[1.0, 1.0, 1.0],
        starting_bankroll=10.0,
    )
    assert path.ruined is False
    assert path.ending_bankroll == 12.0
    assert path.min_bankroll == 10.0
    assert path.max_drawdown == 1.0
    assert path.rounds_to_ruin is None


def test_path_ruins_when_stake_exceeds_cash():
    path = simulate_bankroll_path(
        rewards=[-5.0, -1.0],
        stakes=[5.0, 8.0],
        starting_bankroll=10.0,
    )
    assert path.ruined is True
    assert path.rounds_played == 1
    assert path.rounds_to_ruin == 1
    assert path.ending_bankroll == 5.0


def test_path_ruins_at_non_positive_bankroll():
    path = simulate_bankroll_path(
        rewards=[-10.0],
        stakes=[1.0],
        starting_bankroll=10.0,
    )
    assert path.ruined is True
    assert path.ending_bankroll == 0.0
    assert path.rounds_to_ruin == 1


def test_trip_risk_of_ruin_counts_independent_restarts():
    # Trip 1: ruin after losing the unit bankroll.
    # Trip 2: small win, survives.
    rewards = [-1.0, 1.0]
    stakes = [1.0, 1.0]
    trips = summarize_bankroll_trips(
        rewards,
        stakes,
        starting_bankroll=1.0,
        trip_rounds=1,
    )
    assert trips.n_trips == 2
    assert trips.ruined_trips == 1
    assert trips.risk_of_ruin == 0.5


def test_bankroll_report_includes_path_and_trips():
    report = bankroll_report(
        rewards=[1.0, -2.0, 1.0, 1.0],
        stakes=[1.0, 1.0, 1.0, 1.0],
        starting_bankroll=5.0,
        trip_rounds=2,
    )
    assert "path" in report and "trips" in report
    assert report["trips"]["n_trips"] == 2
    assert report["path"]["starting_bankroll"] == 5.0


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_bankroll_path([1.0], [1.0], starting_bankroll=0.0)
    with pytest.raises(ValueError):
        simulate_bankroll_path([1.0], [1.0, 2.0], starting_bankroll=10.0)
    with pytest.raises(ValueError):
        summarize_bankroll_trips([1.0], [1.0], starting_bankroll=10.0, trip_rounds=0)
