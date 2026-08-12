#!/usr/bin/env python3
"""Paired eval: flat unit rule vs Hi-Lo spread + rule play.

Counting needs a **persistent shoe** across rounds (fresh shoes always start at
TC≈0). Each shoe session is seeded for pairing; both policies then play the
same number of consecutive rounds. With identical rule play, card sequences
stay aligned so flat vs spread is a fair paired comparison.

Reports EV per round, EV per unit wagered, and spread utilization by true count.
Does not modify published flat-bet artifacts under docs/results/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents.betting import FlatBetSchedule, TrueCountBetSchedule
from agents.cli_seeds import add_seed_arguments, seeds_from_args, summarize_variable_betting_runs
from agents.rule import RuleAgent
from agents.spread_rule import SpreadRuleAgent
from game import BlackjackGame

DEFAULT_ROUNDS_PER_SHOE = 100
DEFAULT_MULTI_SEED_OUTPUT = Path(
    "agents/results/variable_betting/multi_seed_variable_betting_results.json"
)


def make_shoe_session(base_seed: int, session_index: int) -> BlackjackGame:
    """Fresh shoe shuffled under the same pairing key as agents.common."""
    seed = (int(base_seed) + int(session_index) * 1_000_003) & 0x7FFFFFFF
    random.seed(seed)
    return BlackjackGame()


def _bucket_true_count(tc: float) -> str:
    floor_tc = math.floor(tc)
    if floor_tc <= 0:
        return "<=0"
    if floor_tc >= 4:
        return ">=4"
    return str(floor_tc)


def evaluate_spread_policy(
    agent: SpreadRuleAgent,
    episodes: int,
    *,
    seed: int,
    rounds_per_shoe: int = DEFAULT_ROUNDS_PER_SHOE,
) -> dict[str, Any]:
    """Run consecutive rounds on seeded shoes and collect betting metrics."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if rounds_per_shoe <= 0:
        raise ValueError("rounds_per_shoe must be positive")

    total_reward = 0.0
    total_stake = 0.0
    bet_counts: dict[str, int] = defaultdict(int)
    tc_bucket_counts: dict[str, int] = defaultdict(int)
    tc_bucket_stake: dict[str, float] = defaultdict(float)
    outcome: dict[str, int] = defaultdict(int)

    sessions = math.ceil(episodes / rounds_per_shoe)
    rounds_played = 0
    session_index = 0
    while rounds_played < episodes:
        game = make_shoe_session(seed, session_index)
        session_index += 1
        rounds_this_shoe = min(rounds_per_shoe, episodes - rounds_played)
        for _ in range(rounds_this_shoe):
            reward = agent.play_episode(game)
            stake = float(agent.last_bet)
            total_reward += reward
            total_stake += stake
            bet_key = str(int(stake) if stake == int(stake) else stake)
            bet_counts[bet_key] += 1
            bucket = _bucket_true_count(agent.last_true_count)
            tc_bucket_counts[bucket] += 1
            tc_bucket_stake[bucket] += stake
            if reward > 0:
                outcome["win"] += 1
            elif reward < 0:
                outcome["loss"] += 1
            else:
                outcome["draw"] += 1
            rounds_played += 1

    return {
        "episodes": episodes,
        "seed": seed,
        "rounds_per_shoe": rounds_per_shoe,
        "shoe_sessions": sessions,
        "average_reward": total_reward / episodes,
        "average_stake": total_stake / episodes,
        "ev_per_unit_wagered": total_reward / total_stake if total_stake else 0.0,
        "total_reward": total_reward,
        "total_stake": total_stake,
        "bet_fraction": {
            bet: count / episodes for bet, count in sorted(bet_counts.items())
        },
        "true_count_fraction": {
            bucket: tc_bucket_counts[bucket] / episodes
            for bucket in ("<=0", "1", "2", "3", ">=4")
            if bucket in tc_bucket_counts
        },
        "true_count_average_stake": {
            bucket: tc_bucket_stake[bucket] / tc_bucket_counts[bucket]
            for bucket in ("<=0", "1", "2", "3", ">=4")
            if bucket in tc_bucket_counts
        },
        "outcome": dict(outcome),
    }


def run_comparison(
    episodes: int,
    seed: int,
    *,
    rounds_per_shoe: int = DEFAULT_ROUNDS_PER_SHOE,
) -> dict[str, Any]:
    flat = SpreadRuleAgent(bet_policy=FlatBetSchedule(bet=1.0))
    spread = SpreadRuleAgent(bet_policy=TrueCountBetSchedule())
    flat_stats = evaluate_spread_policy(
        flat, episodes, seed=seed, rounds_per_shoe=rounds_per_shoe
    )
    spread_stats = evaluate_spread_policy(
        spread, episodes, seed=seed, rounds_per_shoe=rounds_per_shoe
    )
    return {
        "episodes": episodes,
        "seed": seed,
        "rounds_per_shoe": rounds_per_shoe,
        "paired_eval": True,
        "flat_rule": flat_stats,
        "spread_rule": spread_stats,
        "delta_average_reward": (
            spread_stats["average_reward"] - flat_stats["average_reward"]
        ),
        "play_agent": "RuleAgent",
        "bet_schedule": "TrueCountBetSchedule(default 1-8)",
    }


def compact_run_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a comparison summary for multi-seed aggregation."""
    flat = summary["flat_rule"]
    spread = summary["spread_rule"]
    return {
        "seed": summary["seed"],
        "episodes": summary["episodes"],
        "rounds_per_shoe": summary["rounds_per_shoe"],
        "flat_average_reward": flat["average_reward"],
        "spread_average_reward": spread["average_reward"],
        "spread_ev_per_unit_wagered": spread["ev_per_unit_wagered"],
        "spread_average_stake": spread["average_stake"],
        "delta_average_reward": summary["delta_average_reward"],
        "spread_bet_fraction": spread["bet_fraction"],
        "true_count_fraction": spread["true_count_fraction"],
    }


def print_comparison(summary: dict[str, Any]) -> None:
    flat = summary["flat_rule"]
    spread = summary["spread_rule"]
    print(
        f"Episodes: {summary['episodes']}  seed={summary['seed']}  "
        f"rounds/shoe={summary['rounds_per_shoe']}  paired"
    )
    print(
        f"Flat rule:   EV/round={flat['average_reward']:+.4f}  "
        f"EV/unit={flat['ev_per_unit_wagered']:+.4f}  "
        f"avg stake={flat['average_stake']:.2f}"
    )
    print(
        f"Spread rule: EV/round={spread['average_reward']:+.4f}  "
        f"EV/unit={spread['ev_per_unit_wagered']:+.4f}  "
        f"avg stake={spread['average_stake']:.2f}"
    )
    print(f"Delta EV/round (spread − flat): {summary['delta_average_reward']:+.4f}")
    print(f"Spread bet mix: {spread['bet_fraction']}")
    print(f"True-count mix: {spread['true_count_fraction']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    add_seed_arguments(parser)
    parser.add_argument(
        "--rounds-per-shoe",
        type=int,
        default=DEFAULT_ROUNDS_PER_SHOE,
        help="Consecutive rounds per seeded shoe session (counting needs penetration)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON path (single-seed comparison, or multi-seed aggregate). "
            "Multi-seed default: agents/results/variable_betting/"
            "multi_seed_variable_betting_results.json"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short paired run (500 episodes) for CI",
    )
    args = parser.parse_args()

    episodes = 500 if args.smoke else args.episodes
    seeds = seeds_from_args(args)
    multi = len(seeds) > 1
    _ = RuleAgent

    if not multi:
        summary = run_comparison(
            episodes, seeds[0], rounds_per_shoe=args.rounds_per_shoe
        )
        print_comparison(summary)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Wrote {args.output}")
        return

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        summary = run_comparison(
            episodes, seed, rounds_per_shoe=args.rounds_per_shoe
        )
        print_comparison(summary)
        print("---")
        runs.append(compact_run_row(summary))

    payload = {
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "episodes": episodes,
        "rounds_per_shoe": args.rounds_per_shoe,
        "paired_eval": True,
        "play_agent": "RuleAgent",
        "bet_schedule": "TrueCountBetSchedule(default 1-8)",
        "runs": runs,
        "summary": summarize_variable_betting_runs(runs, seeds),
        "artifact_note": (
            "Multi-seed aggregate for rule + Hi-Lo spread. "
            "Copy to docs/results/ only when publishing; "
            "does not modify flat-bet ablation/gap-close tables."
        ),
    }
    out = args.output or DEFAULT_MULTI_SEED_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")
    s = payload["summary"]
    print(
        "Mean ± std EV/round — "
        f"flat {s['flat_average_reward']['mean']:+.4f}±{s['flat_average_reward']['std']:.4f}, "
        f"spread {s['spread_average_reward']['mean']:+.4f}±{s['spread_average_reward']['std']:.4f}, "
        f"delta {s['delta_average_reward']['mean']:+.4f}±{s['delta_average_reward']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
