#!/usr/bin/env python3
"""Paired bake-off: bet+play PG agents vs flat rule and Hi-Lo spread rule.

Loads one or more PG checkpoints, runs the same multi-seed consecutive-shoe
protocol as §5.5, and writes a combined aggregate for publishing (§5.7).

Does not modify flat-bet ablation/gap-close tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.cli_seeds import parse_seeds, summarize_variable_betting_runs
from agents.common import agent_results_path
import scripts.run_variable_betting_eval as vb_eval

DEFAULT_OUTPUT = Path("docs/results/pg_spread_bakeoff_results.json")
DEFAULT_AGENTS = ("reinforce", "a2c", "ppo")
DEFAULT_CHECKPOINTS = {
    "reinforce": agent_results_path("reinforce", "reinforce_bet_play_model.pt"),
    "a2c": agent_results_path("a2c", "a2c_bet_play_model.pt"),
    "ppo": agent_results_path("ppo", "ppo_bet_play_model.pt"),
}


def eval_agent(
    agent_name: str,
    checkpoint: Path,
    *,
    episodes: int,
    seeds: list[int],
    rounds_per_shoe: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        agent = vb_eval.load_pg_agent(agent_name, checkpoint)
        summary = vb_eval.run_comparison(
            episodes,
            seed,
            rounds_per_shoe=rounds_per_shoe,
            pg_agent=agent,
            pg_agent_name=agent_name,
        )
        vb_eval.print_comparison(summary)
        print("---")
        runs.append(vb_eval.compact_run_row(summary))
    return {
        "pg_agent": agent_name,
        "pg_checkpoint": str(checkpoint),
        "seeds": seeds,
        "episodes": episodes,
        "rounds_per_shoe": rounds_per_shoe,
        "paired_eval": True,
        "runs": runs,
        "summary": summarize_variable_betting_runs(runs, seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument(
        "--rounds-per-shoe",
        type=int,
        default=vb_eval.DEFAULT_ROUNDS_PER_SHOE,
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=",".join(DEFAULT_AGENTS),
        help="Comma-separated PG agent names (reinforce,a2c,ppo)",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Recorded train seed for the published artifact metadata",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=200_000,
        help="Recorded train episode budget for artifact metadata",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under agents/results/<agent>/ "
            "(e.g. bet_focus for post-§5.7 stake-retention trains)."
        ),
    )
    parser.add_argument(
        "--bet-focus",
        action="store_true",
        help="Shorthand for --checkpoint-subdir bet_focus and train-episodes 500000.",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help=(
            "Shorthand for --checkpoint-subdir unfreeze and train-episodes 200000 "
            "(phase-2 after bet-focus)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Combined JSON path (default {DEFAULT_OUTPUT}; "
            "docs/results/pg_bet_focus_bakeoff_results.json with --bet-focus; "
            "docs/results/pg_unfreeze_bakeoff_results.json with --unfreeze)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short CI bake-off (500 episodes, seed 42 only)",
    )
    args = parser.parse_args()

    if args.bet_focus and args.unfreeze:
        parser.error("--bet-focus and --unfreeze are mutually exclusive")
    if args.bet_focus and args.checkpoint_subdir is None:
        args.checkpoint_subdir = "bet_focus"
    if args.unfreeze and args.checkpoint_subdir is None:
        args.checkpoint_subdir = "unfreeze"
    if args.bet_focus and args.train_episodes == 200_000:
        args.train_episodes = 500_000
    if args.unfreeze and args.train_episodes == 200_000:
        args.train_episodes = 200_000
    if args.output is None:
        if args.bet_focus:
            args.output = Path("docs/results/pg_bet_focus_bakeoff_results.json")
        elif args.unfreeze:
            args.output = Path("docs/results/pg_unfreeze_bakeoff_results.json")
        else:
            args.output = DEFAULT_OUTPUT

    try:
        seeds = (
            [42]
            if args.smoke
            else parse_seeds(seed=args.seed, seeds=args.seeds)
        )
    except ValueError as exc:
        parser.error(str(exc))

    agent_names = [p.strip() for p in args.agents.split(",") if p.strip()]
    if not agent_names:
        parser.error("--agents must be non-empty")
    for name in agent_names:
        if name not in DEFAULT_CHECKPOINTS:
            parser.error(f"unknown agent {name!r}; expected one of {DEFAULT_AGENTS}")

    episodes = 500 if args.smoke else args.episodes
    agent_blocks: list[dict[str, Any]] = []
    for name in agent_names:
        checkpoint = Path(DEFAULT_CHECKPOINTS[name])
        if args.checkpoint_subdir:
            checkpoint = checkpoint.parent / args.checkpoint_subdir / checkpoint.name
        if not checkpoint.is_file():
            parser.error(f"missing checkpoint for {name}: {checkpoint}")
        print(f"\n===== Bake-off {name}  checkpoint={checkpoint} =====")
        agent_blocks.append(
            eval_agent(
                name,
                checkpoint,
                episodes=episodes,
                seeds=seeds,
                rounds_per_shoe=args.rounds_per_shoe,
            )
        )

    if args.bet_focus or args.checkpoint_subdir == "bet_focus":
        artifact_note = (
            "Bet-focus PG vs rule+Hi-Lo bake-off (500k freeze-play / teacher CE). "
            "Does not replace published §5.7 "
            "docs/results/pg_spread_bakeoff_results.json."
        )
    elif args.unfreeze or args.checkpoint_subdir == "unfreeze":
        artifact_note = (
            "Unfreeze-after-bet-focus PG bake-off (200k thawed play / teacher CE). "
            "Does not replace published §5.7 / §5.8 bake-off JSONs."
        )
    else:
        artifact_note = (
            "Published §5.7 PG vs rule+Hi-Lo bake-off. "
            "Checkpoints under agents/results/<agent>/ remain gitignored; "
            "does not modify flat-bet or §5.5 multi-seed tables."
        )

    payload: dict[str, Any] = {
        "smoke": bool(args.smoke),
        "train_seed": args.train_seed,
        "train_episodes": args.train_episodes,
        "bet_focus": bool(args.bet_focus or args.checkpoint_subdir == "bet_focus"),
        "unfreeze": bool(args.unfreeze or args.checkpoint_subdir == "unfreeze"),
        "checkpoint_subdir": args.checkpoint_subdir,
        "warmstart": True,
        "eval_episodes": episodes,
        "eval_seeds": seeds,
        "rounds_per_shoe": args.rounds_per_shoe,
        "paired_eval": True,
        "agents": agent_blocks,
        "artifact_note": artifact_note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for block in agent_blocks:
        s = block["summary"]
        name = block["pg_agent"]
        print(
            f"{name}: PG {s['pg_average_reward']['mean']:+.4f}±"
            f"{s['pg_average_reward']['std']:.4f}  "
            f"spread {s['spread_average_reward']['mean']:+.4f}±"
            f"{s['spread_average_reward']['std']:.4f}  "
            f"ΔPG−spread {s['delta_pg_minus_spread']['mean']:+.4f}±"
            f"{s['delta_pg_minus_spread']['std']:.4f}"
        )


if __name__ == "__main__":
    main()
