#!/usr/bin/env python3
"""Run Double DQN ablation study conditions A–D with JSON summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agents.cli_seeds import (
    add_seed_arguments,
    seed_artifact_dir,
    seeds_from_args,
    summarize_ablation_runs,
)
from agents.common import (
    agent_results_path,
    evaluate_greedy,
    neural_training_kwargs,
    run_neural_training_loop,
    save_torch_checkpoint,
    set_seed,
)
from agents.double_dqn import DoubleQNetworkLearningAgent
from agents.study_protocol import ABLATION_CONDITIONS
from config import (
    NEURAL_FINAL_EVAL_EPISODES,
    NEURAL_LEARNING_CURVE_FILENAME,
    NEURAL_TRAINING_EPISODES,
)
from game import BlackjackGame

SMOKE_EPISODES = 20
SMOKE_PRINT_INTERVAL = 5
SMOKE_CHECKPOINT_EVAL_EPISODES = 10
SMOKE_FINAL_EVAL_EPISODES = 100

WIN_CATEGORIES = frozenset(
    {"normal_win", "blackjack_win", "big_win_double_or_split"}
)
LOSS_CATEGORIES = frozenset(
    {"normal_loss", "big_loss_double_or_split"}
)


def ablation_base_dir() -> Path:
    """Directory for per-condition artifacts and the summary JSON."""
    return agent_results_path("double_dqn", "ablation")


def default_output_path() -> Path:
    return ablation_base_dir() / "ablation_results.json"


def distribution_to_rates(distribution: dict[str, int]) -> dict[str, float]:
    """Aggregate greedy-eval outcome counts into win/loss/draw rates."""
    total = sum(distribution.values()) or 1
    wins = sum(distribution.get(category, 0) for category in WIN_CATEGORIES)
    losses = sum(distribution.get(category, 0) for category in LOSS_CATEGORIES)
    draws = distribution.get("draw", 0)
    return {
        "win_rate": round(wins / total, 6),
        "loss_rate": round(losses / total, 6),
        "draw_rate": round(draws / total, 6),
    }


def smoke_loop_kwargs(episodes: int) -> dict[str, int]:
    """Training-loop overrides for fast CI smoke runs."""
    return {
        "phase_a_episodes": min(5, episodes // 2),
        "warmstart_episodes": min(5, episodes),
        "print_interval": SMOKE_PRINT_INTERVAL,
        "checkpoint_eval_episodes": SMOKE_CHECKPOINT_EVAL_EPISODES,
    }


def run_ablation_condition(
    condition_id: str,
    *,
    episodes: int,
    seed: int,
    smoke: bool = False,
    eval_episodes: int | None = None,
    ablation_base: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train one ablation condition and return a machine-readable summary row."""
    if condition_id not in ABLATION_CONDITIONS:
        raise KeyError(f"Unknown ablation condition: {condition_id}")

    spec = ABLATION_CONDITIONS[condition_id]
    training_episodes = SMOKE_EPISODES if smoke else episodes
    final_eval_episodes = (
        SMOKE_FINAL_EVAL_EPISODES
        if eval_episodes is None and smoke
        else NEURAL_FINAL_EVAL_EPISODES
        if eval_episodes is None
        else eval_episodes
    )

    base_dir = ablation_base_dir() if ablation_base is None else ablation_base
    condition_dir = base_dir / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)

    model_path = condition_dir / "model.pt"
    curve_path = condition_dir / NEURAL_LEARNING_CURVE_FILENAME

    set_seed(seed)
    game = BlackjackGame()
    agent = DoubleQNetworkLearningAgent(**neural_training_kwargs(), device=device)

    loop_kwargs: dict[str, Any] = {
        "curriculum": spec["curriculum"],
        "warmstart": spec["warmstart"],
        "force_shoe_off": spec["force_shoe_off"],
        "learning_curve_path": curve_path,
    }
    if smoke:
        loop_kwargs.update(smoke_loop_kwargs(training_episodes))

    run_neural_training_loop(agent, game, training_episodes, **loop_kwargs)
    save_torch_checkpoint(agent, model_path)

    average_reward, distribution = evaluate_greedy(
        agent,
        final_eval_episodes,
        seed=seed,
    )
    rates = distribution_to_rates(distribution)

    return {
        "condition_id": condition_id,
        "label": spec["label"],
        "average_reward": round(average_reward, 6),
        "training_steps": agent.training_steps,
        "win_rate": rates["win_rate"],
        "loss_rate": rates["loss_rate"],
        "draw_rate": rates["draw_rate"],
        "episodes": training_episodes,
        "eval_episodes": final_eval_episodes,
        "seed": seed,
        "curriculum": spec["curriculum"],
        "warmstart": spec["warmstart"],
        "force_shoe_off": spec["force_shoe_off"],
        "model_path": str(model_path),
        "learning_curve_path": str(curve_path),
    }


def write_ablation_results(
    results: list[dict[str, Any]],
    output_path: Path | str,
    *,
    seeds: list[int],
    smoke: bool,
) -> Path:
    """Persist the ablation summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seeds": seeds,
        "smoke": smoke,
        "conditions": results,
    }
    if len(seeds) == 1:
        payload["seed"] = seeds[0]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(ABLATION_CONDITIONS.keys()),
        choices=list(ABLATION_CONDITIONS.keys()),
        metavar="CONDITION",
        help="Ablation condition ids to run (default: all A–D)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NEURAL_TRAINING_EPISODES,
        help=f"Training episodes per condition (default: {NEURAL_TRAINING_EPISODES})",
    )
    add_seed_arguments(parser)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Fast CI mode: {SMOKE_EPISODES} training episodes, tiny eval/checkpoints"
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=(
            "Greedy eval episodes after training "
            f"(default: {SMOKE_FINAL_EVAL_EPISODES} in smoke, "
            f"{NEURAL_FINAL_EVAL_EPISODES} otherwise)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, mps, cuda). Default: CUDA if available else CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path for the ablation summary (default: under ablation base)",
    )
    args = parser.parse_args(argv)

    try:
        seeds = seeds_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    invalid = [condition for condition in args.conditions if condition not in ABLATION_CONDITIONS]
    if invalid:
        print(f"error: unknown conditions: {', '.join(invalid)}", file=sys.stderr)
        return 1

    multi = len(seeds) > 1
    root_base = ablation_base_dir()
    all_results: list[dict[str, Any]] = []
    per_seed_paths: list[dict[str, Any]] = []

    for seed in seeds:
        seed_base = seed_artifact_dir(root_base, seed, multi=multi)
        results: list[dict[str, Any]] = []
        for condition_id in args.conditions:
            print(
                f"\n=== Ablation {condition_id}: "
                f"{ABLATION_CONDITIONS[condition_id]['label']} (seed={seed}) ==="
            )
            row = run_ablation_condition(
                condition_id,
                episodes=args.episodes,
                seed=seed,
                smoke=args.smoke,
                eval_episodes=args.eval_episodes,
                ablation_base=seed_base,
                device=args.device,
            )
            results.append(row)
            all_results.append(row)
            print(
                f"average_reward={row['average_reward']:.4f} "
                f"training_steps={row['training_steps']} "
                f"win/loss/draw={row['win_rate']:.3f}/"
                f"{row['loss_rate']:.3f}/{row['draw_rate']:.3f}"
            )

        seed_output = (
            args.output
            if args.output is not None and not multi
            else seed_base / "ablation_results.json"
        )
        if args.output is not None and multi:
            seed_output = seed_artifact_dir(args.output.parent, seed, multi=True) / args.output.name
        write_ablation_results(
            results,
            seed_output,
            seeds=[seed],
            smoke=args.smoke,
        )
        per_seed_paths.append({"seed": seed, "output": str(seed_output)})
        print(f"\nWrote ablation summary to {seed_output}")

    if multi:
        aggregate_path = (
            args.output
            if args.output is not None
            else root_base / "multi_seed_ablation_results.json"
        )
        write_ablation_results(
            all_results,
            aggregate_path,
            seeds=seeds,
            smoke=args.smoke,
        )
        # Attach per-seed path index and mean/std summary for consumers.
        payload = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
        payload["per_seed"] = per_seed_paths
        payload["summary"] = summarize_ablation_runs(all_results, seeds)
        Path(aggregate_path).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote multi-seed aggregate to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
