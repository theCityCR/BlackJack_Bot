#!/usr/bin/env python3
"""Close the gap to the rule baseline with a true hand-only Double DQN run.

Protocol:
  - 8-D hand encoder (no shoe dims)
  - 100k rule-agent warm-start (behavior cloning)
  - 500k RL episodes, shoe features never enabled
  - Final greedy eval vs rule baseline on paired per-episode shoes (same seed)

Artifact layout:
  - Full runs write under ``agents/results/double_dqn/gap_close/``
  - ``--smoke`` writes under ``.../gap_close_smoke/`` so CI cannot clobber a
    full checkpoint. Overwriting an existing non-smoke summary requires
    ``--force``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.cli_seeds import add_seed_arguments, seed_artifact_dir, seeds_from_args
from agents.common import (
    agent_results_path,
    evaluate_greedy,
    neural_training_kwargs,
    run_neural_training_loop,
    save_torch_checkpoint,
    set_seed,
)
from agents.double_dqn import DoubleQNetworkLearningAgent
from agents.rule import RuleAgent
from config import (
    GAP_CLOSE_CHECKPOINT_EVAL_EPISODES,
    GAP_CLOSE_EPSILON_MIN,
    GAP_CLOSE_EVAL_EPISODES,
    GAP_CLOSE_PRINT_INTERVAL,
    GAP_CLOSE_TRAINING_EPISODES,
    GAP_CLOSE_WARMSTART_EPISODES,
    NEURAL_LEARNING_CURVE_FILENAME,
)
from game import BlackjackGame

FULL_RESULTS_DIR = agent_results_path("double_dqn", "gap_close")
SMOKE_RESULTS_DIR = agent_results_path("double_dqn", "gap_close_smoke")
SUMMARY_FILENAME = "gap_close_results.json"
MODEL_FILENAME = "hand_only_gap_close_model.pt"


def results_dir_for(*, smoke: bool) -> Path:
    """Resolve the artifact directory for a smoke or full gap-close run."""
    return SMOKE_RESULTS_DIR if smoke else FULL_RESULTS_DIR


def guard_full_results_overwrite(results_dir: Path, *, force: bool) -> None:
    """Refuse to clobber a previous non-smoke gap-close summary without --force."""
    if results_dir.resolve() != FULL_RESULTS_DIR.resolve():
        return

    summary_path = results_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if payload.get("smoke", True):
        return
    if force:
        return

    raise SystemExit(
        f"Refusing to overwrite full gap-close artifacts in {results_dir}. "
        "Re-run with --force if you intend to replace them, or keep using "
        "--smoke (writes to gap_close_smoke/)."
    )


def run_gap_close(
    *,
    seed: int = 42,
    warmstart_episodes: int = GAP_CLOSE_WARMSTART_EPISODES,
    train_episodes: int = GAP_CLOSE_TRAINING_EPISODES,
    eval_episodes: int = GAP_CLOSE_EVAL_EPISODES,
    smoke: bool = False,
    force: bool = False,
    results_dir: Path | None = None,
) -> dict:
    if smoke:
        warmstart_episodes = min(50, warmstart_episodes)
        train_episodes = min(200, train_episodes)
        eval_episodes = min(200, eval_episodes)
        print_interval = 50
        checkpoint_eval = 20
    else:
        print_interval = GAP_CLOSE_PRINT_INTERVAL
        checkpoint_eval = GAP_CLOSE_CHECKPOINT_EVAL_EPISODES

    out_dir = results_dir if results_dir is not None else results_dir_for(smoke=smoke)
    if results_dir is None:
        guard_full_results_overwrite(out_dir, force=force)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    curve_path = out_dir / NEURAL_LEARNING_CURVE_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME

    set_seed(seed)
    kwargs = neural_training_kwargs()
    kwargs["epsilon_min"] = GAP_CLOSE_EPSILON_MIN
    agent = DoubleQNetworkLearningAgent(hand_only_encoder=True, **kwargs)
    game = BlackjackGame()

    print(
        f"Gap-close protocol: 8-D hand encoder, warmstart={warmstart_episodes}, "
        f"train={train_episodes}, eval={eval_episodes}, seed={seed}, "
        f"artifacts={out_dir}"
    )

    run_neural_training_loop(
        agent,
        game,
        train_episodes,
        curriculum=False,
        warmstart=True,
        warmstart_episodes=warmstart_episodes,
        force_shoe_off=True,
        learning_curve_path=curve_path,
        print_interval=print_interval,
        checkpoint_eval_episodes=checkpoint_eval,
    )
    save_torch_checkpoint(agent, model_path)

    agent_reward, agent_dist = evaluate_greedy(agent, eval_episodes, seed=seed)
    rule_agent = RuleAgent()
    rule_reward, rule_dist = evaluate_greedy(rule_agent, eval_episodes, seed=seed)

    summary = {
        "seed": seed,
        "smoke": smoke,
        "hand_only_encoder": True,
        "state_size": agent.input_size,
        "warmstart_episodes": warmstart_episodes,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "epsilon_min": GAP_CLOSE_EPSILON_MIN,
        "agent": {
            "average_reward": round(agent_reward, 6),
            "training_steps": agent.training_steps,
            "distribution": dict(agent_dist),
        },
        "rule_baseline": {
            "average_reward": round(rule_reward, 6),
            "distribution": dict(rule_dist),
        },
        "gap": round(agent_reward - rule_reward, 6),
        "model_path": str(model_path),
        "learning_curve_path": str(curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nHand-only agent: {agent_reward:.4f}  "
        f"Rule baseline: {rule_reward:.4f}  "
        f"Gap (agent - rule): {summary['gap']:.4f}"
    )
    print(f"Wrote {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_seed_arguments(parser)
    parser.add_argument(
        "--warmstart-episodes",
        type=int,
        default=GAP_CLOSE_WARMSTART_EPISODES,
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=GAP_CLOSE_TRAINING_EPISODES,
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=GAP_CLOSE_EVAL_EPISODES,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny episode counts for CI (writes to gap_close_smoke/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing non-smoke gap_close/ summary",
    )
    args = parser.parse_args(argv)
    try:
        seeds = seeds_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    multi = len(seeds) > 1
    base = results_dir_for(smoke=args.smoke)
    summaries: list[dict] = []

    for seed in seeds:
        out_dir = seed_artifact_dir(base, seed, multi=multi)
        # Only guard the legacy single-seed full path.
        if not multi and not args.smoke:
            guard_full_results_overwrite(out_dir, force=args.force)
        summary = run_gap_close(
            seed=seed,
            warmstart_episodes=args.warmstart_episodes,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            smoke=args.smoke,
            force=True if multi else args.force,
            results_dir=out_dir,
        )
        summaries.append(summary)

    if multi:
        aggregate_path = base / "multi_seed_gap_close_results.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "seeds": seeds,
                    "smoke": args.smoke,
                    "runs": [
                        {
                            "seed": row["seed"],
                            "gap": row["gap"],
                            "agent_average_reward": row["agent"]["average_reward"],
                            "rule_average_reward": row["rule_baseline"]["average_reward"],
                            "model_path": row["model_path"],
                            "learning_curve_path": row["learning_curve_path"],
                        }
                        for row in summaries
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote multi-seed aggregate to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
