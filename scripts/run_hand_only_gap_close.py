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
  - After training, ``hand_only_gap_close_model.pt`` can be re-evaluated with
    ``--eval-only`` (no retraining).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.cli_seeds import add_seed_arguments, seed_artifact_dir, seeds_from_args
from agents.common import (
    agent_results_path,
    evaluate_greedy,
    load_torch_checkpoint,
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


def _outcome_rates(distribution: dict[str, int], episodes: int) -> dict[str, float]:
    wins = sum(
        distribution.get(category, 0)
        for category in (
            "normal_win",
            "blackjack_win",
            "big_win_double_or_split",
        )
    )
    losses = sum(
        distribution.get(category, 0)
        for category in (
            "normal_loss",
            "big_loss_double_or_split",
        )
    )
    draws = distribution.get("draw", 0)
    return {
        "win_rate": round(wins / episodes, 6),
        "loss_rate": round(losses / episodes, 6),
        "draw_rate": round(draws / episodes, 6),
    }


def build_gap_close_summary(
    *,
    seed: int,
    smoke: bool,
    agent,
    agent_reward: float,
    agent_dist: dict[str, int],
    rule_reward: float,
    rule_dist: dict[str, int],
    warmstart_episodes: int,
    train_episodes: int,
    eval_episodes: int,
    model_path: Path,
    learning_curve_path: Path,
    trained: bool,
) -> dict:
    """Serialize a paired gap-close eval (with or without a fresh train)."""
    return {
        "seed": seed,
        "smoke": smoke,
        "hand_only_encoder": True,
        "state_size": agent.input_size,
        "warmstart_episodes": warmstart_episodes,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "epsilon_min": GAP_CLOSE_EPSILON_MIN,
        "paired_eval": True,
        "trained": trained,
        "agent": {
            "average_reward": round(agent_reward, 6),
            "training_steps": agent.training_steps,
            "distribution": dict(agent_dist),
            **_outcome_rates(agent_dist, eval_episodes),
        },
        "rule_baseline": {
            "average_reward": round(rule_reward, 6),
            "distribution": dict(rule_dist),
            **_outcome_rates(rule_dist, eval_episodes),
        },
        "gap": round(agent_reward - rule_reward, 6),
        "model_path": str(model_path),
        "learning_curve_path": str(learning_curve_path),
    }


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
    eval_only: bool = False,
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
    if results_dir is None and not eval_only:
        guard_full_results_overwrite(out_dir, force=force)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    curve_path = out_dir / NEURAL_LEARNING_CURVE_FILENAME
    summary_path = out_dir / SUMMARY_FILENAME

    if eval_only:
        if not model_path.exists():
            raise SystemExit(
                f"--eval-only requires an existing checkpoint at {model_path}"
            )
        print(
            f"Gap-close eval-only: loading {model_path}, "
            f"eval={eval_episodes}, seed={seed}"
        )
        agent, checkpoint = load_torch_checkpoint(
            DoubleQNetworkLearningAgent,
            model_path,
            device="cpu",
            hand_only_encoder=True,
            **{**neural_training_kwargs(), "epsilon_min": GAP_CLOSE_EPSILON_MIN},
        )
        if "warmstart_episodes" in checkpoint:
            warmstart_episodes = int(checkpoint["warmstart_episodes"])
        if "train_episodes" in checkpoint:
            train_episodes = int(checkpoint["train_episodes"])
        trained = False
    else:
        set_seed(seed)
        kwargs = neural_training_kwargs()
        kwargs["epsilon_min"] = GAP_CLOSE_EPSILON_MIN
        agent = DoubleQNetworkLearningAgent(hand_only_encoder=True, **kwargs)
        game = BlackjackGame()

        # Fresh curve file so smoke leftovers / prior runs do not append.
        if curve_path.exists():
            curve_path.unlink()

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
        save_torch_checkpoint(
            agent,
            model_path,
            extra={
                "warmstart_episodes": warmstart_episodes,
                "train_episodes": train_episodes,
                "seed": seed,
            },
        )
        trained = True

    agent_reward, agent_dist = evaluate_greedy(agent, eval_episodes, seed=seed)
    rule_agent = RuleAgent()
    rule_reward, rule_dist = evaluate_greedy(rule_agent, eval_episodes, seed=seed)

    summary = build_gap_close_summary(
        seed=seed,
        smoke=smoke,
        agent=agent,
        agent_reward=agent_reward,
        agent_dist=agent_dist,
        rule_reward=rule_reward,
        rule_dist=rule_dist,
        warmstart_episodes=warmstart_episodes,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        model_path=model_path,
        learning_curve_path=curve_path,
        trained=trained,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nHand-only agent: {agent_reward:.4f}  "
        f"Rule baseline: {rule_reward:.4f}  "
        f"Gap (agent - rule): {summary['gap']:.4f}"
    )
    print(f"Checkpoint: {model_path}")
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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load hand_only_gap_close_model.pt and re-run paired eval",
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
        # Only guard the legacy single-seed full path when training.
        if not multi and not args.smoke and not args.eval_only:
            guard_full_results_overwrite(out_dir, force=args.force)
        summary = run_gap_close(
            seed=seed,
            warmstart_episodes=args.warmstart_episodes,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            smoke=args.smoke,
            force=True if multi else args.force,
            eval_only=args.eval_only,
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
