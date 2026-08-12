"""Shared CLI helpers for bet+play policy-gradient trainers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from agents.cli_seeds import add_seed_arguments, seed_artifact_dir, seeds_from_args
from agents.common import (
    agent_results_path,
    evaluate_greedy,
    print_distribution,
    save_policy_checkpoint,
    set_seed,
)
from agents.learning_curves import LearningCurveLogger
from agents.pg_warmstart import warmstart_from_spread_rule
from config import (
    PG_CHECKPOINT_EVAL_EPISODES,
    PG_FINAL_EVAL_EPISODES,
    PG_LEARNING_CURVE_ENABLED,
    PG_LEARNING_CURVE_FILENAME,
    PG_PRINT_INTERVAL,
    PG_TRAINING_EPISODES,
    PG_WARMSTART_ENABLED,
    PG_WARMSTART_EPISODES,
    RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
)
from game import BlackjackGame


def build_pg_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes", type=int, default=PG_TRAINING_EPISODES)
    add_seed_arguments(parser)
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Skip SpreadRuleAgent behavior cloning before RL",
    )
    parser.add_argument(
        "--reshuffle-threshold",
        type=int,
        default=None,
        help=(
            "Cut card: reshuffle when remaining cards ≤ this value "
            f"(default {RESHUFFLE_WHEN_CARDS_REMAINING_BELOW} from config)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, mps, cuda). Default: CUDA if available else CPU.",
    )
    return parser


def run_pg_training_loop(
    agent: Any,
    game: BlackjackGame,
    num_episodes: int,
    *,
    print_interval: int = PG_PRINT_INTERVAL,
    checkpoint_eval_episodes: int = PG_CHECKPOINT_EVAL_EPISODES,
    warmstart: bool | None = None,
    warmstart_episodes: int | None = None,
    learning_curve_path: Path | str | None = None,
) -> Any:
    """Train a bet+play PG agent with periodic greedy probes."""
    do_warmstart = PG_WARMSTART_ENABLED if warmstart is None else warmstart
    ws_episodes = (
        PG_WARMSTART_EPISODES if warmstart_episodes is None else warmstart_episodes
    )
    if do_warmstart and ws_episodes > 0:
        warmstart_from_spread_rule(agent, game, ws_episodes)

    curve_logger = None
    if learning_curve_path is not None and PG_LEARNING_CURVE_ENABLED:
        curve_logger = LearningCurveLogger(learning_curve_path)

    running = 0.0
    window = 0
    for episode in range(1, num_episodes + 1):
        reward = agent.train_one_episode(game)
        running += reward
        window += 1

        if episode % print_interval == 0 or episode == num_episodes:
            mean_train = running / max(1, window)
            running = 0.0
            window = 0
            eval_mean, _ = evaluate_greedy(
                agent,
                checkpoint_eval_episodes,
                reshuffle_threshold=game.reshuffle_threshold,
            )
            print(
                f"Episode {episode}/{num_episodes}  "
                f"train_window={mean_train:.4f}  "
                f"greedy_eval={eval_mean:.4f}  "
                f"steps={agent.training_steps}"
            )
            if curve_logger is not None:
                curve_logger.append(
                    episode=episode,
                    training_steps=agent.training_steps,
                    eval_reward=eval_mean,
                    epsilon=0.0,
                    shoe_features_on=True,
                )

    # Flush partial PPO rollout if any.
    if hasattr(agent, "update_from_rollout") and getattr(
        agent, "_episodes_in_rollout", 0
    ):
        agent.update_from_rollout()
        agent.clear_rollout()

    return agent


def run_pg_train_main(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    description: str,
) -> None:
    parser = build_pg_arg_parser(description)
    args = parser.parse_args()
    try:
        seeds = seeds_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    multi = len(seeds) > 1
    aggregate: list[dict[str, Any]] = []
    agent_base = agent_results_path(agent_name, model_filename).parent

    for seed in seeds:
        set_seed(seed)
        out_dir = seed_artifact_dir(agent_base, seed, multi=multi)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / model_filename
        curve_path = out_dir / PG_LEARNING_CURVE_FILENAME

        factory_kwargs: dict[str, Any] = {}
        if args.device is not None:
            factory_kwargs["device"] = args.device

        game = BlackjackGame(reshuffle_threshold=args.reshuffle_threshold)
        agent = agent_factory(**factory_kwargs)
        print(f"\n=== Train seed={seed} artifacts={out_dir} ===")
        run_pg_training_loop(
            agent,
            game,
            args.episodes,
            warmstart=not args.no_warmstart,
            learning_curve_path=curve_path,
        )
        save_policy_checkpoint(agent, model_path)
        print(f"Saved checkpoint: {model_path}")

        mean_reward, distribution = evaluate_greedy(
            agent,
            PG_FINAL_EVAL_EPISODES,
            seed=seed,
            reshuffle_threshold=args.reshuffle_threshold,
        )
        print(f"Final greedy eval ({PG_FINAL_EVAL_EPISODES} eps): {mean_reward:.6f}")
        print_distribution(distribution)

        summary = {
            "agent": agent_name,
            "seed": seed,
            "episodes": args.episodes,
            "final_eval_episodes": PG_FINAL_EVAL_EPISODES,
            "mean_reward": mean_reward,
            "distribution": dict(distribution),
            "checkpoint": str(model_path),
            "warmstart": not args.no_warmstart,
            "reshuffle_threshold": game.reshuffle_threshold,
        }
        summary_path = out_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        aggregate.append(summary)

    if multi:
        agg_path = agent_base / "multi_seed_train_summary.json"
        agg_path.write_text(json.dumps({"runs": aggregate}, indent=2) + "\n")
        print(f"Wrote multi-seed summary: {agg_path}")


def train_pg_agent(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    num_episodes: int = PG_TRAINING_EPISODES,
    warmstart: bool | None = None,
    learning_curve_path: Path | str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    reshuffle_threshold: int | None = None,
) -> Any:
    """Programmatic trainer used by tests and notebooks."""
    game = BlackjackGame(reshuffle_threshold=reshuffle_threshold)
    agent = agent_factory(**(agent_kwargs or {}))
    curve = learning_curve_path
    if curve is None:
        curve = agent_results_path(agent_name, PG_LEARNING_CURVE_FILENAME)
    run_pg_training_loop(
        agent,
        game,
        num_episodes,
        warmstart=warmstart,
        learning_curve_path=curve,
    )
    save_policy_checkpoint(
        agent, agent_results_path(agent_name, model_filename)
    )
    return agent
