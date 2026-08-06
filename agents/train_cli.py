"""Shared CLI helpers for neural agent trainers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from agents.common import (
    agent_results_path,
    evaluate_greedy,
    neural_training_kwargs,
    print_distribution,
    run_neural_training_loop,
    save_torch_checkpoint,
    set_seed,
)
from config import (
    NEURAL_FINAL_EVAL_EPISODES,
    NEURAL_LEARNING_CURVE_FILENAME,
    NEURAL_TRAINING_EPISODES,
)
from game import BlackjackGame


def build_neural_arg_parser(
    description: str,
    *,
    include_device: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes", type=int, default=NEURAL_TRAINING_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Use full shoe features from episode 1 (skip hand-only phase A)",
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Skip rule-agent behavior cloning before RL",
    )
    if include_device:
        parser.add_argument(
            "--device",
            default=None,
            help="Torch device (cpu, mps, cuda). Default: CUDA if available else "
            "CPU. MPS is opt-in (often slower for this workload).",
        )
    return parser


def run_neural_train_main(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    description: str,
    include_device: bool = False,
    force_shoe_off: bool = False,
) -> None:
    """Parse args, train, evaluate, and save a neural agent checkpoint."""
    parser = build_neural_arg_parser(description, include_device=include_device)
    args = parser.parse_args()

    set_seed(args.seed)
    model_path = agent_results_path(agent_name, model_filename)
    curve_path = agent_results_path(agent_name, NEURAL_LEARNING_CURVE_FILENAME)

    factory_kwargs = dict(neural_training_kwargs())
    if include_device and getattr(args, "device", None) is not None:
        factory_kwargs["device"] = args.device

    game = BlackjackGame()
    agent = agent_factory(**factory_kwargs)
    run_neural_training_loop(
        agent,
        game,
        args.episodes,
        curriculum=False if args.no_curriculum else None,
        warmstart=False if args.no_warmstart else None,
        force_shoe_off=force_shoe_off,
        learning_curve_path=curve_path,
    )
    save_torch_checkpoint(agent, model_path)

    final_reward, final_distribution = evaluate_greedy(
        agent,
        NEURAL_FINAL_EVAL_EPISODES,
    )
    print(f"Final evaluation episodes: {NEURAL_FINAL_EVAL_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final distribution:")
    print_distribution(final_distribution)
    print(f"Training steps:           {agent.training_steps}")
    print(f"\nSaved model to: {model_path}")


def train_neural_agent(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    num_episodes: int = NEURAL_TRAINING_EPISODES,
    curriculum: bool | None = None,
    warmstart: bool | None = None,
    force_shoe_off: bool = False,
    learning_curve_path: Path | str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Programmatic train helper used by ablation / gap-close scripts."""
    model_path = agent_results_path(agent_name, model_filename)
    curve_path = (
        agent_results_path(agent_name, NEURAL_LEARNING_CURVE_FILENAME)
        if learning_curve_path is None
        else learning_curve_path
    )

    kwargs = dict(neural_training_kwargs())
    if agent_kwargs:
        kwargs.update(agent_kwargs)

    game = BlackjackGame()
    agent = agent_factory(**kwargs)
    run_neural_training_loop(
        agent,
        game,
        num_episodes,
        curriculum=curriculum,
        warmstart=warmstart,
        force_shoe_off=force_shoe_off,
        learning_curve_path=curve_path,
    )
    save_torch_checkpoint(agent, model_path)
    return agent
