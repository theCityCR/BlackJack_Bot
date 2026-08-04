"""Train the Double DQN Blackjack agent."""

from __future__ import annotations

import argparse

from agents.common import (
    evaluate_greedy,
    neural_training_kwargs,
    package_results_path,
    print_distribution,
    run_neural_training_loop,
    save_torch_checkpoint,
    set_seed,
)
from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from config import (
    NEURAL_FINAL_EVAL_EPISODES,
    NEURAL_LEARNING_CURVE_FILENAME,
    NEURAL_TRAINING_EPISODES,
)
from game import BlackjackGame


MODEL_PATH = package_results_path(__file__, "double_q_network_model.pt")
CURVE_PATH = package_results_path(__file__, NEURAL_LEARNING_CURVE_FILENAME)


def train(
    num_episodes: int = NEURAL_TRAINING_EPISODES,
    *,
    curriculum: bool | None = None,
    warmstart: bool | None = None,
    force_shoe_off: bool = False,
    learning_curve_path=None,
) -> DoubleQNetworkLearningAgent:
    game = BlackjackGame()
    agent = DoubleQNetworkLearningAgent(**neural_training_kwargs())
    run_neural_training_loop(
        agent,
        game,
        num_episodes,
        curriculum=curriculum,
        warmstart=warmstart,
        force_shoe_off=force_shoe_off,
        learning_curve_path=(
            CURVE_PATH if learning_curve_path is None else learning_curve_path
        ),
    )
    save_torch_checkpoint(agent, MODEL_PATH)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    set_seed(args.seed)
    agent = train(
        args.episodes,
        curriculum=False if args.no_curriculum else None,
        warmstart=False if args.no_warmstart else None,
    )

    final_reward, final_distribution = evaluate_greedy(
        agent,
        NEURAL_FINAL_EVAL_EPISODES,
    )
    print(f"Final evaluation episodes: {NEURAL_FINAL_EVAL_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final distribution:")
    print_distribution(final_distribution)
    print(f"Training steps:           {agent.training_steps}")
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
