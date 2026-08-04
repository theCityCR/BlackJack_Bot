"""Train the Dueling Double DQN agent with prioritized experience replay."""

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
from agents.prioritized_replay.dueling_dqn_prioritized_agent import (
    PrioritizedDuelingDQNAgent,
)
from config import (
    NEURAL_FINAL_EVAL_EPISODES,
    NEURAL_TRAINING_EPISODES,
)
from game import BlackjackGame


MODEL_PATH = package_results_path(__file__, "dueling_dqn_prioritized_model.pt")


def train(
    num_episodes: int = NEURAL_TRAINING_EPISODES,
    *,
    curriculum: bool | None = None,
) -> PrioritizedDuelingDQNAgent:
    game = BlackjackGame()
    agent = PrioritizedDuelingDQNAgent(**neural_training_kwargs())
    run_neural_training_loop(
        agent,
        game,
        num_episodes,
        curriculum=curriculum,
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
    args = parser.parse_args()

    set_seed(args.seed)
    agent = train(
        args.episodes,
        curriculum=False if args.no_curriculum else None,
    )

    final_reward, final_distribution = evaluate_greedy(
        agent,
        NEURAL_FINAL_EVAL_EPISODES,
    )
    print(f"Saved model to: {MODEL_PATH}")
    print()
    print(f"Final evaluation episodes: {NEURAL_FINAL_EVAL_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final evaluation distribution:")
    print_distribution(final_distribution)
    print(f"Training steps:           {agent.training_steps}")


if __name__ == "__main__":
    main()
