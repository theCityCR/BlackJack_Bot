"""Train the tabular Q-learning Blackjack agent."""

from __future__ import annotations

import argparse
import random

from agents.common import (
    agent_results_path,
    evaluate_greedy,
    print_distribution,
)
from agents.tabular_q import QLearningAgent
from config import NUM_TRAINING_EPISODES
from game import BlackjackGame


PRINT_INTERVAL = 5_000
FINAL_EVALUATION_EPISODES = 5_000
MODEL_PATH = agent_results_path("tabular_q", "q_table.json")


def train(num_episodes: int = NUM_TRAINING_EPISODES) -> QLearningAgent:
    game = BlackjackGame()
    agent = QLearningAgent()
    total_reward = 0.0

    for episode in range(1, num_episodes + 1):
        reward = agent.train_one_episode(game)
        total_reward += reward

        if episode % PRINT_INTERVAL == 0:
            eval_reward, eval_distribution = evaluate_greedy(
                agent,
                FINAL_EVALUATION_EPISODES,
            )
            print(f"Episode {episode}")
            print(f"Average training reward: {total_reward / episode:.4f}")
            print(f"Evaluation reward:        {eval_reward:.4f}")
            print(f"Epsilon:                  {agent.epsilon:.4f}")
            print(f"Q-table states:           {len(agent.q_table)}")
            print(f"Training steps:           {agent.training_steps}")
            print("Evaluation distribution:")
            print_distribution(eval_distribution)
            print()

    agent.save(str(MODEL_PATH))
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=NUM_TRAINING_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    agent = train(args.episodes)

    final_reward, final_distribution = evaluate_greedy(
        agent,
        FINAL_EVALUATION_EPISODES,
        seed=args.seed,
    )
    print(f"Saved Q-table to: {MODEL_PATH}")
    print()
    print(f"Final evaluation episodes: {FINAL_EVALUATION_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final evaluation distribution:")
    print_distribution(final_distribution)


if __name__ == "__main__":
    main()
