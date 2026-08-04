"""Train the Dueling Double DQN Blackjack agent."""

from __future__ import annotations

import argparse
from collections import defaultdict

from agents.common import (
    categorize_reward,
    evaluate_greedy,
    package_results_path,
    print_distribution,
    save_torch_checkpoint,
    set_seed,
)
from agents.dueling_dqn.dueling_dqn_agent import DuelingDQNAgent
from game import BlackjackGame


NUM_TRAINING_EPISODES = 100_000
FINAL_EVALUATION_EPISODES = 100_000
PRINT_INTERVAL = 5_000
MODEL_PATH = package_results_path(__file__, "dueling_dqn_model.pt")


def train(num_episodes: int = NUM_TRAINING_EPISODES) -> DuelingDQNAgent:
    game = BlackjackGame()
    agent = DuelingDQNAgent(
        learning_rate=0.001,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9999,
        replay_size=100_000,
        batch_size=256,
        target_update_interval=2_000,
        min_replay_size=5_000,
        train_updates_per_episode=2,
    )

    interval_reward = 0.0
    interval_distribution: dict[str, int] = defaultdict(int)

    for episode in range(1, num_episodes + 1):
        reward = agent.train_one_episode(game)
        interval_reward += reward
        interval_distribution[categorize_reward(reward)] += 1

        if episode % PRINT_INTERVAL == 0:
            print(f"Episode {episode}")
            print(
                f"Average reward over last {PRINT_INTERVAL} episodes: "
                f"{interval_reward / PRINT_INTERVAL:.4f}"
            )
            print(f"Epsilon:                  {agent.epsilon:.4f}")
            print(f"Replay buffer size:       {len(agent.replay_buffer)}")
            print(f"Training steps:           {agent.training_steps}")
            print("Training distribution:")
            print_distribution(interval_distribution)
            print()
            interval_reward = 0.0
            interval_distribution = defaultdict(int)

    save_torch_checkpoint(agent, MODEL_PATH)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=NUM_TRAINING_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    agent = train(args.episodes)

    final_reward, final_distribution = evaluate_greedy(
        agent,
        FINAL_EVALUATION_EPISODES,
    )
    print(f"Saved model to: {MODEL_PATH}")
    print()
    print(f"Final evaluation episodes: {FINAL_EVALUATION_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final evaluation distribution:")
    print_distribution(final_distribution)


if __name__ == "__main__":
    main()
