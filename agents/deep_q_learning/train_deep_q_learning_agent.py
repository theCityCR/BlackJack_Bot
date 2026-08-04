"""Train the vanilla DQN Blackjack agent."""

from __future__ import annotations

import argparse

from agents.common import (
    evaluate_greedy,
    package_results_path,
    print_distribution,
    save_torch_checkpoint,
    set_seed,
)
from agents.deep_q_learning.deep_q_learning_agent import DeepQLearningAgent
from game import BlackjackGame


NUM_DQN_TRAINING_EPISODES = 200_000
DQN_EVALUATION_EPISODES = 100_000
CHECKPOINT_EVALUATION_EPISODES = 5_000
PRINT_INTERVAL = 5_000
MODEL_PATH = package_results_path(__file__, "deep_q_learning_model.pt")


def train(num_episodes: int = NUM_DQN_TRAINING_EPISODES) -> DeepQLearningAgent:
    game = BlackjackGame()
    agent = DeepQLearningAgent(
        learning_rate=0.001,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        replay_size=100_000,
        batch_size=128,
        target_update_interval=1000,
        min_replay_size=1000,
        train_updates_per_episode=1,
    )

    total_reward = 0.0

    for episode in range(1, num_episodes + 1):
        reward = agent.train_one_episode(game)
        total_reward += reward

        if episode % PRINT_INTERVAL == 0:
            eval_reward, _ = evaluate_greedy(agent, CHECKPOINT_EVALUATION_EPISODES)
            print(f"Episode {episode}")
            print(f"Average training reward: {total_reward / episode:.4f}")
            print(f"Evaluation reward: {eval_reward:.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Replay buffer size: {len(agent.replay_buffer)}")
            print(f"Training steps: {agent.training_steps}")
            print()

    save_torch_checkpoint(agent, MODEL_PATH)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=NUM_DQN_TRAINING_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    agent = train(args.episodes)

    final_eval_reward, final_distribution = evaluate_greedy(
        agent,
        DQN_EVALUATION_EPISODES,
    )
    print(f"Final evaluation episodes: {DQN_EVALUATION_EPISODES}")
    print(f"Final average reward: {final_eval_reward:.4f}")
    print("Final distribution:")
    print_distribution(final_distribution)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
