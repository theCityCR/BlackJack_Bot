import os
from collections import defaultdict

import torch

from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from game import BlackjackGame


NUM_TRAINING_EPISODES = 200_000
CHECKPOINT_EVALUATION_EPISODES = 10_000
FINAL_EVALUATION_EPISODES = 100_000
PRINT_INTERVAL = 5_000

MODEL_PATH = "results/double_q_network_model.pt"


def categorize_reward(reward: float) -> str:
    if reward == 0:
        return "draw"
    if reward == 1:
        return "normal_win"
    if reward == -1:
        return "normal_loss"
    if reward == 1.5:
        return "blackjack_win"
    if reward > 1:
        return "big_win_double_or_split"
    if reward < -1:
        return "big_loss_double_or_split"

    return "other"


def evaluate(agent: DoubleQNetworkLearningAgent, num_episodes: int):
    game = BlackjackGame()

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_reward = 0.0
    distribution = defaultdict(int)

    for _ in range(num_episodes):
        reward = agent.play_episode(game)
        total_reward += reward
        distribution[categorize_reward(reward)] += 1

    agent.epsilon = old_epsilon

    return total_reward / num_episodes, distribution


def print_distribution(distribution):
    total = sum(distribution.values())

    for category in sorted(distribution):
        count = distribution[category]
        percentage = count / total
        print(f"{category:30s}: {count:8d} ({percentage:.2%})")


def save_agent(agent: DoubleQNetworkLearningAgent, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "target_model_state_dict": agent.target_model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "training_steps": agent.training_steps,
        },
        path,
    )


def train():
    game = BlackjackGame()

    agent = DoubleQNetworkLearningAgent(
        learning_rate=0.0005,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        replay_size=100_000,
        batch_size=128,
        target_update_interval=5_000,
        min_replay_size=1_000,
        train_updates_per_episode=2,
    )

    total_training_reward = 0.0

    for episode in range(1, NUM_TRAINING_EPISODES + 1):
        reward = agent.train_one_episode(game)
        total_training_reward += reward

        if episode % PRINT_INTERVAL == 0:
            eval_reward, eval_distribution = evaluate(
                agent,
                CHECKPOINT_EVALUATION_EPISODES,
            )

            print(f"Episode {episode}")
            print(f"Average training reward: {total_training_reward / episode:.4f}")
            print(f"Evaluation reward:        {eval_reward:.4f}")
            print(f"Epsilon:                  {agent.epsilon:.4f}")
            print(f"Replay buffer size:       {len(agent.replay_buffer)}")
            print(f"Training steps:           {agent.training_steps}")
            print("Evaluation distribution:")
            print_distribution(eval_distribution)
            print()

    save_agent(agent, MODEL_PATH)
    return agent


def main():
    agent = train()

    final_reward, final_distribution = evaluate(
        agent,
        FINAL_EVALUATION_EPISODES,
    )

    print(f"Final evaluation episodes: {FINAL_EVALUATION_EPISODES}")
    print(f"Final average reward:      {final_reward:.4f}")
    print("Final distribution:")
    print_distribution(final_distribution)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()