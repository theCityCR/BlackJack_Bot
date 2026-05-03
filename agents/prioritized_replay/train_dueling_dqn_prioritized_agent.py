import os
from collections import defaultdict

import torch

from agents.prioritized_replay.dueling_dqn_prioritized_agent import DuelingDQNAgent
from game import BlackjackGame


NUM_TRAINING_EPISODES = 100_000
FINAL_EVALUATION_EPISODES = 100_000
PRINT_INTERVAL = 5_000

MODEL_PATH = "results/dueling_dqn_prioritized_model.pt"


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


def evaluate(agent: DuelingDQNAgent, num_episodes: int):
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


def save_agent(agent: DuelingDQNAgent, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "target_model_state_dict": agent.target_model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "training_steps": agent.training_steps,
            "replay_beta": agent.replay_buffer.beta,
        },
        path,
    )


def train():
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
        train_updates_per_episode=1,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        priority_beta_increment=0.00001,
    )

    interval_reward = 0.0
    interval_distribution = defaultdict(int)

    for episode in range(1, NUM_TRAINING_EPISODES + 1):
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
            print(f"Replay beta:              {agent.replay_buffer.beta:.4f}")
            print(f"Replay buffer size:       {len(agent.replay_buffer)}")
            print(f"Training steps:           {agent.training_steps}")
            print("Training distribution:")
            print_distribution(interval_distribution)
            print()

            interval_reward = 0.0
            interval_distribution = defaultdict(int)

    return agent


def main():
    agent = train()

    save_agent(agent, MODEL_PATH)

    final_reward, final_distribution = evaluate(
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