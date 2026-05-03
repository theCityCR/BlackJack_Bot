from agents.double_q_network_learning.double_q_network_learning_agent import DeepQLearningAgent
from game import BlackjackGame


NUM_DQN_TRAINING_EPISODES = 200_000
FINAL_EVALUATION_EPISODES = 100_000
CHECKPOINT_EVALUATION_EPISODES = 20_000
PRINT_INTERVAL = 5_000


def evaluate(agent: DeepQLearningAgent, num_episodes: int) -> float:
    game = BlackjackGame()
    total_reward = 0.0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        total_reward += agent.play_episode(game)

    agent.epsilon = old_epsilon

    return total_reward / num_episodes


def train() -> DeepQLearningAgent:
    game = BlackjackGame()

    agent = DeepQLearningAgent(
        learning_rate=0.0005,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        replay_size=100_000,
        batch_size=128,
        target_update_interval=5000,
        min_replay_size=1000,
        train_updates_per_episode=2,
    )

    total_reward = 0.0

    for episode in range(1, NUM_DQN_TRAINING_EPISODES + 1):
        reward = agent.train_one_episode(game)
        total_reward += reward

        if episode % PRINT_INTERVAL == 0:
            eval_reward = evaluate(agent, CHECKPOINT_EVALUATION_EPISODES)

            print(f"Episode {episode}")
            print(f"Average training reward: {total_reward / episode:.4f}")
            print(f"Evaluation reward: {eval_reward:.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Replay buffer size: {len(agent.replay_buffer)}")
            print(f"Training steps: {agent.training_steps}")
            print()

    return agent


def main():
    agent = train()

    final_eval_reward = evaluate(agent, FINAL_EVALUATION_EPISODES)

    print(f"Final evaluation episodes: {FINAL_EVALUATION_EPISODES}")
    print(f"Final average reward: {final_eval_reward:.4f}")


if __name__ == "__main__":
    main()