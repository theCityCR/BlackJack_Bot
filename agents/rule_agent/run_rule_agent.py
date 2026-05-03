from collections import defaultdict

from agents.rule_agent.rule_agent import RuleAgent
from game import BlackjackGame


NUM_EPISODES = 500_000


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


def run_simulation(num_episodes: int):
    game = BlackjackGame()
    agent = RuleAgent()

    total_reward = 0.0
    distribution = defaultdict(int)

    for _ in range(num_episodes):
        reward = agent.play_episode(game)
        total_reward += reward

        category = categorize_reward(reward)
        distribution[category] += 1

    avg_reward = total_reward / num_episodes

    return avg_reward, distribution


def main():
    avg_reward, distribution = run_simulation(NUM_EPISODES)

    print(f"Ran {NUM_EPISODES} episodes")
    print(f"Average reward per game: {avg_reward:.4f}")
    print("\nDistribution:")

    total = sum(distribution.values())

    for category in sorted(distribution):
        count = distribution[category]
        percentage = count / total
        print(f"{category:30s}: {count:8d} ({percentage:.2%})")


if __name__ == "__main__":
    main()