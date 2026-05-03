from collections import defaultdict

from agents.rule_agent.rule_agent import RuleAgent
from game import BlackjackGame


# =========================
# Configuration
# =========================
NUM_EPISODES = 500000   # <-- change this freely


def run_simulation(num_episodes):
    game = BlackjackGame()
    agent = RuleAgent()

    total_reward = 0.0
    distribution = defaultdict(int)

    for _ in range(num_episodes):
        reward = agent.play_episode(game)
        total_reward += reward

        if reward == 1:
            distribution["win"] += 1
        elif reward == -1:
            distribution["loss"] += 1
        elif reward == 0:
            distribution["draw"] += 1
        elif reward > 1:
            distribution["big_win (blackjack/double/split)"] += 1
        elif reward < -1:
            distribution["big_loss (double/split)"] += 1

    avg_reward = total_reward / num_episodes

    return avg_reward, distribution


def main():
    avg_reward, distribution = run_simulation(NUM_EPISODES)

    print(f"Ran {NUM_EPISODES} episodes")
    print(f"Average reward per game: {avg_reward:.4f}")
    print("\nDistribution:")

    total = sum(distribution.values())
    for k, v in distribution.items():
        print(f"{k:35s}: {v} ({v/total:.2%})")


if __name__ == "__main__":
    main()