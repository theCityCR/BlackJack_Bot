"""Run a small evaluation of the rule-based Blackjack agent."""

import argparse
import random

from agents.rule import RuleAgent
from game import BlackjackGame


def evaluate(episodes: int, seed: int | None = None) -> dict[str, float]:
    """Evaluate the rule agent and return aggregate reward statistics."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if seed is not None:
        random.seed(seed)

    game = BlackjackGame()
    agent = RuleAgent()
    rewards = [agent.play_episode(game) for _ in range(episodes)]

    return {
        "episodes": episodes,
        "total_reward": sum(rewards),
        "average_reward": sum(rewards) / episodes,
        "profitable_rounds": sum(reward > 0 for reward in rewards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    results = evaluate(args.episodes, args.seed)
    print(f"Episodes: {results['episodes']}")
    print(f"Total reward: {results['total_reward']:.1f}")
    print(f"Average reward: {results['average_reward']:.4f}")
    print(f"Profitable rounds: {results['profitable_rounds']}")


if __name__ == "__main__":
    main()
