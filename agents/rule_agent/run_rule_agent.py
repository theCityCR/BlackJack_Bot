"""Thin wrapper around the seeded rule-agent CLI in main.py."""

from __future__ import annotations

import argparse

from main import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results = evaluate(args.episodes, args.seed)
    print(f"Episodes: {results['episodes']}")
    print(f"Total reward: {results['total_reward']:.1f}")
    print(f"Average reward: {results['average_reward']:.4f}")
    print(f"Profitable rounds: {results['profitable_rounds']}")


if __name__ == "__main__":
    main()
