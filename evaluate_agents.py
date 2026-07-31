"""Evaluate the rule agent and locally available trained checkpoints."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch

from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from agents.dueling_dqn.dueling_dqn_agent import DuelingDQNAgent
from agents.prioritized_replay.dueling_dqn_prioritized_agent import (
    DuelingDQNAgent as PrioritizedDuelingDQNAgent,
)
from agents.rule_agent.rule_agent import RuleAgent
from game import BlackjackGame


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "Double DQN": (
        DoubleQNetworkLearningAgent,
        PROJECT_ROOT / "agents/double_q_network_learning/results/double_q_network_model.pt",
    ),
    "Dueling DQN": (
        DuelingDQNAgent,
        PROJECT_ROOT / "agents/dueling_dqn/results/dueling_dqn_model.pt",
    ),
    "Dueling Double DQN + PER": (
        PrioritizedDuelingDQNAgent,
        PROJECT_ROOT / "agents/prioritized_replay/results/dueling_dqn_prioritized_model.pt",
    ),
}


def load_agent(agent_class, checkpoint_path: Path):
    """Construct an inference-only agent from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        agent = agent_class(device="cpu")
    except TypeError:
        agent = agent_class()

    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.model.eval()
    agent.epsilon = 0.0
    return agent, checkpoint.get("training_steps")


def evaluate_agent(name: str, agent, episodes: int, seed: int) -> dict[str, Any]:
    """Run one agent with a reproducible shoe sequence."""
    random.seed(seed)
    torch.manual_seed(seed)
    game = BlackjackGame()
    rewards = [agent.play_episode(game) for _ in range(episodes)]

    wins = sum(reward > 0 for reward in rewards)
    losses = sum(reward < 0 for reward in rewards)
    draws = episodes - wins - losses

    return {
        "agent": name,
        "episodes": episodes,
        "average_reward": round(sum(rewards) / episodes, 6),
        "win_rate": round(wins / episodes, 6),
        "loss_rate": round(losses / episodes, 6),
        "draw_rate": round(draws / episodes, 6),
    }


def write_results(results: list[dict[str, Any]], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"seed": seed, "results": results}
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)

    write_svg(results, output_dir / "benchmark_results.svg")


def write_svg(results: list[dict[str, Any]], path: Path) -> None:
    """Create a dependency-free average-reward comparison chart."""
    width, height = 900, 110 + 76 * len(results)
    plot_left, zero_x, scale = 250, 500, 1200
    rows = []

    for index, result in enumerate(results):
        y = 85 + index * 76
        reward = result["average_reward"]
        bar_width = abs(reward) * scale
        bar_x = zero_x if reward >= 0 else zero_x - bar_width
        color = "#2f855a" if reward >= 0 else "#c53030"
        rows.append(
            f'<text x="{plot_left - 15}" y="{y + 6}" text-anchor="end">{result["agent"]}</text>'
            f'<rect x="{bar_x:.1f}" y="{y - 17}" width="{bar_width:.1f}" height="28" rx="4" fill="{color}"/>'
            f'<text x="{zero_x + (10 if reward >= 0 else -10)}" y="{y + 5}" '
            f'text-anchor="{"start" if reward >= 0 else "end"}" font-weight="600">{reward:+.4f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 15px; fill: #1a202c; }}</style>
<text x="30" y="35" font-size="22" font-weight="700">Average reward per round</text>
<text x="30" y="58" fill="#4a5568">Higher is better; identical seed, rules, and episode count</text>
<line x1="{zero_x}" y1="68" x2="{zero_x}" y2="{height - 25}" stroke="#718096" stroke-width="2"/>
{''.join(rows)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs/results")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    agents = [("Rule-based baseline", RuleAgent(), None)]
    for name, (agent_class, checkpoint_path) in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue
        agent, training_steps = load_agent(agent_class, checkpoint_path)
        agents.append((name, agent, training_steps))

    results = []
    for name, agent, training_steps in agents:
        result = evaluate_agent(name, agent, args.episodes, args.seed)
        result["training_steps"] = training_steps
        results.append(result)
        print(f"{name:28s} average reward: {result['average_reward']:+.4f}")

    write_results(results, args.output_dir, args.seed)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
