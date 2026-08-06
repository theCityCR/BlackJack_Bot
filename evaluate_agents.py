"""Evaluate the rule agent and locally available trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from agents.common import agent_results_path, evaluate_greedy
from agents.dqn import DeepQLearningAgent
from agents.double_dqn import DoubleQNetworkLearningAgent
from agents.dueling import DuelingDQNAgent
from agents.prioritized import PrioritizedDuelingDQNAgent
from agents.tabular_q import QLearningAgent
from agents.rule import RuleAgent


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "eval"
DOCS_RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
PUBLISHED_NEURAL_AGENTS = {
    "Double DQN",
    "Dueling Double DQN",
    "Dueling Double DQN + PER",
}

TORCH_CHECKPOINTS: dict[str, tuple[type, Path]] = {
    "DQN": (
        DeepQLearningAgent,
        agent_results_path("dqn", "deep_q_learning_model.pt"),
    ),
    "Double DQN": (
        DoubleQNetworkLearningAgent,
        agent_results_path("double_dqn", "double_q_network_model.pt"),
    ),
    "Dueling Double DQN": (
        DuelingDQNAgent,
        agent_results_path("dueling", "dueling_dqn_model.pt"),
    ),
    "Dueling Double DQN + PER": (
        PrioritizedDuelingDQNAgent,
        agent_results_path("prioritized", "dueling_dqn_prioritized_model.pt"),
    ),
}

Q_TABLE_PATH = agent_results_path("tabular_q", "q_table.json")


def load_torch_agent(agent_class, checkpoint_path: Path):
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


def load_q_learning_agent(checkpoint_path: Path):
    agent = QLearningAgent.load(str(checkpoint_path))
    agent.epsilon = 0.0
    return agent, getattr(agent, "training_steps", None)


def evaluate_agent(name: str, agent, episodes: int, seed: int) -> dict[str, Any]:
    """Run one agent on paired per-episode shoes for ``seed``."""
    average_reward, distribution = evaluate_greedy(agent, episodes, seed=seed)

    wins = sum(
        distribution.get(category, 0)
        for category in (
            "normal_win",
            "blackjack_win",
            "big_win_double_or_split",
        )
    )
    losses = sum(
        distribution.get(category, 0)
        for category in (
            "normal_loss",
            "big_loss_double_or_split",
        )
    )
    draws = distribution.get("draw", 0)

    return {
        "agent": name,
        "episodes": episodes,
        "average_reward": round(average_reward, 6),
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

    with (output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent",
                "episodes",
                "average_reward",
                "win_rate",
                "loss_rate",
                "draw_rate",
                "training_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    write_svg(results, output_dir / "benchmark_results.svg")


def write_svg(results: list[dict[str, Any]], path: Path) -> None:
    width = 760
    height = 90 + 76 * len(results)
    plot_left = 260
    plot_right = 700
    zero_x = plot_left + (plot_right - plot_left) * 0.55
    scale = (plot_right - plot_left) * 0.45 / 0.15

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
<text x="30" y="58" fill="#4a5568">Higher is better; paired per-episode shoes, same seed and rules</text>
<line x1="{zero_x}" y1="68" x2="{zero_x}" y2="{height - 25}" stroke="#718096" stroke-width="2"/>
{''.join(rows)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def collect_agents() -> list[tuple[str, Any, Any]]:
    agents: list[tuple[str, Any, Any]] = [("Rule-based baseline", RuleAgent(), None)]

    if Q_TABLE_PATH.exists():
        agent, training_steps = load_q_learning_agent(Q_TABLE_PATH)
        agents.append(("Q-learning", agent, training_steps))
    else:
        print(f"Skipping Q-learning: checkpoint not found at {Q_TABLE_PATH}")

    for name, (agent_class, checkpoint_path) in TORCH_CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue
        agent, training_steps = load_torch_agent(agent_class, checkpoint_path)
        agents.append((name, agent, training_steps))

    return agents


def guard_docs_publish(output_dir: Path, evaluated_names: set[str]) -> None:
    """Refuse incomplete overwrites of the published portfolio results."""
    if output_dir.resolve() != DOCS_RESULTS_DIR.resolve():
        return

    missing = PUBLISHED_NEURAL_AGENTS - evaluated_names
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(
            "Refusing to overwrite docs/results without the published neural "
            f"checkpoints ({missing_list}). Train those agents first, or write "
            "to the default results/eval directory."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write JSON/CSV/SVG (default: results/eval).",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    agents = collect_agents()
    evaluated_names = {name for name, _, _ in agents}
    guard_docs_publish(args.output_dir, evaluated_names)

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
