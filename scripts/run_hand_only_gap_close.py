#!/usr/bin/env python3
"""Close the gap to the rule baseline with a true hand-only Double DQN run.

Protocol:
  - 8-D hand encoder (no shoe dims)
  - 100k rule-agent warm-start (behavior cloning)
  - 500k RL episodes, shoe features never enabled
  - Greedy eval vs rule baseline on the same seed/episode count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.common import (
    agent_results_path,
    evaluate_greedy,
    neural_training_kwargs,
    run_neural_training_loop,
    save_torch_checkpoint,
    set_seed,
)
from agents.double_dqn import DoubleQNetworkLearningAgent
from agents.rule import RuleAgent
from config import (
    GAP_CLOSE_CHECKPOINT_EVAL_EPISODES,
    GAP_CLOSE_EPSILON_MIN,
    GAP_CLOSE_EVAL_EPISODES,
    GAP_CLOSE_PRINT_INTERVAL,
    GAP_CLOSE_TRAINING_EPISODES,
    GAP_CLOSE_WARMSTART_EPISODES,
    NEURAL_LEARNING_CURVE_FILENAME,
)
from game import BlackjackGame

RESULTS_DIR = agent_results_path("double_dqn", "gap_close")


def run_gap_close(
    *,
    seed: int = 42,
    warmstart_episodes: int = GAP_CLOSE_WARMSTART_EPISODES,
    train_episodes: int = GAP_CLOSE_TRAINING_EPISODES,
    eval_episodes: int = GAP_CLOSE_EVAL_EPISODES,
    smoke: bool = False,
) -> dict:
    if smoke:
        warmstart_episodes = min(50, warmstart_episodes)
        train_episodes = min(200, train_episodes)
        eval_episodes = min(200, eval_episodes)
        print_interval = 50
        checkpoint_eval = 20
    else:
        print_interval = GAP_CLOSE_PRINT_INTERVAL
        checkpoint_eval = GAP_CLOSE_CHECKPOINT_EVAL_EPISODES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / "hand_only_gap_close_model.pt"
    curve_path = RESULTS_DIR / NEURAL_LEARNING_CURVE_FILENAME
    summary_path = RESULTS_DIR / "gap_close_results.json"

    set_seed(seed)
    kwargs = neural_training_kwargs()
    kwargs["epsilon_min"] = GAP_CLOSE_EPSILON_MIN
    agent = DoubleQNetworkLearningAgent(hand_only_encoder=True, **kwargs)
    game = BlackjackGame()

    print(
        f"Gap-close protocol: 8-D hand encoder, warmstart={warmstart_episodes}, "
        f"train={train_episodes}, eval={eval_episodes}, seed={seed}"
    )

    run_neural_training_loop(
        agent,
        game,
        train_episodes,
        curriculum=False,
        warmstart=True,
        warmstart_episodes=warmstart_episodes,
        force_shoe_off=True,
        learning_curve_path=curve_path,
        print_interval=print_interval,
        checkpoint_eval_episodes=checkpoint_eval,
    )
    save_torch_checkpoint(agent, model_path)

    agent_reward, agent_dist = evaluate_greedy(agent, eval_episodes)

    set_seed(seed)
    rule_agent = RuleAgent()
    rule_reward, rule_dist = evaluate_greedy(rule_agent, eval_episodes)

    summary = {
        "seed": seed,
        "smoke": smoke,
        "hand_only_encoder": True,
        "state_size": agent.input_size,
        "warmstart_episodes": warmstart_episodes,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "epsilon_min": GAP_CLOSE_EPSILON_MIN,
        "agent": {
            "average_reward": round(agent_reward, 6),
            "training_steps": agent.training_steps,
            "distribution": dict(agent_dist),
        },
        "rule_baseline": {
            "average_reward": round(rule_reward, 6),
            "distribution": dict(rule_dist),
        },
        "gap": round(agent_reward - rule_reward, 6),
        "model_path": str(model_path),
        "learning_curve_path": str(curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nHand-only agent: {agent_reward:.4f}  "
        f"Rule baseline: {rule_reward:.4f}  "
        f"Gap (agent - rule): {summary['gap']:.4f}"
    )
    print(f"Wrote {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmstart-episodes",
        type=int,
        default=GAP_CLOSE_WARMSTART_EPISODES,
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=GAP_CLOSE_TRAINING_EPISODES,
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=GAP_CLOSE_EVAL_EPISODES,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny episode counts for CI",
    )
    args = parser.parse_args(argv)
    run_gap_close(
        seed=args.seed,
        warmstart_episodes=args.warmstart_episodes,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
