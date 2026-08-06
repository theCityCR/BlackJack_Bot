"""Train the Double DQN Blackjack agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.common import agent_results_path
from agents.double_dqn import DoubleQNetworkLearningAgent
from agents.train_cli import run_neural_train_main, train_neural_agent
from config import NEURAL_LEARNING_CURVE_FILENAME, NEURAL_TRAINING_EPISODES

AGENT_NAME = "double_dqn"
MODEL_FILENAME = "double_q_network_model.pt"

MODEL_PATH = agent_results_path(AGENT_NAME, MODEL_FILENAME)
CURVE_PATH = agent_results_path(AGENT_NAME, NEURAL_LEARNING_CURVE_FILENAME)


def train(
    num_episodes: int = NEURAL_TRAINING_EPISODES,
    *,
    curriculum: bool | None = None,
    warmstart: bool | None = None,
    force_shoe_off: bool = False,
    learning_curve_path: Path | str | None = None,
    device: str | None = None,
) -> DoubleQNetworkLearningAgent:
    agent_kwargs: dict[str, Any] = {}
    if device is not None:
        agent_kwargs["device"] = device
    return train_neural_agent(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DoubleQNetworkLearningAgent,
        num_episodes=num_episodes,
        curriculum=curriculum,
        warmstart=warmstart,
        force_shoe_off=force_shoe_off,
        learning_curve_path=learning_curve_path,
        agent_kwargs=agent_kwargs or None,
    )


def main() -> None:
    run_neural_train_main(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DoubleQNetworkLearningAgent,
        description=__doc__,
        include_device=True,
    )


if __name__ == "__main__":
    main()
