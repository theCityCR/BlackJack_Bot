"""Train the REINFORCE bet+play Blackjack agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.common import agent_results_path
from agents.reinforce import ReinforceAgent
from agents.train_pg_cli import run_pg_train_main, train_pg_agent
from config import PG_LEARNING_CURVE_FILENAME, PG_TRAINING_EPISODES

AGENT_NAME = "reinforce"
MODEL_FILENAME = "reinforce_bet_play_model.pt"

MODEL_PATH = agent_results_path(AGENT_NAME, MODEL_FILENAME)
CURVE_PATH = agent_results_path(AGENT_NAME, PG_LEARNING_CURVE_FILENAME)


def train(
    num_episodes: int = PG_TRAINING_EPISODES,
    *,
    warmstart: bool | None = None,
    learning_curve_path: Path | str | None = None,
    device: str | None = None,
) -> ReinforceAgent:
    agent_kwargs: dict[str, Any] = {}
    if device is not None:
        agent_kwargs["device"] = device
    return train_pg_agent(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=ReinforceAgent,
        num_episodes=num_episodes,
        warmstart=warmstart,
        learning_curve_path=learning_curve_path,
        agent_kwargs=agent_kwargs or None,
    )


def main() -> None:
    run_pg_train_main(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=ReinforceAgent,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
