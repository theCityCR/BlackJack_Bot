"""Train the Dueling Double DQN Blackjack agent."""

from __future__ import annotations

from agents.dueling import DuelingDQNAgent
from agents.train_cli import run_neural_train_main, train_neural_agent
from config import NEURAL_TRAINING_EPISODES

AGENT_NAME = "dueling"
MODEL_FILENAME = "dueling_dqn_model.pt"


def train(num_episodes: int = NEURAL_TRAINING_EPISODES, **kwargs):
    return train_neural_agent(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DuelingDQNAgent,
        num_episodes=num_episodes,
        **kwargs,
    )


def main() -> None:
    run_neural_train_main(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DuelingDQNAgent,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
