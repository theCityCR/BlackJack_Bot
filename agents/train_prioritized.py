"""Train the prioritized Dueling Double DQN Blackjack agent."""

from __future__ import annotations

from agents.prioritized import PrioritizedDuelingDQNAgent
from agents.train_cli import run_neural_train_main, train_neural_agent
from config import NEURAL_TRAINING_EPISODES

AGENT_NAME = "prioritized"
MODEL_FILENAME = "dueling_dqn_prioritized_model.pt"


def train(num_episodes: int = NEURAL_TRAINING_EPISODES, **kwargs):
    return train_neural_agent(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=PrioritizedDuelingDQNAgent,
        num_episodes=num_episodes,
        **kwargs,
    )


def main() -> None:
    run_neural_train_main(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=PrioritizedDuelingDQNAgent,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
