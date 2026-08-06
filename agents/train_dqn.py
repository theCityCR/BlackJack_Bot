"""Train the vanilla DQN Blackjack agent."""

from __future__ import annotations

from agents.dqn import DeepQLearningAgent
from agents.train_cli import run_neural_train_main, train_neural_agent
from config import NEURAL_TRAINING_EPISODES

AGENT_NAME = "dqn"
MODEL_FILENAME = "deep_q_learning_model.pt"


def train(num_episodes: int = NEURAL_TRAINING_EPISODES, **kwargs):
    return train_neural_agent(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DeepQLearningAgent,
        num_episodes=num_episodes,
        **kwargs,
    )


def main() -> None:
    run_neural_train_main(
        agent_name=AGENT_NAME,
        model_filename=MODEL_FILENAME,
        agent_factory=DeepQLearningAgent,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
