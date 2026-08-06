"""Tabular Q-learning hyperparameters (legacy / simple agent)."""

NUM_TRAINING_EPISODES = 50_000

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99999

# Relative name only; trainers resolve under agents/results/tabular_q/.
Q_TABLE_PATH = "q_table.json"
