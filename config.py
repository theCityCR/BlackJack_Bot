"""
config.py

Global configuration values for the Blackjack ML project.

Default ruleset:
- 2-deck shoe that persists between rounds
- Reshuffle when fewer than 26 cards remain
- Dealer stands on all 17s (S17), including soft 17
- Double after split allowed (DAS)
- Re-split allowed up to MAX_PLAYER_HANDS
- No hitting split aces
"""

# =========================
# Training Settings
# =========================

NUM_TRAINING_EPISODES = 50_000
EVALUATION_EPISODES = 5_000

# =========================
# Q-Learning Hyperparameters
# =========================

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99999

# =========================
# Shared Neural DQN Training
# =========================
# All neural agents use these defaults so architecture comparisons are not
# confounded by unequal episode budgets or optimizer schedules. Architecture
# (network topology, Double/Dueling targets, prioritized replay) remains the
# only intentional difference between neural trainers.

NEURAL_TRAINING_EPISODES = 200_000
NEURAL_CHECKPOINT_EVAL_EPISODES = 5_000
NEURAL_FINAL_EVAL_EPISODES = 25_000
NEURAL_PRINT_INTERVAL = 5_000

NEURAL_LEARNING_RATE = 0.001
NEURAL_DISCOUNT_FACTOR = 1.0
NEURAL_EPSILON_START = 1.0
NEURAL_EPSILON_MIN = 0.05
NEURAL_EPSILON_DECAY = 0.99995
NEURAL_REPLAY_SIZE = 100_000
NEURAL_BATCH_SIZE = 128
NEURAL_TARGET_UPDATE_INTERVAL = 2_000
NEURAL_MIN_REPLAY_SIZE = 1_000
NEURAL_TRAIN_UPDATES_PER_EPISODE = 2

# =========================
# Game Settings
# =========================

DEALER_STAND_THRESHOLD = 17

# Finite deck settings.
NUM_DECKS = 2

# The game should not reshuffle every round.
# It should reshuffle only when the remaining deck is low.
RESHUFFLE_WHEN_CARDS_REMAINING_BELOW = 26

# Kept for compatibility with older code/tests.
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9,
               10, 10, 10, 10]

DOUBLE_REWARD_MULTIPLIER = 2

MAX_PLAYER_HANDS = 4
ALLOW_RESPLIT = True
ALLOW_DOUBLE_AFTER_SPLIT = True
ALLOW_HIT_SPLIT_ACES = False

# =========================
# Rewards
# =========================

REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_DRAW = 0

# =========================
# File Paths
# =========================

# Relative name only; trainers resolve under agents/<pkg>/results/.
Q_TABLE_PATH = "q_table.json"
