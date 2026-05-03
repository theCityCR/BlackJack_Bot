"""
config.py

Global configuration values for the Blackjack ML project.
"""

# =========================
# Training Settings
# =========================

NUM_TRAINING_EPISODES = 50_000
EVALUATION_EPISODES = 5_000

# =========================
# Q-Learning Hyperparameters
# =========================

LEARNING_RATE = 0.1          # α
DISCOUNT_FACTOR = 1.0        # γ (Blackjack is episodic, so 1.0 is fine)

EPSILON_START = 1.0          # initial exploration
EPSILON_END = 0.05           # minimum exploration
EPSILON_DECAY = 0.99999      # decay per step (not per episode)

# =========================
# Game Settings
# =========================

DEALER_STAND_THRESHOLD = 17  # dealer stands on 17+

# Infinite deck card distribution.
# Ace is represented as 1. Face cards are represented as 10.
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9,
               10, 10, 10, 10]  # 10, J, Q, K

# Double-down settings.
# In standard blackjack, double is usually available only as the first decision
# on a two-card hand. The game environment will enforce this.
DOUBLE_REWARD_MULTIPLIER = 2

# Split settings.
# These constants are not used by cards.py directly, but they give game.py
# one clear place to read the rule choices from when split is implemented.
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

Q_TABLE_PATH = "results/q_table.json"
