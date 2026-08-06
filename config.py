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
# Shared experimental protocol: equal episode budgets and optimizer schedules so
# architecture / curriculum / warm-start comparisons are not confounded by compute.
# Architecture (network topology, Double/Dueling targets, prioritized replay)
# remains an intentional experimental factor.

NEURAL_TRAINING_EPISODES = 200_000
# Sparse mid-run greedy probes keep wall-clock down; final eval stays large.
NEURAL_CHECKPOINT_EVAL_EPISODES = 500
NEURAL_FINAL_EVAL_EPISODES = 25_000
NEURAL_PRINT_INTERVAL = 25_000

NEURAL_LEARNING_RATE = 0.001
NEURAL_DISCOUNT_FACTOR = 1.0
NEURAL_EPSILON_START = 1.0
NEURAL_EPSILON_MIN = 0.05
# Slower decay keeps exploration into mid-training under the 200k budget.
NEURAL_EPSILON_DECAY = 0.99997
NEURAL_REPLAY_SIZE = 100_000
NEURAL_BATCH_SIZE = 128
NEURAL_TARGET_UPDATE_INTERVAL = 2_000
NEURAL_MIN_REPLAY_SIZE = 1_000
# Two updates/episode is a speed/quality compromise vs 4 (final metric is still
# greedy eval after the full episode budget).
NEURAL_TRAIN_UPDATES_PER_EPISODE = 2

# State curriculum: Phase A hand features only; Phase B full shoe-aware state.
# Replay is cleared at the phase boundary so encodings stay consistent.
NEURAL_CURRICULUM_ENABLED = True
NEURAL_CURRICULUM_PHASE_A_EPISODES = 100_000

# Behavior cloning from the rule baseline before RL (hand-only encoding when
# curriculum is on). Disable with trainer --no-warmstart.
NEURAL_WARMSTART_ENABLED = True
NEURAL_WARMSTART_EPISODES = 5_000

# Periodic greedy-eval rows written beside checkpoints.
NEURAL_LEARNING_CURVE_ENABLED = True
NEURAL_LEARNING_CURVE_FILENAME = "learning_curve.csv"

# Double DQN ablation condition ids (see agents/study_protocol.py).
ABLATION_CONDITION_A = "A_full_scratch"
ABLATION_CONDITION_B = "B_hand_only"
ABLATION_CONDITION_C = "C_curriculum"
ABLATION_CONDITION_D = "D_curriculum_warmstart"

# Hand-only gap-close experiment: true 8-D encoder, long rule cloning, long RL.
GAP_CLOSE_WARMSTART_EPISODES = 100_000
GAP_CLOSE_TRAINING_EPISODES = 500_000
GAP_CLOSE_EVAL_EPISODES = 25_000
GAP_CLOSE_PRINT_INTERVAL = 25_000
GAP_CLOSE_CHECKPOINT_EVAL_EPISODES = 500
GAP_CLOSE_EPSILON_MIN = 0.01

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
