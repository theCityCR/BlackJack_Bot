"""Casino rules for the finite-shoe Blackjack environment."""

# Dealer stands on all 17s (S17), including soft 17.
DEALER_STAND_THRESHOLD = 17

# Finite deck settings.
NUM_DECKS = 2

# Cut card: reshuffle when remaining cards ≤ this value (not every round).
# Default 26 on a 2-deck shoe ≈ 75% dealt penetration. Override via
# BlackjackGame(reshuffle_threshold=…) / CLI --reshuffle-threshold.
RESHUFFLE_WHEN_CARDS_REMAINING_BELOW = 26

DOUBLE_REWARD_MULTIPLIER = 2
BLACKJACK_PAYOUT = 1.5

MAX_PLAYER_HANDS = 4
ALLOW_RESPLIT = True
ALLOW_DOUBLE_AFTER_SPLIT = True
ALLOW_HIT_SPLIT_ACES = False

REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_DRAW = 0

# Variable betting (flat protocols keep bet=1.0). Stake units for Hi-Lo spread.
BET_MIN = 1
BET_MAX = 8
