"""Shared hyperparameters for bet+play policy-gradient agents.

Separate from the flat-bet DQN protocol so published study knobs stay untouched.
"""

# Episode budget aligned with the neural study default wall-clock scale.
PG_TRAINING_EPISODES = 200_000
PG_CHECKPOINT_EVAL_EPISODES = 500
PG_FINAL_EVAL_EPISODES = 25_000
PG_PRINT_INTERVAL = 25_000

PG_LEARNING_RATE = 0.0003
PG_DISCOUNT_FACTOR = 1.0
PG_ENTROPY_COEF = 0.01
PG_CRITIC_COEF = 0.5
PG_MAX_GRAD_NORM = 0.5

# REINFORCE moving-average baseline (EMA) for return centering.
PG_REINFORCE_BASELINE_MOMENTUM = 0.99

# PPO rollout / optimization.
PG_PPO_CLIP = 0.2
PG_PPO_EPOCHS = 4
PG_PPO_MINIBATCH_SIZE = 64
PG_PPO_ROLLOUT_EPISODES = 64

# Behavior cloning from SpreadRuleAgent before RL.
PG_WARMSTART_ENABLED = True
PG_WARMSTART_EPISODES = 5_000

PG_LEARNING_CURVE_ENABLED = True
PG_LEARNING_CURVE_FILENAME = "learning_curve.csv"
