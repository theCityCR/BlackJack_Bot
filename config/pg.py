"""Shared hyperparameters for bet+play policy-gradient agents.

Separate from the flat-bet DQN protocol so published study knobs stay untouched.

§5.7 bake-off used 200k episodes / 5k warm-start / shared entropy 0.01 with
joint bet+play learning. The ``--bet-focus`` CLI preset layers longer budget,
stronger SpreadRule retention, higher bet-head entropy, and frozen rule play.
``--unfreeze`` is the phase-2 preset: load bet_focus weights, thaw play, keep
teacher bet CE, mid-run checkpoints for pause/resume.
"""

# Episode budget aligned with the neural study default wall-clock scale.
PG_TRAINING_EPISODES = 200_000
PG_CHECKPOINT_EVAL_EPISODES = 500
PG_FINAL_EVAL_EPISODES = 25_000
PG_PRINT_INTERVAL = 25_000

PG_LEARNING_RATE = 0.0003
PG_DISCOUNT_FACTOR = 1.0
# Shared fallback / §5.7 default.
PG_ENTROPY_COEF = 0.01
PG_BET_ENTROPY_COEF = 0.01
PG_PLAY_ENTROPY_COEF = 0.01
PG_CRITIC_COEF = 0.5
PG_MAX_GRAD_NORM = 0.5

# Soft retention toward SpreadRule bet during RL (0 = off; §5.7 used 0).
PG_TEACHER_BET_CE_COEF = 0.0

# When True: rule chart plays; only the bet head is trained (§5.7 used False).
PG_FREEZE_PLAY = False

# Bet-focus preset (stake retention after §5.7 constant-stake collapse).
PG_BET_FOCUS_TRAINING_EPISODES = 500_000
PG_BET_FOCUS_WARMSTART_EPISODES = 20_000
PG_BET_FOCUS_BET_ENTROPY_COEF = 0.05
PG_BET_FOCUS_PLAY_ENTROPY_COEF = 0.0
PG_BET_FOCUS_TEACHER_BET_CE_COEF = 0.1
PG_BET_FOCUS_FREEZE_PLAY = True
# Leaner probes so wall-clock goes to RL (bake-off does the heavy eval).
PG_BET_FOCUS_PRINT_INTERVAL = 50_000
PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES = 100
PG_BET_FOCUS_FINAL_EVAL_EPISODES = 2_000
PG_BET_FOCUS_ARTIFACT_SUBDIR = "bet_focus"

# Unfreeze-after-bet-focus: thaw play head from a bet_focus checkpoint while
# retaining teacher stake CE. Mid-run checkpoints + --resume survive shutdown.
PG_UNFREEZE_TRAINING_EPISODES = 200_000
PG_UNFREEZE_WARMSTART_EPISODES = 0
PG_UNFREEZE_BET_ENTROPY_COEF = 0.05
PG_UNFREEZE_PLAY_ENTROPY_COEF = 0.01
PG_UNFREEZE_TEACHER_BET_CE_COEF = 0.1
PG_UNFREEZE_FREEZE_PLAY = False
PG_UNFREEZE_PRINT_INTERVAL = 25_000
PG_UNFREEZE_CHECKPOINT_EVAL_EPISODES = 100
PG_UNFREEZE_FINAL_EVAL_EPISODES = 2_000
PG_UNFREEZE_ARTIFACT_SUBDIR = "unfreeze"
PG_UNFREEZE_INIT_SUBDIR = "bet_focus"

# Behavior cloning from SpreadRuleAgent before RL.
PG_WARMSTART_ENABLED = True
PG_WARMSTART_EPISODES = 5_000
# Accumulate this many CE episodes before one optimizer step.
PG_WARMSTART_BATCH_SIZE = 32

# REINFORCE moving-average baseline (EMA) for return centering.
PG_REINFORCE_BASELINE_MOMENTUM = 0.99

# PPO rollout / optimization.
PG_PPO_CLIP = 0.2
PG_PPO_EPOCHS = 4
PG_PPO_MINIBATCH_SIZE = 64
PG_PPO_ROLLOUT_EPISODES = 64

PG_LEARNING_CURVE_ENABLED = True
PG_LEARNING_CURVE_FILENAME = "learning_curve.csv"
