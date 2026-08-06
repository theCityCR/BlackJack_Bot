"""Shared utilities for neural Blackjack agents."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import (
    MAX_PLAYER_HANDS,
    NEURAL_BATCH_SIZE,
    NEURAL_CHECKPOINT_EVAL_EPISODES,
    NEURAL_CURRICULUM_ENABLED,
    NEURAL_CURRICULUM_PHASE_A_EPISODES,
    NEURAL_DISCOUNT_FACTOR,
    NEURAL_EPSILON_DECAY,
    NEURAL_EPSILON_MIN,
    NEURAL_EPSILON_START,
    NEURAL_LEARNING_CURVE_ENABLED,
    NEURAL_LEARNING_CURVE_FILENAME,
    NEURAL_LEARNING_RATE,
    NEURAL_MIN_REPLAY_SIZE,
    NEURAL_PRINT_INTERVAL,
    NEURAL_REPLAY_SIZE,
    NEURAL_TARGET_UPDATE_INTERVAL,
    NEURAL_TRAIN_UPDATES_PER_EPISODE,
    NEURAL_WARMSTART_ENABLED,
    NEURAL_WARMSTART_EPISODES,
    NUM_DECKS,
)
from game import Action, GameState


ACTION_LIST = [Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_LIST)}

INITIAL_SHOE_SIZE = 52 * NUM_DECKS
HAND_FEATURE_COUNT = 8
SHOE_FEATURE_COUNT = 11  # remaining-card fraction + 10 rank counts
STATE_SIZE = HAND_FEATURE_COUNT + SHOE_FEATURE_COUNT


@dataclass
class Transition:
    state: torch.Tensor
    action_index: int
    reward: float
    next_state: torch.Tensor | None
    done: bool
    next_legal_action_indices: list[int]


def encode_state(
    state: GameState,
    *,
    use_shoe_features: bool = True,
) -> torch.Tensor:
    """Encode a GameState into the shared 19-feature vector.

    When ``use_shoe_features`` is False (curriculum phase A), the shoe fraction
    and remaining-card counts are zeroed so the agent learns hand policy first.
    """
    count_vector = tuple(state.count_vector)
    cards_remaining = sum(count_vector)

    hand_features = [
        state.player_value / 21,
        state.dealer_upcard / 10,
        float(state.usable_ace),
        float(state.can_double),
        float(state.can_split),
        float(state.is_split_hand),
        state.active_hand_index / MAX_PLAYER_HANDS,
        state.num_hands / MAX_PLAYER_HANDS,
    ]

    if use_shoe_features:
        if cards_remaining == 0:
            normalized_count_vector = [0.0] * 10
        else:
            normalized_count_vector = [
                count / cards_remaining for count in count_vector
            ]
        shoe_features = [
            cards_remaining / INITIAL_SHOE_SIZE,
            *normalized_count_vector,
        ]
    else:
        shoe_features = [0.0] * SHOE_FEATURE_COUNT

    return torch.tensor(
        hand_features + shoe_features,
        dtype=torch.float32,
    )


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible training runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Resolve the torch device for neural agents.

    Precedence:
    1. Explicit ``device`` argument
    2. ``BLACKJACK_TORCH_DEVICE`` env var
    3. CUDA if available
    4. CPU

    Apple MPS is supported when requested explicitly (``device="mps"`` or the
    env var), but is not chosen automatically: for these small Blackjack DQN
    batches, host↔MPS copies usually make training slower than CPU.
    """
    import os

    if device is None:
        device = os.environ.get("BLACKJACK_TORCH_DEVICE")

    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def categorize_reward(reward: float) -> str:
    if reward == 0:
        return "draw"
    if reward == 1:
        return "normal_win"
    if reward == -1:
        return "normal_loss"
    if reward == 1.5:
        return "blackjack_win"
    if reward > 1:
        return "big_win_double_or_split"
    if reward < -1:
        return "big_loss_double_or_split"
    return "other"


def print_distribution(distribution: dict[str, int]) -> None:
    total = sum(distribution.values())
    for category in sorted(distribution):
        count = distribution[category]
        percentage = count / total if total else 0.0
        print(f"{category:30s}: {count:8d} ({percentage:.2%})")


def neural_training_kwargs() -> dict[str, Any]:
    """Shared optimizer / replay hyperparameters for all neural agents."""
    return {
        "learning_rate": NEURAL_LEARNING_RATE,
        "discount_factor": NEURAL_DISCOUNT_FACTOR,
        "epsilon": NEURAL_EPSILON_START,
        "epsilon_min": NEURAL_EPSILON_MIN,
        "epsilon_decay": NEURAL_EPSILON_DECAY,
        "replay_size": NEURAL_REPLAY_SIZE,
        "batch_size": NEURAL_BATCH_SIZE,
        "target_update_interval": NEURAL_TARGET_UPDATE_INTERVAL,
        "min_replay_size": NEURAL_MIN_REPLAY_SIZE,
        "train_updates_per_episode": NEURAL_TRAIN_UPDATES_PER_EPISODE,
    }


def clear_replay_buffer(agent: Any) -> None:
    """Drop stored transitions (used at curriculum phase boundaries)."""
    buffer = agent.replay_buffer
    if hasattr(buffer, "clear"):
        buffer.clear()
        return

    raise TypeError(
        f"Replay buffer {type(buffer).__name__} does not support clear()"
    )


def run_neural_training_loop(
    agent: Any,
    game: Any,
    num_episodes: int,
    *,
    print_interval: int = NEURAL_PRINT_INTERVAL,
    checkpoint_eval_episodes: int = NEURAL_CHECKPOINT_EVAL_EPISODES,
    curriculum: bool | None = None,
    phase_a_episodes: int | None = None,
    warmstart: bool | None = None,
    warmstart_episodes: int | None = None,
    learning_curve_path: Path | str | None = None,
    force_shoe_off: bool = False,
) -> Any:
    """Train for ``num_episodes`` with periodic greedy evaluation logging.

    Experimental protocol hooks:
    - Curriculum: phase A hand-only encodings, then shoe-aware phase B.
    - Warm-start: clone the rule baseline before RL (hand-only when curriculum
      or ``force_shoe_off`` is active).
    - Learning curves: optional CSV of greedy eval metrics at print intervals.
    - ``force_shoe_off``: keep shoe features disabled for the entire run
      (ablation B).
    """
    from agents.learning_curves import LearningCurveLogger
    from agents.warmstart import warmstart_from_rule_agent

    device = getattr(agent, "device", None)
    if device is not None:
        print(f"Training device: {device}")

    use_curriculum = (
        NEURAL_CURRICULUM_ENABLED if curriculum is None else curriculum
    )
    phase_a = (
        NEURAL_CURRICULUM_PHASE_A_EPISODES
        if phase_a_episodes is None
        else phase_a_episodes
    )
    if not use_curriculum or force_shoe_off:
        phase_a = 0

    phase_a = max(0, min(phase_a, num_episodes))
    if force_shoe_off:
        agent.use_shoe_features = False
    else:
        agent.use_shoe_features = phase_a == 0

    use_warmstart = (
        NEURAL_WARMSTART_ENABLED if warmstart is None else warmstart
    )
    clone_episodes = (
        NEURAL_WARMSTART_EPISODES
        if warmstart_episodes is None
        else warmstart_episodes
    )
    if use_warmstart and clone_episodes > 0:
        # Match phase-A / hand-only encoding during cloning.
        if force_shoe_off or phase_a > 0:
            agent.use_shoe_features = False
        warmstart_from_rule_agent(agent, game, clone_episodes)
        if force_shoe_off:
            agent.use_shoe_features = False
        elif phase_a > 0:
            agent.use_shoe_features = False
        else:
            agent.use_shoe_features = True

    if use_curriculum and phase_a > 0 and not force_shoe_off:
        print(
            f"Curriculum phase A: hand features only "
            f"(episodes 1–{phase_a})"
        )
    if force_shoe_off:
        print("Ablation: shoe features forced off for entire run")

    curve_logger = None
    log_curves = NEURAL_LEARNING_CURVE_ENABLED
    if learning_curve_path is not None and log_curves:
        curve_logger = LearningCurveLogger(learning_curve_path)

    total_training_reward = 0.0

    for episode in range(1, num_episodes + 1):
        if (
            use_curriculum
            and phase_a > 0
            and not force_shoe_off
            and episode == phase_a + 1
        ):
            agent.use_shoe_features = True
            clear_replay_buffer(agent)
            print(
                f"Curriculum phase B: enabling shoe features "
                f"(episodes {phase_a + 1}–{num_episodes}); "
                "cleared replay buffer"
            )

        reward = agent.train_one_episode(game)
        total_training_reward += reward

        if episode % print_interval == 0:
            eval_reward, eval_distribution = evaluate_greedy(
                agent,
                checkpoint_eval_episodes,
            )
            shoe_mode = "on" if agent.use_shoe_features else "off"
            print(f"Episode {episode}")
            print(f"Average training reward: {total_training_reward / episode:.4f}")
            print(f"Evaluation reward:        {eval_reward:.4f}")
            print(f"Epsilon:                  {agent.epsilon:.4f}")
            print(f"Shoe features:            {shoe_mode}")
            print(f"Replay buffer size:       {len(agent.replay_buffer)}")
            print(f"Training steps:           {agent.training_steps}")
            print("Evaluation distribution:")
            print_distribution(eval_distribution)
            print()

            if curve_logger is not None:
                curve_logger.append(
                    episode=episode,
                    training_steps=agent.training_steps,
                    eval_reward=eval_reward,
                    epsilon=agent.epsilon,
                    shoe_features_on=bool(agent.use_shoe_features),
                )

    return agent


def evaluate_greedy(agent: Any, num_episodes: int) -> tuple[float, dict[str, int]]:
    """Run greedy episodes and return mean reward plus outcome distribution."""
    from game import BlackjackGame

    game = BlackjackGame()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_reward = 0.0
    distribution: dict[str, int] = defaultdict(int)

    for _ in range(num_episodes):
        reward = agent.play_episode(game)
        total_reward += reward
        distribution[categorize_reward(reward)] += 1

    agent.epsilon = old_epsilon
    return total_reward / num_episodes, distribution


def save_torch_checkpoint(agent: Any, path: Path | str) -> Path:
    """Persist a neural agent's weights and training metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": agent.model.state_dict(),
        "target_model_state_dict": agent.target_model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "training_steps": agent.training_steps,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def package_results_path(package_file: str, filename: str) -> Path:
    """Resolve agents/<pkg>/results/<filename> from a module __file__."""
    return Path(package_file).resolve().parent / "results" / filename
