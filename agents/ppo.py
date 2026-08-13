"""Proximal Policy Optimization for hierarchical bet+play Blackjack."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from agents.policy_base import (
    BetDecision,
    BetPlayPolicyAgent,
    EpisodeTrajectory,
    PlayDecision,
)
from config import (
    PG_BET_ENTROPY_COEF,
    PG_CRITIC_COEF,
    PG_DISCOUNT_FACTOR,
    PG_ENTROPY_COEF,
    PG_FREEZE_PLAY,
    PG_LEARNING_RATE,
    PG_MAX_GRAD_NORM,
    PG_PLAY_ENTROPY_COEF,
    PG_PPO_CLIP,
    PG_PPO_EPOCHS,
    PG_PPO_MINIBATCH_SIZE,
    PG_PPO_ROLLOUT_EPISODES,
    PG_TEACHER_BET_CE_COEF,
)
from game import BlackjackGame


class PPOAgent(BetPlayPolicyAgent):
    """Clipped PPO with Monte Carlo advantages over short bet+play rollouts."""

    def __init__(
        self,
        *,
        learning_rate: float = PG_LEARNING_RATE,
        discount_factor: float = PG_DISCOUNT_FACTOR,
        entropy_coef: float = PG_ENTROPY_COEF,
        bet_entropy_coef: float | None = PG_BET_ENTROPY_COEF,
        play_entropy_coef: float | None = PG_PLAY_ENTROPY_COEF,
        teacher_bet_ce_coef: float = PG_TEACHER_BET_CE_COEF,
        freeze_play: bool = PG_FREEZE_PLAY,
        critic_coef: float = PG_CRITIC_COEF,
        max_grad_norm: float = PG_MAX_GRAD_NORM,
        clip_epsilon: float = PG_PPO_CLIP,
        ppo_epochs: int = PG_PPO_EPOCHS,
        minibatch_size: int = PG_PPO_MINIBATCH_SIZE,
        rollout_episodes: int = PG_PPO_ROLLOUT_EPISODES,
        device: str | None = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            entropy_coef=entropy_coef,
            bet_entropy_coef=bet_entropy_coef,
            play_entropy_coef=play_entropy_coef,
            teacher_bet_ce_coef=teacher_bet_ce_coef,
            freeze_play=freeze_play,
            critic_coef=critic_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_critic=True,
        )
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_episodes = max(1, int(rollout_episodes))
        self._bet_buffer: list[BetDecision] = []
        self._play_buffer: list[PlayDecision] = []
        self._episodes_in_rollout = 0

    def clear_rollout(self) -> None:
        self._bet_buffer.clear()
        self._play_buffer.clear()
        self._episodes_in_rollout = 0

    def train_one_episode(self, game: BlackjackGame) -> float:
        trajectory = self.collect_trajectory(game)
        self._bet_buffer.append(trajectory.bet)
        self._play_buffer.extend(trajectory.play)
        self._episodes_in_rollout += 1
        if self._episodes_in_rollout >= self.rollout_episodes:
            self.update_from_rollout()
            self.clear_rollout()
        return float(trajectory.round_reward)

    def update_from_trajectory(
        self, trajectory: EpisodeTrajectory
    ) -> dict[str, float]:
        """Single-episode PPO update (used by tests / flush)."""
        self._bet_buffer.append(trajectory.bet)
        self._play_buffer.extend(trajectory.play)
        metrics = self.update_from_rollout()
        self.clear_rollout()
        return metrics

    def update_from_rollout(self) -> dict[str, float]:
        if not self._bet_buffer:
            return {"loss": 0.0}

        bet_batch = self._stack_bet_batch(self._bet_buffer)
        play_batch = None if self.freeze_play else self._stack_play_batch(self._play_buffer)
        last_loss = 0.0

        for _ in range(self.ppo_epochs):
            last_loss = self._ppo_epoch(bet_batch, play_batch)

        self.training_steps += 1
        return {"loss": last_loss}

    def _ppo_epoch(
        self,
        bet_batch: dict[str, torch.Tensor],
        play_batch: dict[str, Any] | None,
    ) -> float:
        loss_total = 0.0
        updates = 0

        bet_n = bet_batch["action"].size(0)
        for start in range(0, bet_n, self.minibatch_size):
            end = min(start + self.minibatch_size, bet_n)
            mb = {k: v[start:end] for k, v in bet_batch.items()}
            loss_total += self._ppo_bet_step(mb)
            updates += 1

        if play_batch is not None:
            play_n = play_batch["action"].size(0)
            for start in range(0, play_n, self.minibatch_size):
                end = min(start + self.minibatch_size, play_n)
                mb = {
                    k: (v[start:end] if torch.is_tensor(v) else v[start:end])
                    for k, v in play_batch.items()
                }
                loss_total += self._ppo_play_step(mb)
                updates += 1

        return loss_total / max(1, updates)

    def _ppo_bet_step(self, batch: dict[str, torch.Tensor]) -> float:
        log_prob, values, entropy = self.evaluate_bet_log_probs(batch)
        advantages = batch["returns"] - batch["old_value"]
        advantages = advantages - advantages.mean()
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        unclipped = ratio * advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, batch["returns"])
        teacher_ce = self.teacher_bet_ce_loss(batch)
        loss = (
            policy_loss
            + self.critic_coef * value_loss
            - self.bet_entropy_coef * entropy.mean()
            + self.teacher_bet_ce_coef * teacher_ce
        )
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return float(loss.item())

    def _ppo_play_step(self, batch: dict[str, Any]) -> float:
        log_prob, values, entropy = self.evaluate_play_log_probs(batch)
        advantages = batch["returns"] - batch["old_value"]
        advantages = advantages - advantages.mean()
        ratio = torch.exp(log_prob - batch["old_log_prob"])
        unclipped = ratio * advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, batch["returns"])
        loss = (
            policy_loss
            + self.critic_coef * value_loss
            - self.play_entropy_coef * entropy.mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return float(loss.item())
