"""Advantage Actor-Critic (A2C) for hierarchical bet+play Blackjack."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agents.policy_base import BetPlayPolicyAgent, EpisodeTrajectory
from config import (
    PG_CRITIC_COEF,
    PG_DISCOUNT_FACTOR,
    PG_ENTROPY_COEF,
    PG_LEARNING_RATE,
    PG_MAX_GRAD_NORM,
)


class A2CAgent(BetPlayPolicyAgent):
    """On-policy actor-critic with Monte Carlo advantages (γ=1)."""

    def __init__(
        self,
        *,
        learning_rate: float = PG_LEARNING_RATE,
        discount_factor: float = PG_DISCOUNT_FACTOR,
        entropy_coef: float = PG_ENTROPY_COEF,
        critic_coef: float = PG_CRITIC_COEF,
        max_grad_norm: float = PG_MAX_GRAD_NORM,
        device: str | None = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_critic=True,
        )

    def update_from_trajectory(
        self, trajectory: EpisodeTrajectory
    ) -> dict[str, float]:
        bet_batch = self._stack_bet_batch([trajectory.bet])
        play_batch = self._stack_play_batch(trajectory.play)

        bet_log_prob, bet_values, bet_entropy = self.evaluate_bet_log_probs(bet_batch)
        bet_adv = bet_batch["returns"] - bet_values.detach()
        policy_loss = -(bet_log_prob * bet_adv).mean()
        value_loss = F.mse_loss(bet_values, bet_batch["returns"])
        entropy = bet_entropy.mean()

        if play_batch is not None:
            play_log_prob, play_values, play_entropy = self.evaluate_play_log_probs(
                play_batch
            )
            play_adv = play_batch["returns"] - play_values.detach()
            policy_loss = policy_loss - (play_log_prob * play_adv).mean()
            value_loss = value_loss + F.mse_loss(play_values, play_batch["returns"])
            entropy = entropy + play_entropy.mean()

        loss = (
            policy_loss
            + self.critic_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        self.training_steps += 1
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
