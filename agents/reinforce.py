"""REINFORCE agent for hierarchical bet+play Blackjack."""

from __future__ import annotations

import torch

from agents.policy_base import BetPlayPolicyAgent, EpisodeTrajectory
from config import (
    PG_DISCOUNT_FACTOR,
    PG_ENTROPY_COEF,
    PG_LEARNING_RATE,
    PG_MAX_GRAD_NORM,
    PG_REINFORCE_BASELINE_MOMENTUM,
)


class ReinforceAgent(BetPlayPolicyAgent):
    """Vanilla policy gradient with optional EMA return baseline (no critic)."""

    def __init__(
        self,
        *,
        learning_rate: float = PG_LEARNING_RATE,
        discount_factor: float = PG_DISCOUNT_FACTOR,
        entropy_coef: float = PG_ENTROPY_COEF,
        max_grad_norm: float = PG_MAX_GRAD_NORM,
        baseline_momentum: float = PG_REINFORCE_BASELINE_MOMENTUM,
        device: str | None = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            entropy_coef=entropy_coef,
            critic_coef=0.0,
            max_grad_norm=max_grad_norm,
            device=device,
            use_critic=False,
        )
        self.baseline_momentum = baseline_momentum
        self.return_baseline = 0.0
        self._baseline_initialized = False

    def _center(self, returns: torch.Tensor) -> torch.Tensor:
        mean = float(returns.mean().item())
        if not self._baseline_initialized:
            self.return_baseline = mean
            self._baseline_initialized = True
        else:
            self.return_baseline = (
                self.baseline_momentum * self.return_baseline
                + (1.0 - self.baseline_momentum) * mean
            )
        return returns - self.return_baseline

    def update_from_trajectory(
        self, trajectory: EpisodeTrajectory
    ) -> dict[str, float]:
        bet_batch = self._stack_bet_batch([trajectory.bet])
        play_batch = self._stack_play_batch(trajectory.play)

        bet_log_prob, _, bet_entropy = self.evaluate_bet_log_probs(bet_batch)
        bet_adv = self._center(bet_batch["returns"])
        loss = -(bet_log_prob * bet_adv.detach()).mean()
        entropy = bet_entropy.mean()

        if play_batch is not None:
            play_log_prob, _, play_entropy = self.evaluate_play_log_probs(play_batch)
            play_adv = self._center(play_batch["returns"])
            loss = loss - (play_log_prob * play_adv.detach()).mean()
            entropy = entropy + play_entropy.mean()

        loss = loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        self.training_steps += 1
        return {"loss": float(loss.item()), "entropy": float(entropy.item())}
