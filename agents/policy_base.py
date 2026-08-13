"""Shared on-policy base for hierarchical bet+play policy-gradient agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.common import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    SHOE_FEATURE_COUNT,
    STATE_SIZE,
    encode_shoe,
    encode_state,
    resolve_torch_device,
)
from agents.counting import true_count_from_shoe
from agents.networks import BetPlayActorCritic
from agents.rule import RuleAgent
from agents.spread_rule import SpreadRuleAgent
from config import (
    BET_MAX,
    BET_MIN,
    NUM_DECKS,
    PG_BET_ENTROPY_COEF,
    PG_ENTROPY_COEF,
    PG_FREEZE_PLAY,
    PG_PLAY_ENTROPY_COEF,
    PG_TEACHER_BET_CE_COEF,
)
from game import Action, BlackjackGame, GameState, ShoeObservation

BET_ACTION_COUNT = BET_MAX - BET_MIN + 1


def bet_index_to_stake(bet_index: int) -> float:
    if not 0 <= bet_index < BET_ACTION_COUNT:
        raise ValueError(f"bet_index out of range: {bet_index}")
    return float(BET_MIN + bet_index)


def stake_to_bet_index(stake: float) -> int:
    index = int(round(stake)) - BET_MIN
    if not 0 <= index < BET_ACTION_COUNT:
        raise ValueError(f"stake {stake} outside bet action range")
    return index


@dataclass
class BetDecision:
    shoe_features: torch.Tensor
    bet_index: int
    log_prob: float
    value: float
    entropy: float
    teacher_bet_index: int
    return_: float = 0.0


@dataclass
class PlayDecision:
    state_features: torch.Tensor
    action_index: int
    legal_indices: list[int]
    log_prob: float
    value: float
    entropy: float
    hand_index: int
    return_: float = 0.0


@dataclass
class EpisodeTrajectory:
    bet: BetDecision
    play: list[PlayDecision] = field(default_factory=list)
    round_reward: float = 0.0


class BetPlayPolicyAgent(ABC):
    """On-policy hierarchical agent: shoe → stake, then state → play action."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.0003,
        discount_factor: float = 1.0,
        entropy_coef: float = PG_ENTROPY_COEF,
        bet_entropy_coef: float | None = None,
        play_entropy_coef: float | None = None,
        teacher_bet_ce_coef: float = PG_TEACHER_BET_CE_COEF,
        freeze_play: bool = PG_FREEZE_PLAY,
        critic_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | None = None,
        use_critic: bool = True,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        # Prefer head-specific coefs; fall back to shared entropy_coef when unset.
        self.bet_entropy_coef = float(
            PG_BET_ENTROPY_COEF if bet_entropy_coef is None else bet_entropy_coef
        )
        self.play_entropy_coef = float(
            PG_PLAY_ENTROPY_COEF if play_entropy_coef is None else play_entropy_coef
        )
        self.teacher_bet_ce_coef = float(teacher_bet_ce_coef)
        self.freeze_play = bool(freeze_play)
        # When play is frozen, the rule chart owns hit/stand so EV measures stake quality.
        self.use_rule_play = self.freeze_play
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.use_critic = use_critic
        self.device = resolve_torch_device(device)
        self.shoe_size = SHOE_FEATURE_COUNT
        self.state_size = STATE_SIZE
        self.bet_actions = BET_ACTION_COUNT
        self.play_actions = len(ACTION_LIST)
        self.model = BetPlayActorCritic(
            shoe_size=self.shoe_size,
            state_size=self.state_size,
            bet_actions=self.bet_actions,
            play_actions=self.play_actions,
        ).to(self.device)
        self._rule_play = RuleAgent() if self.use_rule_play else None
        self._spread_teacher = SpreadRuleAgent()
        if self.freeze_play:
            self._apply_freeze_play()
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        self.training_steps = 0
        self.last_bet: float = 1.0
        self.last_true_count: float = 0.0
        self.last_shoe: ShoeObservation | None = None
        # Greedy eval path (evaluate_greedy looks for epsilon on DQN agents).
        self.epsilon = 0.0

    def _play_parameter_modules(self) -> tuple[torch.nn.Module, ...]:
        return (self.model.play_body, self.model.play_policy, self.model.play_value)

    def _apply_freeze_play(self) -> None:
        for module in self._play_parameter_modules():
            for param in module.parameters():
                param.requires_grad = False
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.learning_rate)
    def encode_shoe(self, shoe: ShoeObservation) -> torch.Tensor:
        return encode_shoe(shoe)

    def encode_state(self, state: GameState) -> torch.Tensor:
        return encode_state(state, use_shoe_features=True)

    def legal_action_indices(self, available_actions) -> list[int]:
        return [ACTION_TO_INDEX[action] for action in available_actions]

    def teacher_bet_index(self, shoe: ShoeObservation) -> int:
        return stake_to_bet_index(self._spread_teacher.choose_bet(shoe))

    @staticmethod
    def mask_illegal_logits(
        logits: torch.Tensor,
        legal_indices: list[int],
    ) -> torch.Tensor:
        masked = torch.full_like(logits, float("-inf"))
        if logits.dim() == 1:
            masked[legal_indices] = logits[legal_indices]
        else:
            for row in range(logits.size(0)):
                masked[row, legal_indices] = logits[row, legal_indices]
        return masked

    def _bet_distribution(
        self, shoe_features: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.model.bet_logits_value(shoe_features)
        return Categorical(logits=logits), value

    def _play_distribution(
        self,
        state_features: torch.Tensor,
        legal_indices: list[int],
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.model.play_logits_value(state_features)
        masked = self.mask_illegal_logits(logits, legal_indices)
        return Categorical(logits=masked), value

    def choose_bet(self, shoe: ShoeObservation, *, greedy: bool = False) -> float:
        self.last_shoe = shoe
        self.last_true_count = true_count_from_shoe(shoe, num_decks=NUM_DECKS)
        features = self.encode_shoe(shoe).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, _ = self._bet_distribution(features)
            if greedy:
                bet_index = int(dist.probs.argmax(dim=-1).item())
            else:
                bet_index = int(dist.sample().item())
        self.last_bet = bet_index_to_stake(bet_index)
        return self.last_bet

    def choose_action(
        self,
        state: GameState,
        available_actions,
        *,
        greedy: bool = False,
    ) -> Action:
        if self.use_rule_play:
            assert self._rule_play is not None
            return self._rule_play.choose_action(state, available_actions)
        legal_indices = self.legal_action_indices(available_actions)
        features = self.encode_state(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, _ = self._play_distribution(features, legal_indices)
            if greedy:
                action_index = int(dist.probs.argmax(dim=-1).item())
            else:
                action_index = int(dist.sample().item())
        return ACTION_LIST[action_index]

    def _sample_bet_decision(self, shoe: ShoeObservation) -> BetDecision:
        self.last_shoe = shoe
        self.last_true_count = true_count_from_shoe(shoe, num_decks=NUM_DECKS)
        features = self.encode_shoe(shoe).to(self.device)
        dist, value = self._bet_distribution(features.unsqueeze(0))
        sample = dist.sample()
        bet_index = int(sample.item())
        self.last_bet = bet_index_to_stake(bet_index)
        return BetDecision(
            shoe_features=features.detach().cpu(),
            bet_index=bet_index,
            log_prob=float(dist.log_prob(sample).item()),
            value=float(value.item()),
            entropy=float(dist.entropy().item()),
            teacher_bet_index=self.teacher_bet_index(shoe),
        )

    def _sample_play_decision(
        self,
        state: GameState,
        available_actions,
        hand_index: int,
    ) -> tuple[Action, PlayDecision]:
        legal_indices = self.legal_action_indices(available_actions)
        features = self.encode_state(state).to(self.device)
        dist, value = self._play_distribution(features.unsqueeze(0), legal_indices)
        sample = dist.sample()
        action_index = int(sample.item())
        decision = PlayDecision(
            state_features=features.detach().cpu(),
            action_index=action_index,
            legal_indices=list(legal_indices),
            log_prob=float(dist.log_prob(sample).item()),
            value=float(value.item()),
            entropy=float(dist.entropy().item()),
            hand_index=hand_index,
        )
        return ACTION_LIST[action_index], decision

    def _assign_returns(self, trajectory: EpisodeTrajectory, game: BlackjackGame) -> None:
        trajectory.bet.return_ = float(trajectory.round_reward)
        hand_rewards = game.hand_rewards
        for step in trajectory.play:
            reward = hand_rewards[step.hand_index]
            step.return_ = 0.0 if reward is None else float(reward)

    def collect_trajectory(self, game: BlackjackGame) -> EpisodeTrajectory:
        """Roll out one bet+play round without updating parameters."""
        shoe = game.prepare_round()
        bet_decision = self._sample_bet_decision(shoe)
        state = game.deal(bet=self.last_bet)
        trajectory = EpisodeTrajectory(bet=bet_decision)

        if state is None:
            trajectory.round_reward = float(game.round_reward)
            self._assign_returns(trajectory, game)
            return trajectory

        done = False
        while not done:
            available = game.available_actions()
            if self.use_rule_play:
                assert self._rule_play is not None
                action = self._rule_play.choose_action(state, available)
            else:
                hand_index = game.active_hand_index
                action, play_decision = self._sample_play_decision(
                    state, available, hand_index
                )
                trajectory.play.append(play_decision)
            state, reward, done = game.step(action)
            if done:
                trajectory.round_reward = float(reward)

        self._assign_returns(trajectory, game)
        return trajectory

    def play_episode(self, game: BlackjackGame, render: bool = False) -> float:
        """Greedy bet + play for evaluation."""
        shoe = game.prepare_round()
        bet = self.choose_bet(shoe, greedy=True)
        state = game.deal(bet=bet)

        if state is None:
            if render:
                game.render()
            return float(game.round_reward)

        done = False
        if render:
            game.render()

        reward = 0.0
        while not done:
            available_actions = game.available_actions()
            action = self.choose_action(state, available_actions, greedy=True)
            next_state, reward, done = game.step(action)
            if render:
                print(f"Action: {action.name}")
                game.render()
                print()
            state = next_state

        return float(reward)

    def train_one_episode(self, game: BlackjackGame) -> float:
        trajectory = self.collect_trajectory(game)
        self.update_from_trajectory(trajectory)
        return float(trajectory.round_reward)

    @abstractmethod
    def update_from_trajectory(self, trajectory: EpisodeTrajectory) -> dict[str, float]:
        """Apply one on-policy update from a finished episode."""

    def _stack_bet_batch(
        self, decisions: list[BetDecision]
    ) -> dict[str, torch.Tensor]:
        shoes = torch.stack([d.shoe_features for d in decisions]).to(self.device)
        return {
            "shoe": shoes,
            "action": torch.tensor(
                [d.bet_index for d in decisions], dtype=torch.long, device=self.device
            ),
            "teacher_action": torch.tensor(
                [d.teacher_bet_index for d in decisions],
                dtype=torch.long,
                device=self.device,
            ),
            "old_log_prob": torch.tensor(
                [d.log_prob for d in decisions], dtype=torch.float32, device=self.device
            ),
            "returns": torch.tensor(
                [d.return_ for d in decisions], dtype=torch.float32, device=self.device
            ),
            "old_value": torch.tensor(
                [d.value for d in decisions], dtype=torch.float32, device=self.device
            ),
        }

    def _stack_play_batch(
        self, decisions: list[PlayDecision]
    ) -> dict[str, Any] | None:
        if not decisions:
            return None
        states = torch.stack([d.state_features for d in decisions]).to(self.device)
        return {
            "state": states,
            "action": torch.tensor(
                [d.action_index for d in decisions],
                dtype=torch.long,
                device=self.device,
            ),
            "old_log_prob": torch.tensor(
                [d.log_prob for d in decisions],
                dtype=torch.float32,
                device=self.device,
            ),
            "returns": torch.tensor(
                [d.return_ for d in decisions],
                dtype=torch.float32,
                device=self.device,
            ),
            "old_value": torch.tensor(
                [d.value for d in decisions],
                dtype=torch.float32,
                device=self.device,
            ),
            "legal_indices": [list(d.legal_indices) for d in decisions],
        }

    def evaluate_bet_log_probs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model.bet_logits_value(batch["shoe"])
        dist = Categorical(logits=logits)
        return dist.log_prob(batch["action"]), values, dist.entropy()

    def evaluate_play_log_probs(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model.play_logits_value(batch["state"])
        masked_rows = []
        for row, legal in enumerate(batch["legal_indices"]):
            masked_rows.append(self.mask_illegal_logits(logits[row], legal))
        masked = torch.stack(masked_rows, dim=0)
        dist = Categorical(logits=masked)
        return dist.log_prob(batch["action"]), values, dist.entropy()

    def teacher_bet_ce_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.teacher_bet_ce_coef <= 0.0:
            return torch.zeros((), device=self.device)
        logits, _ = self.model.bet_logits_value(batch["shoe"])
        return F.cross_entropy(logits, batch["teacher_action"])

    def _clip_grads(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
