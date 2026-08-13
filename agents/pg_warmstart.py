"""Behavior cloning warm-start from SpreadRuleAgent for bet+play PG agents."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from agents.common import ACTION_TO_INDEX
from agents.policy_base import stake_to_bet_index
from agents.spread_rule import SpreadRuleAgent
from config import PG_WARMSTART_BATCH_SIZE


def warmstart_from_spread_rule(
    agent: Any,
    game: Any,
    num_episodes: int,
    *,
    bet_only: bool | None = None,
    batch_size: int = PG_WARMSTART_BATCH_SIZE,
) -> None:
    """Clone Hi-Lo bet + rule play into ``agent`` via supervised CE.

    When ``bet_only`` is True (default: agent's ``freeze_play``), only the bet
    head is cloned; play stays on the rule chart at train/eval time.

    Losses accumulate over ``batch_size`` episodes before one optimizer step
    so long warm-starts are not one backward per shoe.
    """
    clone_play = not (
        agent.freeze_play if bet_only is None else bool(bet_only)
    )
    teacher = SpreadRuleAgent()
    total_bet_loss = 0.0
    total_play_loss = 0.0
    bet_steps = 0
    play_steps = 0
    batch = max(1, int(batch_size))

    pending_loss: torch.Tensor | None = None
    pending_count = 0

    def _flush() -> None:
        nonlocal pending_loss, pending_count
        if pending_loss is None or pending_count == 0:
            return
        loss = pending_loss / pending_count
        agent.optimizer.zero_grad()
        loss.backward()
        agent._clip_grads()
        agent.optimizer.step()
        agent.training_steps += 1
        pending_loss = None
        pending_count = 0

    for _ in range(num_episodes):
        shoe = game.prepare_round()
        target_bet = teacher.choose_bet(shoe)
        target_bet_index = stake_to_bet_index(target_bet)

        shoe_features = agent.encode_shoe(shoe).unsqueeze(0).to(agent.device)
        bet_logits, _ = agent.model.bet_logits_value(shoe_features)
        bet_loss = F.cross_entropy(
            bet_logits,
            torch.tensor([target_bet_index], dtype=torch.long, device=agent.device),
        )

        state = game.deal(bet=target_bet)
        play_loss = torch.tensor(0.0, device=agent.device)
        n_play = 0

        if state is not None:
            done = False
            while not done:
                available = game.available_actions()
                target_action = teacher.choose_action(state, available)
                if clone_play:
                    legal = agent.legal_action_indices(available)
                    features = agent.encode_state(state).unsqueeze(0).to(agent.device)
                    logits, _ = agent.model.play_logits_value(features)
                    masked = agent.mask_illegal_logits(logits[0], legal).unsqueeze(0)
                    play_loss = play_loss + F.cross_entropy(
                        masked,
                        torch.tensor(
                            [ACTION_TO_INDEX[target_action]],
                            dtype=torch.long,
                            device=agent.device,
                        ),
                    )
                    n_play += 1
                state, _, done = game.step(target_action)

        episode_loss = bet_loss + (play_loss if clone_play else 0.0)
        pending_loss = (
            episode_loss if pending_loss is None else pending_loss + episode_loss
        )
        pending_count += 1
        if pending_count >= batch:
            _flush()

        total_bet_loss += float(bet_loss.item())
        bet_steps += 1
        if n_play:
            total_play_loss += float(play_loss.item())
            play_steps += n_play

    _flush()

    avg_bet = total_bet_loss / max(1, bet_steps)
    avg_play = total_play_loss / max(1, play_steps)
    mode = "bet-only" if not clone_play else "bet+play"
    print(
        f"Warm-start ({mode}): cloned spread rule for {num_episodes} episodes "
        f"(batch={batch}, bet_ce={avg_bet:.4f}, play_ce={avg_play:.4f}, "
        f"steps={agent.training_steps})"
    )
