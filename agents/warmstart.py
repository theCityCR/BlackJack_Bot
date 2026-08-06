"""Behavior cloning warm-start from the rule-based baseline.

Drive episodes with RuleAgent, store transitions via agent.remember using
per-hand reward attribution, and call train_step when the buffer is ready.
Does not decay epsilon during warm-start.
"""

from __future__ import annotations

from typing import Any

from agents.episode import attribute_hand_transitions
from agents.rule import RuleAgent


def warmstart_from_rule_agent(
    agent: Any,
    game: Any,
    num_episodes: int,
) -> None:
    """Clone the rule policy into ``agent`` for ``num_episodes`` rounds."""
    rule_agent = RuleAgent()

    for _ in range(num_episodes):
        state = game.reset()

        if state is None:
            continue

        transitions = []
        done = False

        while not done:
            hand_index = game.active_hand_index
            available_actions = game.available_actions()
            action = rule_agent.choose_action(state, available_actions)

            next_state, _, done = game.step(action)
            next_available_actions = None if done else game.available_actions()

            transitions.append(
                {
                    "hand_index": hand_index,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "next_available_actions": next_available_actions,
                }
            )
            state = next_state

        attribute_hand_transitions(
            transitions,
            game.hand_rewards,
            agent.remember,
        )

        if len(agent.replay_buffer) >= agent.min_replay_size:
            agent.train_step()

    print(
        f"Warm-start: cloned rule policy for {num_episodes} episodes "
        f"(replay={len(agent.replay_buffer)}, steps={agent.training_steps})"
    )
