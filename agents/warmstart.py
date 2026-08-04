"""Behavior cloning warm-start from the rule-based baseline.

Contract (Wave 1):
    warmstart_from_rule_agent(agent, game, num_episodes) -> None

- Drive episodes with RuleAgent.choose_action (legal actions only).
- Store transitions via agent.remember(...) using the same hand-reward
  attribution pattern as train_one_episode.
- Call agent.train_step() when the replay buffer is large enough.
- Do not decay epsilon during warm-start; leave epsilon at its start value
  for the subsequent RL phase (or set epsilon to epsilon_min after cloning
  only if documented—prefer leaving epsilon high and filling the buffer).
- Respect agent.use_shoe_features when encoding (caller sets hand-only when
  curriculum is enabled).
- Print a one-line summary when finished.
"""

from __future__ import annotations

from typing import Any

from agents.rule_agent.rule_agent import RuleAgent


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
                    "done": done,
                    "next_available_actions": next_available_actions,
                }
            )

            state = next_state

        hand_rewards = game.hand_rewards

        for i, transition in enumerate(transitions):
            hand_index = transition["hand_index"]
            hand_reward = hand_rewards[hand_index]

            is_last = i == len(transitions) - 1
            next_is_different_hand = (
                not is_last
                and transitions[i + 1]["hand_index"] != hand_index
            )

            terminal_for_this_hand = is_last or next_is_different_hand

            if terminal_for_this_hand:
                agent.remember(
                    transition["state"],
                    transition["action"],
                    hand_reward,
                    None,
                    True,
                    None,
                )
            else:
                agent.remember(
                    transition["state"],
                    transition["action"],
                    0.0,
                    transition["next_state"],
                    False,
                    transition["next_available_actions"],
                )

        if len(agent.replay_buffer) >= agent.min_replay_size:
            agent.train_step()

    print(
        f"Warm-start: cloned rule policy for {num_episodes} episodes "
        f"(replay={len(agent.replay_buffer)}, steps={agent.training_steps})"
    )
