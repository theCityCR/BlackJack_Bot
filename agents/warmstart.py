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


def warmstart_from_rule_agent(
    agent: Any,
    game: Any,
    num_episodes: int,
) -> None:
    """Clone the rule policy into ``agent`` for ``num_episodes`` rounds."""
    raise NotImplementedError(
        "Wave 1: implement rule-agent behavior cloning in agents/warmstart.py"
    )
