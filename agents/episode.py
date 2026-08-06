"""Per-hand transition attribution for split-aware training."""

from __future__ import annotations

from typing import Any, Callable


def attribute_hand_transitions(
    transitions: list[dict[str, Any]],
    hand_rewards: list[float | None],
    remember: Callable[..., None],
) -> None:
    """Store transitions with per-hand terminal rewards (not round totals).

    Each entry in ``transitions`` must include ``hand_index``, ``state``,
    ``action``, ``next_state``, and ``next_available_actions``.
    """
    for index, transition in enumerate(transitions):
        hand_index = transition["hand_index"]
        hand_reward = hand_rewards[hand_index]

        is_last = index == len(transitions) - 1
        next_is_new_hand = (
            not is_last
            and transitions[index + 1]["hand_index"] != hand_index
        )
        terminal_for_this_hand = is_last or next_is_new_hand

        if terminal_for_this_hand:
            remember(
                transition["state"],
                transition["action"],
                hand_reward,
                None,
                True,
                None,
            )
        else:
            remember(
                transition["state"],
                transition["action"],
                0.0,
                transition["next_state"],
                False,
                transition["next_available_actions"],
            )
