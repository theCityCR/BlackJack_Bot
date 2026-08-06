"""
Rule-based Blackjack agent (2-deck S17 DAS basic strategy).

Total-dependent chart aligned with Wizard-of-Odds-style double-deck strategy
where the dealer stands on soft 17 and doubling after split is allowed.
No insurance or surrender (matches the environment).

Dealer Ace is encoded as ``dealer_upcard == 1``.
"""

from __future__ import annotations

from game import Action, BlackjackGame, GameState

# Dealer upcards in chart column order: 2-10, Ace.
_DEALERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 1)

# Cell codes: H hit, S stand, Dh double-or-hit, Ds double-or-stand, P split.
_H = "H"
_S = "S"
_DH = "Dh"
_DS = "Ds"
_P = "P"


def _row(*cells: str) -> dict[int, str]:
    if len(cells) != len(_DEALERS):
        raise ValueError("chart row must have 10 dealer columns")
    return dict(zip(_DEALERS, cells))


# Hard totals 5–21 (totals below 5 do not occur as playable decisions here).
_HARD: dict[int, dict[int, str]] = {
    5: _row(_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    6: _row(_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    7: _row(_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    8: _row(_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    # 2-deck: double 9 vs 2-6 (includes dealer 2).
    9: _row(_DH, _DH, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    # Double 10 vs 2-9 only — never vs 10 or Ace.
    10: _row(_DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _H, _H),
    11: _row(_DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH),
    12: _row(_H, _H, _S, _S, _S, _H, _H, _H, _H, _H),
    13: _row(_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    14: _row(_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    15: _row(_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    16: _row(_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    17: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    18: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    19: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    20: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    21: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
}

# Soft totals (A,2)=13 … (A,9)=20.
_SOFT: dict[int, dict[int, str]] = {
    13: _row(_H, _H, _H, _DH, _DH, _H, _H, _H, _H, _H),
    14: _row(_H, _H, _H, _DH, _DH, _H, _H, _H, _H, _H),
    15: _row(_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    16: _row(_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    17: _row(_H, _DH, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    # 2-deck: double soft 18 vs 2-6; stand 7-8; hit 9-10-A.
    18: _row(_DS, _DS, _DS, _DS, _DS, _S, _S, _H, _H, _H),
    # 2-deck: double soft 19 vs 6 only.
    19: _row(_S, _S, _S, _S, _DS, _S, _S, _S, _S, _S),
    20: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    21: _row(_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
}

# Pair rank -> dealer cell. 5s and 10s never split (looked up as hard totals).
_PAIR: dict[int, dict[int, str]] = {
    1: _row(_P, _P, _P, _P, _P, _P, _P, _P, _P, _P),  # A,A
    2: _row(_P, _P, _P, _P, _P, _P, _H, _H, _H, _H),
    3: _row(_P, _P, _P, _P, _P, _P, _H, _H, _H, _H),
    4: _row(_H, _H, _H, _P, _P, _H, _H, _H, _H, _H),
    6: _row(_P, _P, _P, _P, _P, _P, _H, _H, _H, _H),
    7: _row(_P, _P, _P, _P, _P, _P, _H, _H, _H, _H),
    8: _row(_P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    9: _row(_P, _P, _P, _P, _P, _S, _P, _P, _S, _S),
}


def _resolve_cell(code: str, available: list[Action]) -> Action | None:
    """Map a chart cell to a legal action, or None if the cell is not actionable."""
    if code == _P:
        return Action.SPLIT if Action.SPLIT in available else None
    if code == _DH:
        if Action.DOUBLE in available:
            return Action.DOUBLE
        return Action.HIT if Action.HIT in available else None
    if code == _DS:
        if Action.DOUBLE in available:
            return Action.DOUBLE
        return Action.STAND if Action.STAND in available else None
    if code == _H:
        return Action.HIT if Action.HIT in available else None
    if code == _S:
        return Action.STAND if Action.STAND in available else None
    return None


class RuleAgent:
    """
    Verified 2-deck S17 DAS total-dependent basic strategy.

    Uses the public GameState from game.py. Pair rank is inferred from
    ``player_value`` when ``can_split`` is true (A,A special-cased as soft 12).
    """

    def choose_action(self, state: GameState, available_actions=None) -> Action:
        """Choose one legal action from the current public game state."""
        if available_actions is None:
            available_actions = [Action.HIT, Action.STAND]
            if state.can_double:
                available_actions.append(Action.DOUBLE)
            if state.can_split:
                available_actions.append(Action.SPLIT)
        else:
            available_actions = list(available_actions)

        if len(available_actions) == 1:
            return available_actions[0]

        if state.player_value >= 21:
            return Action.STAND

        dealer = state.dealer_upcard
        value = state.player_value

        if Action.SPLIT in available_actions and state.can_split:
            pair_rank = self._pair_rank(state)
            if pair_rank is not None and pair_rank in _PAIR:
                code = _PAIR[pair_rank].get(dealer, _H)
                action = _resolve_cell(code, available_actions)
                if action is not None:
                    return action
            # 5s / 10s fall through to hard-total chart.

        if state.usable_ace and value in _SOFT:
            code = _SOFT[value].get(dealer, _H)
        else:
            hard_value = min(max(value, 5), 21)
            code = _HARD[hard_value].get(dealer, _H)

        action = _resolve_cell(code, available_actions)
        if action is not None and action in available_actions:
            return action

        if Action.STAND in available_actions:
            return Action.STAND
        return available_actions[0]

    @staticmethod
    def _pair_rank(state: GameState) -> int | None:
        """Infer pair rank from total when splitting is legal."""
        value = state.player_value
        if state.usable_ace and value == 12:
            return 1  # A,A
        if value % 2 != 0:
            return None
        rank = value // 2
        if rank in (5, 10):
            return None  # never split; use hard chart
        return rank

    def play_episode(self, game: BlackjackGame, render: bool = False) -> float:
        """Play one full game episode. Returns the final round reward."""
        state = game.reset()

        if state is None:
            if render:
                game.render()
            return game.round_reward

        done = False

        if render:
            game.render()

        while not done:
            available_actions = game.available_actions()
            action = self.choose_action(state, available_actions)
            next_state, reward, done = game.step(action)

            if render:
                print(f"Action: {action.name}")
                game.render()
                print()

            state = next_state

        return reward
