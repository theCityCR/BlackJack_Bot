"""
Rule-based Blackjack agent.

Supports:
- Hit
- Stand
- Double
- Split

This is a simple basic-strategy-inspired agent, not a perfect casino-specific
basic strategy chart.
"""

from game import Action, BlackjackGame, GameState


class RuleAgent:
    """
    Rule-based Blackjack agent for the expanded game environment.

    Uses the public GameState from game.py:
    - player_value
    - dealer_upcard
    - usable_ace
    - can_double
    - can_split
    - split-hand metadata
    """

    def choose_action(self, state: GameState, available_actions=None) -> Action:
        """
        Choose one legal action from the current public game state.
        """
        if available_actions is None:
            available_actions = [
                Action.HIT,
                Action.STAND,
                Action.DOUBLE,
                Action.SPLIT,
            ]

        # If only one action is legal, take it.
        if len(available_actions) == 1:
            return available_actions[0]

        # Value 21 should always stand.
        if state.player_value >= 21:
            return Action.STAND

        dealer = state.dealer_upcard
        value = state.player_value

        if Action.SPLIT in available_actions and state.can_split:
            split_action = self._split_decision(state)
            if split_action in available_actions:
                return split_action

        if Action.DOUBLE in available_actions and state.can_double:
            double_action = self._double_decision(state)
            if double_action in available_actions:
                return double_action

        if state.usable_ace:
            action = self._soft_total_decision(value, dealer)
        else:
            action = self._hard_total_decision(value, dealer)

        if action in available_actions:
            return action

        # Safe fallback.
        if Action.STAND in available_actions:
            return Action.STAND

        return available_actions[0]

    def _split_decision(self, state: GameState):
        """
        Basic split rules inferred from player_value.

        Because GameState does not expose the actual pair card, we infer it from
        the hand value when can_split is true.
        """
        value = state.player_value
        dealer = state.dealer_upcard

        # A,A has value 12 with usable ace.
        if state.usable_ace and value == 12:
            return Action.SPLIT

        pair_value = value // 2

        # Always split 8s.
        if pair_value == 8:
            return Action.SPLIT

        # Never split 10s or 5s.
        if pair_value in (10, 5):
            return None

        # Split 2s, 3s, and 7s against dealer 2-7.
        if pair_value in (2, 3, 7) and 2 <= dealer <= 7:
            return Action.SPLIT

        # Split 6s against dealer 2-6.
        if pair_value == 6 and 2 <= dealer <= 6:
            return Action.SPLIT

        # Split 4s only against dealer 5-6.
        if pair_value == 4 and 5 <= dealer <= 6:
            return Action.SPLIT

        # Split 9s against 2-6 and 8-9.
        if pair_value == 9 and (2 <= dealer <= 6 or 8 <= dealer <= 9):
            return Action.SPLIT

        return None

    def _double_decision(self, state: GameState):
        """
        Basic double-down rules.
        """
        value = state.player_value
        dealer = state.dealer_upcard

        if state.usable_ace:
            # Soft 13/14: double vs 5-6.
            if value in (13, 14) and 5 <= dealer <= 6:
                return Action.DOUBLE

            # Soft 15/16: double vs 4-6.
            if value in (15, 16) and 4 <= dealer <= 6:
                return Action.DOUBLE

            # Soft 17: double vs 3-6.
            if value == 17 and 3 <= dealer <= 6:
                return Action.DOUBLE

            # Soft 18: double vs 3-6.
            if value == 18 and 3 <= dealer <= 6:
                return Action.DOUBLE

            return None

        # Hard totals.
        if value == 11:
            return Action.DOUBLE

        if value == 10 and dealer <= 9:
            return Action.DOUBLE

        if value == 9 and 3 <= dealer <= 6:
            return Action.DOUBLE

        return None

    def _soft_total_decision(self, value: int, dealer: int) -> Action:
        """
        Hit/stand rules for soft totals.
        """
        if value <= 17:
            return Action.HIT

        if value == 18:
            if dealer in (9, 10, 1):
                return Action.HIT
            return Action.STAND

        return Action.STAND

    def _hard_total_decision(self, value: int, dealer: int) -> Action:
        """
        Hit/stand rules for hard totals.
        """
        if value <= 11:
            return Action.HIT

        if value == 12:
            if 4 <= dealer <= 6:
                return Action.STAND
            return Action.HIT

        if 13 <= value <= 16:
            if 2 <= dealer <= 6:
                return Action.STAND
            return Action.HIT

        return Action.STAND

    def play_episode(self, game: BlackjackGame, render: bool = False) -> float:
        """
        Play one full game episode.

        Returns the final round reward.
        """
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