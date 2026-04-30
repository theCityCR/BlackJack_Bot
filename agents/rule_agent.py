"""
Rule-based Blackjack agent.

This agent is not learning. It follows a fixed strategy:
- Hit while player value is below the stand threshold
- Stand once player value reaches the threshold
"""

from game import Action, BlackjackGame, GameState


class RuleAgent:
    """
    Simple rule-based agent for Blackjack.

    Default rule:
        hit if player value < 17
        stand otherwise
    """

    def __init__(self, stand_threshold: int = 17):
        self.stand_threshold = stand_threshold

    def choose_action(self, state: GameState) -> Action:
        """
        Choose HIT or STAND from the current public game state.
        """
        if state.player_value < self.stand_threshold:
            return Action.HIT

        return Action.STAND

    def play_episode(self, game: BlackjackGame, render: bool = False) -> int:
        """
        Play one full game episode.

        Returns:
            final reward:
                1  = win
                0  = draw
                -1 = loss
        """
        state = game.reset()
        done = False

        if render:
            game.render()

        while not done:
            action = self.choose_action(state)
            next_state, reward, done = game.step(action)

            if render:
                print(f"Action: {action.name}")
                game.render()
                print()

            state = next_state

        return reward