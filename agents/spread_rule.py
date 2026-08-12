"""Rule chart play with a scripted true-count bet spread."""

from __future__ import annotations

from agents.betting import FlatBetSchedule, TrueCountBetSchedule
from agents.counting import true_count_from_shoe
from agents.rule import RuleAgent
from config import NUM_DECKS
from game import Action, BlackjackGame, GameState, ShoeObservation


class SpreadRuleAgent:
    """Two-level policy: Hi-Lo stake, then verified basic-strategy play."""

    def __init__(
        self,
        *,
        play_agent: RuleAgent | None = None,
        bet_policy: TrueCountBetSchedule | FlatBetSchedule | None = None,
        num_decks: int = NUM_DECKS,
    ):
        self.play_agent = play_agent if play_agent is not None else RuleAgent()
        self.bet_policy = (
            bet_policy if bet_policy is not None else TrueCountBetSchedule()
        )
        self.num_decks = num_decks
        self.last_bet: float = 1.0
        self.last_true_count: float = 0.0
        self.last_shoe: ShoeObservation | None = None

    def choose_action(
        self, state: GameState, available_actions: list[Action]
    ) -> Action:
        return self.play_agent.choose_action(state, available_actions)

    def choose_bet(self, shoe: ShoeObservation) -> float:
        self.last_shoe = shoe
        self.last_true_count = true_count_from_shoe(shoe, num_decks=self.num_decks)
        self.last_bet = float(self.bet_policy.choose_bet(shoe))
        return self.last_bet

    def play_episode(self, game: BlackjackGame, render: bool = False) -> float:
        """Bet from the prepared shoe, then play one round with the rule chart."""
        shoe = game.prepare_round()
        bet = self.choose_bet(shoe)
        state = game.deal(bet=bet)

        if state is None:
            if render:
                game.render()
            return game.round_reward

        done = False
        if render:
            game.render()

        reward = 0.0
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
