"""Tests for SpreadRuleAgent and variable-bet game wiring."""

from __future__ import annotations

import random

from agents.betting import FlatBetSchedule, TrueCountBetSchedule
from agents.rule import RuleAgent
from agents.spread_rule import SpreadRuleAgent
from cards import Deck
from game import Action, BlackjackGame, ShoeObservation


def _make_episode_game(base_seed: int, episode_index: int) -> BlackjackGame:
    random.seed((int(base_seed) + int(episode_index) * 1_000_003) & 0x7FFFFFFF)
    return BlackjackGame()


class FixedDeck(Deck):
    def __init__(self, cards):
        self.cards = list(cards)
        self.index = 0
        self.num_decks = 2
        self.reshuffle_threshold = 0

    def draw_card(self):
        card = self.cards[self.index]
        self.index += 1
        return card

    def reset(self):
        pass

    def cards_remaining(self):
        return len(self.cards) - self.index

    def get_count_vector(self):
        remaining = self.cards[self.index :]
        counts = [0] * 10
        for card in remaining:
            if card == 1:
                counts[0] += 1
            elif 2 <= card <= 9:
                counts[card - 1] += 1
            else:
                counts[9] += 1
        return tuple(counts)


def test_prepare_round_returns_shoe_before_deal():
    deck = FixedDeck([10, 9, 8, 7, 6, 5])
    game = BlackjackGame(deck)
    shoe = game.prepare_round()
    assert isinstance(shoe, ShoeObservation)
    assert shoe.cards_remaining == 6
    assert game.dealer_hand is None
    assert game.hand_states == []


def test_reset_with_bet_scales_natural_blackjack_payout():
    # Player A,10; dealer 9,7 — natural BJ pays 1.5 × stake.
    deck = FixedDeck([1, 10, 9, 7])
    game = BlackjackGame(deck)
    state = game.reset(bet=2.0)
    assert state is not None
    assert state.player_value == 21
    _, reward, done = game.step(Action.STAND)
    assert done
    assert reward == 3.0


def test_deal_after_prepare_uses_chosen_stake_on_win():
    # Player 20, dealer 17 → win × bet.
    deck = FixedDeck([10, 10, 9, 8])
    game = BlackjackGame(deck)
    game.prepare_round()
    state = game.deal(bet=4.0)
    assert state is not None
    assert game.round_bet == 4.0
    assert game.hand_bets == [4.0]
    _, reward, done = game.step(Action.STAND)
    assert done
    assert reward == 4.0


def test_spread_rule_agent_records_last_bet():
    agent = SpreadRuleAgent(bet_policy=TrueCountBetSchedule())
    game = _make_episode_game(42, 0)
    reward = agent.play_episode(game)
    assert isinstance(reward, float)
    assert agent.last_bet >= 1.0
    assert agent.last_shoe is not None


def test_flat_spread_agent_matches_rule_agent_on_paired_shoes():
    flat_spread = SpreadRuleAgent(bet_policy=FlatBetSchedule(bet=1.0))
    rule = RuleAgent()
    for episode in range(20):
        g1 = _make_episode_game(7, episode)
        g2 = _make_episode_game(7, episode)
        assert flat_spread.play_episode(g1) == rule.play_episode(g2)


def test_persistent_shoe_sometimes_raises_bet_above_unit():
    agent = SpreadRuleAgent(bet_policy=TrueCountBetSchedule())
    game = _make_episode_game(42, 0)
    seen: set[float] = set()
    for _ in range(200):
        agent.play_episode(game)
        seen.add(agent.last_bet)
    assert max(seen) > 1.0
    assert 1.0 in seen
