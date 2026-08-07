"""Paired per-episode shoe evaluation."""

from __future__ import annotations

from agents.common import episode_rng_seed, evaluate_greedy, make_episode_game
from agents.rule import RuleAgent
from cards import Deck
from game import Action, BlackjackGame


class AlwaysHitAgent:
    """Minimal policy that prefers HIT so it diverges from basic strategy."""

    def play_episode(self, game: BlackjackGame) -> float:
        state = game.reset()
        if state is None:
            return game.round_reward

        reward = 0.0
        done = False
        while not done:
            available = game.available_actions()
            action = Action.HIT if Action.HIT in available else Action.STAND
            _, reward, done = game.step(action)
        return reward


def _track_force_resets(monkeypatch) -> list[tuple[int, ...]]:
    shoes: list[tuple[int, ...]] = []
    original = Deck.force_reset

    def tracking_reset(self):
        original(self)
        shoes.append(tuple(self.cards))

    monkeypatch.setattr(Deck, "force_reset", tracking_reset)
    return shoes


def test_episode_rng_seed_is_stable_and_varies():
    assert episode_rng_seed(42, 0) == episode_rng_seed(42, 0)
    assert episode_rng_seed(42, 0) != episode_rng_seed(42, 1)
    assert episode_rng_seed(1, 0) != episode_rng_seed(2, 0)


def test_make_episode_game_reproduces_shoe_order():
    first = make_episode_game(7, 3)
    second = make_episode_game(7, 3)
    assert first.deck.cards == second.deck.cards
    assert make_episode_game(7, 4).deck.cards != first.deck.cards


def test_evaluate_greedy_with_seed_is_reproducible():
    first = evaluate_greedy(RuleAgent(), 80, seed=11)
    second = evaluate_greedy(RuleAgent(), 80, seed=11)
    assert first == second


def test_evaluate_greedy_pairs_shoes_across_different_agents(monkeypatch):
    """Distinct policies under the same seed see the same opening shoe each episode."""
    shoes = _track_force_resets(monkeypatch)

    evaluate_greedy(RuleAgent(), 12, seed=99)
    rule_shoes = list(shoes)
    shoes.clear()
    evaluate_greedy(AlwaysHitAgent(), 12, seed=99)

    assert shoes == rule_shoes
    assert len(rule_shoes) == 12


def test_evaluate_greedy_without_seed_reuses_persistent_game():
    """Mid-run probes (seed=None) reuse one BlackjackGame across episodes."""
    game_ids: list[int] = []

    class TrackingAgent(RuleAgent):
        def play_episode(self, game, render: bool = False) -> float:
            game_ids.append(id(game))
            return super().play_episode(game, render=render)

    evaluate_greedy(TrackingAgent(), 6, seed=None)
    assert len(game_ids) == 6
    assert len(set(game_ids)) == 1


def test_evaluate_greedy_with_seed_uses_fresh_shoe_per_episode(monkeypatch):
    """Paired eval builds a new opening shoe for each episode index.

    Do not assert unique ``id(game)``: CPython may recycle object ids after GC.
    """
    shoes = _track_force_resets(monkeypatch)
    evaluate_greedy(RuleAgent(), 6, seed=5)
    assert len(shoes) == 6
    assert len(set(shoes)) == 6
