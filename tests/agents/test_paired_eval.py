"""Paired per-episode shoe evaluation."""

from __future__ import annotations

from agents.common import episode_rng_seed, evaluate_greedy, make_episode_game
from agents.rule import RuleAgent
from cards import Deck


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


def test_evaluate_greedy_pairs_shoes_across_agents(monkeypatch):
    """Two agents with the same seed see the same opening shoe each episode."""
    shoes: list[tuple[int, ...]] = []
    original = Deck.force_reset

    def tracking_reset(self):
        original(self)
        shoes.append(tuple(self.cards))

    monkeypatch.setattr(Deck, "force_reset", tracking_reset)

    evaluate_greedy(RuleAgent(), 12, seed=99)
    first_run = list(shoes)
    shoes.clear()
    evaluate_greedy(RuleAgent(), 12, seed=99)
    assert shoes == first_run
    # One fresh shoe shuffle per episode (BlackjackGame / make_episode_game).
    assert len(first_run) == 12
