# Design sketch: variable betting

**Status:** design only — not implemented in `game.py`.  
**Gate:** pursue after flat-bet play is near the verified 2-deck S17 DAS rule baseline under paired eval.

## Goal

Positive EV vs the house under this ruleset comes primarily from a **bet spread**, not from better hit/stand alone. Flat unit betting (current study) only shrinks house edge; variable stake is the follow-on product feature.

## Proposed env API

Extend the round loop with a pre-deal bet decision:

1. Observe shoe features (`count_vector`, shoe fraction / penetration).
2. Choose stake `bet ∈ {1, 2, …, B_max}` (or a discrete multiplier schedule, e.g. 1–8).
3. Deal and play with the existing action space (hit / stand / double / split).
4. Terminal reward = units won/lost at that stake (doubles/splits still scale via hand multipliers × stake).

Suggested surface (illustrative):

```python
class BlackjackGame:
    def reset(self, *, bet: float = 1.0) -> GameState | None: ...
    # or
    def choose_bet(self, bet: float) -> None: ...
    def deal(self) -> GameState | None: ...
```

Keep flat-bet mode as `bet=1.0` so published protocols remain reproducible.

## Metrics

| Metric | Definition |
|---|---|
| EV per round | Mean net units / round (primary “beat the house” claim) |
| EV per unit wagered | Mean net / mean stake (isolates play quality from spread) |
| Spread utilization | Fraction of rounds at each bet size vs true count |

Eval: paired per-episode shoes (same opening shoe for compared policies), ≥50k–100k rounds.

## Policies

Two-level is enough for a first demo:

1. **Bet:** map a true-count proxy derived from `count_vector` → stake (even a fixed Hi-Lo-style schedule).
2. **Play:** verified rule chart (optionally composition deviations later).

End-to-end RL for bet+play is optional after the rule+spread baseline shows positive EV.

## Rules / bankroll notes

- Configurable penetration already reshuffles at ≤26 cards; document interaction with counting strength.
- Optional bankroll / risk-of-ruin reporting for portfolio demos — not required for EV tables.
- Insurance / surrender remain out of scope unless the play chart is extended.

## Non-goals (for this design)

- No change to published flat-bet ablation or gap-close tables.
- No actor-critic requirement; start with a scripted count→bet policy.
