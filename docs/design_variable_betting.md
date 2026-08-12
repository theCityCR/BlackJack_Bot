# Design sketch: variable betting

**Status:** implemented — `BlackjackGame.prepare_round` / `deal(bet=…)` / `reset(bet=1.0)`, Hi-Lo schedule in `agents/betting.py`, `SpreadRuleAgent`, eval via `scripts/run_variable_betting_eval.py`.  
**Gate (original):** pursue after flat-bet play is near the verified 2-deck S17 DAS rule baseline under paired eval. **Waived** for this product path: positive EV vs the house comes from bet spread, not flat-bet DQN parity.

## Goal

Positive EV vs the house under this ruleset comes primarily from a **bet spread**, not from better hit/stand alone. Flat unit betting (current study) only shrinks house edge; variable stake is the follow-on product feature.

## Env API

Pre-deal bet decision:

1. `prepare_round()` → `ShoeObservation` (`count_vector`, `cards_remaining`) after reshuffle check.
2. Choose stake `bet ∈ {1, 2, …, B_max}` (default schedule 1–8).
3. `deal(bet=…)` then play with the existing action space (hit / stand / double / split).
4. Terminal reward = units won/lost at that stake (doubles/splits still scale via hand multipliers × stake).

`reset(bet=1.0)` remains the flat-bet entry point so published protocols stay reproducible.

## Metrics

| Metric | Definition |
|---|---|
| EV per round | Mean net units / round (primary “beat the house” claim) |
| EV per unit wagered | Mean net / mean stake (isolates play quality from spread) |
| Spread utilization | Fraction of rounds at each bet size vs true count |

Eval: consecutive rounds on paired seeded shoes (penetration required for
counting), ≥50k–100k rounds:

```bash
python3 scripts/run_variable_betting_eval.py --episodes 50000 --seed 42
python3 scripts/run_variable_betting_eval.py --smoke
```

## Policies

Two-level demo (shipped):

1. **Bet:** Hi-Lo true count from `count_vector` → `TrueCountBetSchedule` (TC≤0→1, 1→2, 2→4, 3→6, ≥4→8).
2. **Play:** verified rule chart (`RuleAgent`).

End-to-end RL for bet+play remains optional after the rule+spread baseline shows positive EV.

## Rules / bankroll notes

- Penetration reshuffles at ≤26 cards; counting strength is limited by that cut.
- Optional bankroll / risk-of-ruin reporting for portfolio demos — not required for EV tables.
- Insurance / surrender remain out of scope unless the play chart is extended.

## Non-goals

- No change to published flat-bet ablation or gap-close tables under `docs/results/`.
- No actor-critic requirement for the first demo.
