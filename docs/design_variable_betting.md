# Design sketch: variable betting

**Status:** implemented — `BlackjackGame.prepare_round` / `deal(bet=…)` / `reset(bet=1.0)`, Hi-Lo schedule in `agents/betting.py`, `SpreadRuleAgent`, eval via `scripts/run_variable_betting_eval.py`, optional bankroll/RoR via `--bankroll` (`agents/bankroll.py`), and bet+play PG agents (REINFORCE / A2C / PPO) via `agents/train_reinforce|a2c|ppo`.  
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
| Bankroll path | Ending / min bankroll and max drawdown on one continuous walk |
| Trip risk of ruin | Fraction of independent trips that cannot cover the next stake (or hit ≤0) |

Eval: consecutive rounds on paired seeded shoes (penetration required for
counting), ≥50k–100k rounds. Multi-seed published aggregate (100k × seeds 42–44):
[`docs/results/multi_seed_variable_betting_results.json`](results/multi_seed_variable_betting_results.json).
Penetration cut sweep (§5.6, 100k × seed 42):
[`docs/results/penetration_sweep_results.json`](results/penetration_sweep_results.json).

```bash
python3 scripts/run_variable_betting_eval.py --episodes 50000 --seed 42
python3 scripts/run_variable_betting_eval.py --episodes 100000 --seeds 42,43,44 \
  --output docs/results/multi_seed_variable_betting_results.json
python3 scripts/run_variable_betting_eval.py --smoke
# Portfolio / RoR demo (default start 200 units; trips default to --rounds-per-shoe):
python3 scripts/run_variable_betting_eval.py --episodes 50000 --seed 42 --bankroll
python3 scripts/run_variable_betting_eval.py --episodes 100000 --seeds 42,43,44 \
  --bankroll 200 --trip-rounds 100
# Penetration / cut-card sweep (counting edge vs dealt fraction):
python3 scripts/run_penetration_sweep.py --smoke
# Published §5.6 protocol:
python3 scripts/run_penetration_sweep.py --episodes 100000 --seed 42 \
  --thresholds 13,26,39,52 \
  --output docs/results/penetration_sweep_results.json
```

Ruin rule: before each round, if cash `<` the scheduled stake, the path/trip stops
as ruined. After settlement, bankroll `≤ 0` also counts as ruin.
## Policies

Two-level demo (shipped):

1. **Bet:** Hi-Lo true count from `count_vector` → `TrueCountBetSchedule` (TC≤0→1, 1→2, 2→4, 3→6, ≥4→8).
2. **Play:** verified rule chart (`RuleAgent`).

End-to-end RL for bet+play (shipped): REINFORCE / A2C / PPO under `agents/reinforce.py`, `a2c.py`, `ppo.py` with shared `agents/policy_base.py`. Discrete stakes `{1..BET_MAX}` then masked play actions. Warm-start clones SpreadRule via CE (`agents/pg_warmstart.py`). Published bake-off vs rule+Hi-Lo (§5.7): [`docs/results/pg_spread_bakeoff_results.json`](results/pg_spread_bakeoff_results.json).

```bash
python3 -m agents.train_a2c --episodes 1000 --seed 42 --no-warmstart
python3 scripts/run_variable_betting_eval.py --smoke \
  --pg-agent a2c --pg-checkpoint agents/results/a2c/a2c_bet_play_model.pt
python3 scripts/run_pg_spread_bakeoff.py --smoke
# Published §5.7 protocol (after full 200k trains):
python3 scripts/run_pg_spread_bakeoff.py --episodes 100000 --seeds 42,43,44 \
  --output docs/results/pg_spread_bakeoff_results.json
```

## Rules / bankroll notes

- Penetration reshuffles when remaining cards ≤ the cut (`RESHUFFLE_WHEN_CARDS_REMAINING_BELOW`, default 26 ≈ 75% dealt). Override with `--reshuffle-threshold`; published cut sweep is §5.6 / `scripts/run_penetration_sweep.py`.
- Bankroll / risk-of-ruin reporting: `agents/bankroll.py` via `--bankroll` on
  `scripts/run_variable_betting_eval.py` (optional; not part of the published §5.5 EV table).
- Insurance / surrender remain out of scope unless the play chart is extended.

## Non-goals

- No change to published flat-bet ablation or gap-close tables under `docs/results/`.
- PG checkpoints live under `agents/results/<reinforce|a2c|ppo>/` (gitignored); the §5.7 aggregate JSON is the published claim.
