# Does richer shoe state help DQN beat basic strategy?

**A finite-shoe Blackjack study of DQN variants, curricula, and imitation warm-start**

## Abstract

We study flat-bet Blackjack play under a fixed two-deck S17 ruleset with exact remaining-card composition in the observation. Five learning methods (tabular Q-learning through Dueling Double DQN with prioritized replay) are compared to a basic-strategy-inspired rule baseline (~−1.0% EV). Under an earlier unequal training budget, no neural checkpoint beat that baseline. With a shared 200k-episode Double DQN protocol, a **hand-only** ablation (−0.0325) outperforms full shoe-from-scratch (−0.0877); curriculum alone (−0.0654) and curriculum + rule warm-start (−0.0513) sit in between. A longer hand-only gap-close run (100k clone + 500k RL) improves further to about −2.5% EV but still trails the rule agent (~−1.0%). None yet match the rule agent, but the ranking supports the claim that **state design and initialization dominate architecture depth** for this environment.

## 1. Introduction

Blackjack is a short-horizon MDP with partial observability (dealer hole card hidden) and, in a finite shoe, non-stationary odds as cards leave the deck. Basic strategy is a strong hand-feature heuristic for infinite-deck or composition-agnostic play. The natural machine-learning question is whether richer observations—here, **exact remaining rank counts**—plus deeper Q-learning architectures close the gap with that heuristic under flat unit betting.

This repository is framed as a small empirical study, not a casino edge system. We do not claim positive expected value against the house without bet sizing; the headline metric is expected units per round at unit stake.

**Research question.** Under a fixed casino ruleset and equalized training budget, which interventions help DQN-family agents approach the rule baseline: architecture, state curriculum, imitation warm-start, or shoe-feature ablations?

**Hypothesis.** Architecture alone is insufficient; staged state exposure and/or cloning the rule policy matter more than swapping DQN variants.

## 2. Environment

Configured in [`config/`](../config/), implemented in [`game.py`](../game.py) / [`cards.py`](../cards.py):

| Setting | Value |
|---|---|
| Shoe | 2 decks, persists across rounds |
| Penetration | Reshuffle when ≤ 26 cards remain |
| Dealer | Stands on all 17s (S17) |
| Blackjack | Pays 3:2 |
| Double / split | DAS; re-split to 4 hands; no hit on split aces |
| Actions | Hit, Stand, Double, Split |
| Betting | Flat unit stake (no insurance / surrender) |

**State.** Agents observe player total, dealer upcard, usable ace, double/split legality, split-hand context, and a 10-D remaining-rank count vector. Neural agents encode this as a shared **19-D** vector (8 hand features + shoe fraction + 10 normalized counts). Curriculum phase A zeros shoe features so the network can learn hand policy first.

**Reward.** Terminal units won/lost for the round (doubles/splits scale via bet multipliers). Greedy evaluation mean reward is the comparison metric; noisy on-policy training reward is not.

## 3. Methods

### 3.1 Agents

| Agent | Role |
|---|---|
| Rule baseline | Basic-strategy-inspired policy; ignores shoe counts |
| Tabular Q-learning | Discrete `GameState` keys |
| DQN | MLP + replay + target net |
| Double DQN | Double Q targets |
| Dueling Double DQN | Value/advantage streams |
| Dueling Double DQN + PER | Prioritized replay |

Legal-action masking is used throughout neural agents.

### 3.2 Shared experimental protocol

All neural trainers share defaults in `config/protocol.py` (200k episodes, batch 128, ε 1→0.05 with decay 0.99997, target sync every 2k steps, 2 gradient updates/episode). Mid-run learning-curve probes use 500 greedy episodes every 25k training episodes; published comparisons use the 25k final greedy eval. Trainers accept `--no-curriculum` and `--no-warmstart`.

**Curriculum.** Phase A (100k): shoe features off. Phase B: shoe features on; replay cleared at the boundary.

**Warm-start.** Optional behavior cloning from the rule agent (`agents/warmstart.py`) before RL, using the same encoding mode as the upcoming phase A when curriculum is on.

**Learning curves.** Periodic greedy eval rows in `learning_curve.csv`; plot with `scripts/plot_learning_curves.py`.

### 3.3 Ablations (Double DQN)

| ID | Label | Curriculum | Warm-start | Shoe features |
|---|---|---|---|---|
| A | Full from scratch | off | off | on |
| B | Hand-only | off | off | off entire run |
| C | Curriculum | on | off | phase A→B |
| D | Curriculum + warm-start | on | on | phase A→B |

Runner: `scripts/run_ablation_double_dqn.py` (supports `--smoke`).

## 4. Experiments

- **Legacy architecture table:** 25,000 seeded eval rounds (seed 42) on checkpoints trained under unequal budgets (see training-step column).
- **Ablation:** Double DQN conditions A–D under the shared protocol (200k train / 25k eval, seed 42); results in [`docs/results/ablation_results.json`](results/ablation_results.json).

Primary metric: **expected units per round** under flat betting. Win/loss/draw rates use the sign of net round reward.

## 5. Results

### 5.1 Legacy unequal-budget runs (published)

These numbers are **not** from the equalized protocol. They remain useful as a negative result: deeper nets did not beat basic strategy when compute and schedules differed.

| Agent | Avg reward | Win rate | Loss rate | Draw rate | Training steps |
|---|---:|---:|---:|---:|---:|
| Rule-based baseline | **−0.0103** | **43.11%** | **48.13%** | 8.76% | — |
| Double DQN | −0.0594 | 42.00% | 49.52% | 8.48% | 379,264 |
| Dueling Double DQN + PER | −0.0883 | 41.44% | 50.56% | 8.00% | 91,270 |
| Dueling Double DQN | −0.1109 | 40.67% | 51.33% | 8.00% | 182,484 |

![Legacy average reward](results/benchmark_results.svg)

Raw artifacts: [CSV](results/benchmark_results.csv), [JSON](results/benchmark_results.json).

### 5.2 Equalized Double DQN ablations

Double DQN trained for **200,000** episodes per condition under the shared `NEURAL_*` hyperparameters (seed 42). Final greedy evaluation: **25,000** rounds. Rule baseline row is the same seeded eval as §5.1 for reference.

| Condition | Avg reward | Win rate | Loss rate | Draw rate | Training steps |
|---|---:|---:|---:|---:|---:|
| Rule baseline (reference) | **−0.0103** | **43.11%** | **48.13%** | 8.76% | — |
| B. Hand-only | **−0.0325** | 41.98% | 48.71% | 9.31% | 758,860 |
| D. Curriculum + warm-start | −0.0513 | 42.32% | 49.45% | 8.24% | 763,278 |
| C. Curriculum | −0.0654 | 41.76% | 49.92% | 8.32% | 756,260 |
| A. Full from scratch | −0.0877 | 41.20% | 50.16% | 8.64% | 759,120 |

![Equalized ablation learning curves](results/ablation_learning_curves.svg)

Artifacts: [JSON](results/ablation_results.json), per-condition curves under [`results/ablation/`](results/ablation/).

**Takeaway.** With compute held fixed, exposing shoe counts from episode 1 (A) is the weakest setting. Restricting to hand features (B) closes most of the gap toward the rule agent. Curriculum (C) and curriculum + warm-start (D) beat full scratch but do not beat hand-only under this budget—suggesting composition features remain hard to exploit even when staged.

Reproduce:

```bash
python3 scripts/run_ablation_double_dqn.py --seed 42 --episodes 200000 --eval-episodes 25000
python3 scripts/plot_learning_curves.py docs/results/ablation/*/learning_curve.csv \
  --output docs/results/ablation_learning_curves.svg
```

### 5.3 Hand-only gap-close (provisional)

Protocol: true **8-D** hand encoder, **100,000** rule warm-start episodes, **500,000** RL episodes with shoe features off the entire run (`scripts/run_hand_only_gap_close.py`, seed 42). Final greedy evaluation: **25,000** rounds.

| Agent | Avg reward | Training steps |
|---|---:|---:|
| Rule baseline | **−0.0103** | — |
| Hand-only gap-close | −0.0248 | 1,046,535 |
| Gap (agent − rule) | −0.0145 | — |

Relative to equalized condition B (−0.0325), longer cloning + RL closes part of the remaining distance to the rule agent, but flat-bet play is still short of basic-strategy-inspired EV (~−1.0% vs −2.5%).

**Provisional caveats.** These averages come from the training log after the full run completed. The final comparison used the **pre-paired** eval path (agent and rule were not on identical per-episode shoes). The working checkpoint under `agents/results/double_dqn/gap_close/` was later overwritten by a smoke run, so this table is **not** backed by a recoverable model artifact in-tree. Treat the numbers as directional until a paired-eval re-run is published.

Artifact note: [`docs/results/gap_close_results.json`](results/gap_close_results.json). Smoke CI writes to `agents/results/double_dqn/gap_close_smoke/`; full runs refuse to overwrite a prior non-smoke summary unless `--force` is set.

Reproduce (paired eval; replaces the provisional table when finished):

```bash
python3 scripts/run_hand_only_gap_close.py --seed 42
```

## 6. Discussion

Legacy architecture comparisons (§5.1) and the equalized Double DQN ablations (§5.2) both support the hypothesis that **architecture depth is not a substitute for state and training design**. Exact shoe composition enlarges the effective state space; training on full counts from scratch (A) underperforms a hand-only policy (B) by a wide margin. Curriculum and rule warm-start help relative to A, but under a 200k-episode budget the best learned policy remains hand-only—still short of the rule baseline (~−1.0% vs −3.3% EV). The provisional gap-close (§5.3) shows that more imitation and hand-only RL improve further (~−2.5%), yet still do not match the rule agent under this protocol.

**Limitations.** Flat betting only; no insurance/surrender; rule baseline is not a perfect chart; a single seed and one architecture for the ablation table; gap-close §5.3 is provisional (unpaired final eval; checkpoint not retained). Positive EV vs the house would require bet variation (and typically counting), which is future work—not the claim of this study.

**Design history.** The environment went through roughly four major redesigns (splits/doubles → multi-hand rewards → count features → persistent multi-deck shoe). That evolution is part of the experimental story: realism expands the state space faster than naive DQN capacity.

## 7. Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

Train (shared protocol; omit flags to use curriculum + warm-start defaults):

```bash
python3 -m agents.train_double_dqn --seed 42
```

Evaluate checkpoints:

```bash
python3 evaluate_agents.py --episodes 25000 --seed 42
```

Ablations and curves: see §5.2 and `scripts/`. Hand-only gap-close: §5.3 / `scripts/run_hand_only_gap_close.py`.

## License

MIT. See repository root.
