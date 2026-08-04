# Does richer shoe state help DQN beat basic strategy?

**A finite-shoe Blackjack study of DQN variants, curricula, and imitation warm-start**

## Abstract

We study flat-bet Blackjack play under a fixed two-deck S17 ruleset with exact remaining-card composition in the observation. Five learning methods (tabular Q-learning through Dueling Double DQN with prioritized replay) are compared to a basic-strategy-inspired rule baseline. Under an earlier unequal training budget, no neural checkpoint beat the rule agent (~−1.0% EV). We then fix a shared experimental protocol—equal episodes and optimizer schedules, a hand-then-shoe state curriculum, optional rule-policy warm-start, learning-curve logging, and a Double DQN ablation suite—so architecture and state-design effects can be isolated. Published legacy numbers remain below the baseline; equalized and ablation results are produced by the tooling in this repository and should be filled in after a full retrain.

## 1. Introduction

Blackjack is a short-horizon MDP with partial observability (dealer hole card hidden) and, in a finite shoe, non-stationary odds as cards leave the deck. Basic strategy is a strong hand-feature heuristic for infinite-deck or composition-agnostic play. The natural machine-learning question is whether richer observations—here, **exact remaining rank counts**—plus deeper Q-learning architectures close the gap with that heuristic under flat unit betting.

This repository is framed as a small empirical study, not a casino edge system. We do not claim positive expected value against the house without bet sizing; the headline metric is expected units per round at unit stake.

**Research question.** Under a fixed casino ruleset and equalized training budget, which interventions help DQN-family agents approach the rule baseline: architecture, state curriculum, imitation warm-start, or shoe-feature ablations?

**Hypothesis.** Architecture alone is insufficient; staged state exposure and/or cloning the rule policy matter more than swapping DQN variants.

## 2. Environment

Configured in [`config.py`](../config.py), implemented in [`game.py`](../game.py) / [`cards.py`](../cards.py):

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

All neural trainers share defaults in `config.py` (200k episodes, batch 128, ε 1→0.05 with decay 0.99997, target sync every 2k steps, 4 gradient updates/episode). Trainers accept `--no-curriculum` and `--no-warmstart`.

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
- **Equalized protocol:** retrain neural agents with shared defaults; compare greedy `average_reward` and learning curves.
- **Ablation:** Double DQN conditions A–D; machine-readable JSON under `agents/double_q_network_learning/results/ablation/`.

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

### 5.2 Equalized protocol and ablations

*Pending full retrain.* After running the shared trainers and/or:

```bash
python3 scripts/run_ablation_double_dqn.py --seed 42
python3 scripts/plot_learning_curves.py agents/*/*/learning_curve.csv --output docs/results/learning_curves.svg
```

replace this subsection with the new table and curve figure. Smoke validation:

```bash
python3 scripts/run_ablation_double_dqn.py --smoke --conditions A_full_scratch
```

## 6. Discussion

The legacy result supports the hypothesis that **architecture depth is not a substitute for state and training design**. Exact shoe composition enlarges the effective state space; without curriculum or a strong prior (rule warm-start), networks can underfit relative to a compact heuristic.

**Limitations.** Flat betting only; no insurance/surrender; rule baseline is not a perfect chart; published neural numbers use an obsolete budget. Positive EV vs the house would require bet variation (and typically counting), which is future work—not the claim of this study.

**Design history.** The environment went through roughly four major redesigns (splits/doubles → multi-hand rewards → count features → persistent multi-deck shoe). That evolution is part of the experimental story: realism expands the state space faster than naive DQN capacity.

## 7. Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

Train (shared protocol; omit flags to use curriculum + warm-start defaults):

```bash
python3 -m agents.double_q_network_learning.train_double_q_network_learning_agent --seed 42
```

Evaluate checkpoints:

```bash
python3 evaluate_agents.py --episodes 25000 --seed 42
```

Ablations and curves: see §5.2 and `scripts/`.

## License

MIT. See repository root.
