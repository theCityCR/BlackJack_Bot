# Blackjack Reinforcement Learning Study

Finite-shoe Blackjack as a controlled RL benchmark: does richer shoe composition and deeper DQN beat a basic-strategy baseline under a **fair training protocol**?

**Read the study:** [docs/paper.md](docs/paper.md)

## Result

Under a **shared 200k-episode Double DQN protocol**, hand-only training (−0.0325) beats full shoe-from-scratch (−0.0877); curriculum + warm-start (−0.0513) lands in between. The rule baseline remains ahead (−0.0103). Details: [docs/paper.md](docs/paper.md) §5.2.

**Takeaway:** under equal compute, state design and initialization matter more than architecture depth—hand-only wins the ablation, yet still trails the rule baseline. Ablation table rule EV (−0.0103) is historical; the verified 2-deck S17 DAS chart is −0.0034 under paired eval ([§5.3](docs/paper.md)). Multi-seed mean±std (seeds 42–44): [§5.4](docs/paper.md). Rule + Hi-Lo spread (product path) is **+0.0266 ± 0.0107** EV/round over 100k×3 seeds ([§5.5](docs/paper.md)); deeper reshuffle cuts strengthen that edge ([§5.6](docs/paper.md)). Joint bet+play PG (200k) collapses stakes ([§5.7](docs/paper.md)); bet-focus PG (500k, freeze play) recovers a spread and puts REINFORCE near Hi-Lo ([§5.8](docs/paper.md)).

| Condition | Avg reward |
|---|---:|
| Rule-based baseline (historical §5.2) | **−0.0103** |
| Hand-only (equalized) | −0.0325 |
| Curriculum + warm-start | −0.0513 |
| Curriculum | −0.0654 |
| Full shoe from scratch | −0.0877 |

![Ablation learning curves](docs/results/ablation_learning_curves.svg)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 main.py --episodes 1000 --seed 42
```

Train / ablate / plot (details in the study):

```bash
python3 -m agents.train_double_dqn --seed 42
python3 -m agents.train_a2c --seed 42
python3 scripts/run_ablation_double_dqn.py --smoke
python3 scripts/run_variable_betting_eval.py --smoke
python3 scripts/run_penetration_sweep.py --smoke
python3 scripts/run_pg_spread_bakeoff.py --smoke
python3 scripts/run_pg_spread_bakeoff.py --bet-focus --smoke
python3 evaluate_agents.py --episodes 25000 --seed 42
```

Neural agents run on **CUDA when available, otherwise CPU**. Apple **MPS** is supported via `--device mps` or `BLACKJACK_TORCH_DEVICE=mps`, but for these small DQN batches it is usually slower than CPU. Training logs print `Training device: …` at start.

## Layout

```text
docs/paper.md           # Study writeup (start here)
docs/results/           # Published legacy benchmarks
AGENTS.md               # Map for coding agents
agents/                 # Flat agents + shared neural infra + study protocol
config/                 # rules / protocol / tabular knobs
scripts/                # Ablation runner + learning-curve plots
evaluate_agents.py      # Seeded evaluation
```

## License

[MIT](LICENSE)
