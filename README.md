# Blackjack Reinforcement Learning Study

Finite-shoe Blackjack as a controlled RL benchmark: does richer shoe composition and deeper DQN beat a basic-strategy baseline under a **fair training protocol**?

**Read the study:** [docs/paper.md](docs/paper.md)

## Result (legacy preview)

Under an earlier unequal training budget, the rule baseline (~**−1.0%** EV) beat all published neural checkpoints. Equalized curriculum / warm-start / ablation tooling is in-repo; retrain to refresh the table.

| Agent | Avg reward |
|---|---:|
| Rule-based baseline | **−0.0103** |
| Double DQN (legacy) | −0.0594 |
| Dueling + PER (legacy) | −0.0883 |
| Dueling DQN (legacy) | −0.1109 |

![Legacy comparison](docs/results/benchmark_results.svg)

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
python3 -m agents.double_q_network_learning.train_double_q_network_learning_agent --seed 42
python3 scripts/run_ablation_double_dqn.py --smoke
python3 evaluate_agents.py --episodes 25000 --seed 42
```

## Layout

```text
docs/paper.md           # Study writeup (start here)
docs/results/           # Published legacy benchmarks
agents/                 # Rule, tabular, and neural agents + study protocol
scripts/                # Ablation runner + learning-curve plots
config.py               # Rules and shared experimental protocol
evaluate_agents.py      # Seeded evaluation
```

## License

[MIT](LICENSE)
