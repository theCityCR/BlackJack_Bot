# Project overview

This repository is a **finite-shoe Blackjack RL study**. The primary writeup is [`docs/paper.md`](docs/paper.md); the README is a short storefront.

## What lives where

| Artifact | Purpose |
|---|---|
| [`docs/paper.md`](docs/paper.md) | Research question, methods, legacy results, reproducibility |
| [`docs/results/`](docs/results/) | Published legacy benchmark CSV/JSON/SVG |
| [`config.py`](config.py) | Casino rules + shared neural experimental protocol |
| [`agents/study_protocol.py`](agents/study_protocol.py) | Ablation condition ids and learning-curve schema |
| [`scripts/run_ablation_double_dqn.py`](scripts/run_ablation_double_dqn.py) | Double DQN conditions A–D |
| [`scripts/plot_learning_curves.py`](scripts/plot_learning_curves.py) | Stdlib SVG curves from training CSVs |

## Scale (approximate)

- ~6.5k lines of Python including tests
- Five RL variants + rule baseline
- Four major environment/state redesigns before the current shoe-aware encoding

## Future work

Variable betting / bankroll (path to positive EV vs the house), richer rule charts, configurable penetration, and optional actor-critic methods—after closing the flat-bet gap under the equalized protocol.
