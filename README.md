# Blackjack Reinforcement Learning Bot

A finite-shoe Blackjack environment for studying how reinforcement learning methods behave as game rules and state representations become more realistic.

The project implements five RL variants—from tabular Q-learning through Dueling Double DQN with prioritized replay—alongside a basic-strategy-inspired baseline. The custom environment supports card-count features, splits, doubles, multiple player hands, and natural Blackjack payouts.

## Highlights

- **Five RL variants** implemented directly in Python and PyTorch
- **550,000+ configured training episodes** across the core experiments
- **Persistent multi-deck shoe** with exact remaining-card composition
- **194+ automated tests** for game rules, state transitions, and agent behavior
- **Reproducible evaluation** with seeded runs and machine-readable output

## Results

The available trained checkpoints and rule baseline were evaluated for 25,000 rounds each using seed 42 and the same game configuration.

| Agent | Average reward | Win rate | Loss rate | Draw rate |
|---|---:|---:|---:|---:|
| Rule-based baseline | **-0.0103** | **43.11%** | **48.13%** | 8.76% |
| Double DQN | -0.0594 | 42.00% | 49.52% | 8.48% |
| Dueling Double DQN + PER | -0.0883 | 41.44% | 50.56% | 8.00% |
| Dueling DQN | -0.1109 | 40.67% | 51.33% | 8.00% |

![Average reward comparison](docs/results/benchmark_results.svg)

The rule-based agent outperformed the saved neural checkpoints. This is a useful negative result: expanding the state space with exact shoe composition and multi-hand context made learning harder, and additional network complexity alone did not guarantee a stronger policy. The comparison motivated closer attention to state design, reward attribution, training duration, and evaluation methodology.

The raw [CSV](docs/results/benchmark_results.csv) and [JSON](docs/results/benchmark_results.json) results are versioned with the project. Model checkpoints are intentionally excluded because of their size.

## Environment

Each agent observes a `GameState` containing:

- Player total and usable-ace status
- Dealer upcard
- Legal double and split flags
- Active-hand and split-hand context
- Counts of all remaining card values in the shoe

Available actions are **hit**, **stand**, **double**, and **split**. The default shoe contains two complete decks, persists between rounds, and reshuffles at a configurable cut-card threshold.

## Algorithms

1. Tabular Q-learning
2. Deep Q-Network (DQN)
3. Double DQN
4. Dueling DQN
5. Dueling Double DQN with prioritized experience replay

A separate rule-based agent provides a non-learning benchmark. Neural agents use legal-action masking, replay buffers, target networks, and normalized state features.

## Quick Start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

Run the lightweight rule-based benchmark:

```bash
python3 main.py --episodes 1000 --seed 42
```

Evaluate every locally available trained checkpoint and regenerate the results:

```bash
python3 evaluate_agents.py --episodes 25000 --seed 42
```

The evaluator skips checkpoints that are not present and always evaluates the rule baseline.

## Project Structure

```text
├── agents/                 # Rule, tabular, and neural agents
├── docs/results/           # Reproducible benchmark outputs
├── tests/                  # Environment and agent tests
├── cards.py                # Finite shoe and hand abstractions
├── game.py                 # Blackjack rules and state transitions
├── config.py               # Rules and training configuration
├── evaluate_agents.py      # Unified seeded evaluation
└── main.py                 # Rule-agent benchmark entry point
```

## Project Scale

- Approximately **6,470 lines of Python**, including tests
- Approximately **3,500 lines** of environment, agent, training, and evaluation code
- Approximately **2,970 lines** of tests
- Approximately **four major environment/state redesigns** after the initial prototype

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for a longer case-study description.

## License

Released under the [MIT License](LICENSE).
