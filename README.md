# Blackjack Reinforcement Learning Bot

A finite-shoe Blackjack environment for studying how reinforcement learning methods behave as game rules and state representations become more realistic.

The project implements five RL variants—from tabular Q-learning through Dueling Double DQN with prioritized replay—alongside a basic-strategy-inspired baseline. The custom environment supports card-count features, splits, doubles, multiple player hands, and natural Blackjack payouts.

## Highlights

- **Five RL variants** implemented directly in Python and PyTorch
- **800,000+ configured neural training episodes** (4 × 200k under a shared protocol)
- **Shared neural training budget** — same episodes, ε schedule, batch size, and updates/episode
- **Two-phase curriculum** — hand features first, then shoe-aware counts
- **Persistent multi-deck shoe** with exact remaining-card composition
- **Shared 19-feature state encoding** for all neural agents (shoe-aware)
- **200 automated tests** for game rules, state transitions, and agent behavior
- **Reproducible evaluation** with seeded runs and machine-readable output

**Takeaway:** a stronger network does not automatically beat a simpler policy when the state space grows—careful state and reward design matter more than architecture complexity alone.

## Results

The available trained checkpoints and rule baseline were evaluated for 25,000 rounds each using seed 42 and the same game configuration. Win/loss/draw rates use the sign of the **net round reward** (so doubles and splits collapse to one outcome per round).

| Agent | Avg reward | Win rate | Loss rate | Draw rate | Training steps |
|---|---:|---:|---:|---:|---:|
| Rule-based baseline | **-0.0103** | **43.11%** | **48.13%** | 8.76% | — |
| Double DQN | -0.0594 | 42.00% | 49.52% | 8.48% | 379,264 |
| Dueling Double DQN + PER | -0.0883 | 41.44% | 50.56% | 8.00% | 91,270 |
| Dueling Double DQN | -0.1109 | 40.67% | 51.33% | 8.00% | 182,484 |

![Average reward comparison](docs/results/benchmark_results.svg)

The rule-based agent outperformed the saved neural checkpoints. This is a useful negative result: expanding the state space with exact shoe composition and multi-hand context made learning harder, and additional network complexity alone did not guarantee a stronger policy.

Those published checkpoints were trained under an earlier **unequal** protocol (see training steps). Neural trainers now share one budget and hyperparameter schedule in [`config.py`](config.py) so future runs isolate architecture differences. Retrain and re-evaluate to refresh the table.

Vanilla DQN and tabular Q-learning are implemented and trainable, but were not part of the published checkpoint set above. The evaluator includes them automatically when local artifacts are present.

The raw [CSV](docs/results/benchmark_results.csv) and [JSON](docs/results/benchmark_results.json) results are versioned with the project. Model checkpoints are intentionally excluded because of their size.

## Ruleset

Default casino rules in [`config.py`](config.py):

- 2-deck shoe that persists between rounds
- Reshuffle when fewer than 26 cards remain
- Dealer stands on all 17s (S17), including soft 17
- Double after split (DAS) and re-splits allowed (up to 4 hands)
- No hitting split aces

## Environment

Each agent observes a `GameState` containing:

- Player total and usable-ace status
- Dealer upcard
- Legal double and split flags
- Active-hand and split-hand context
- Counts of all remaining card values in the shoe

Neural agents encode this into a shared **19-dimensional** vector (hand features + shoe fraction + normalized count vector). An earlier DQN prototype used 8 features without shoe counts; all neural agents now share the shoe-aware encoding. Training defaults to a **two-phase curriculum**: phase A zeros shoe features so the agent learns hand policy first, then phase B enables the full count vector (replay is cleared at the boundary). Pass `--no-curriculum` to train shoe-aware from episode 1.

Available actions are **hit**, **stand**, **double**, and **split**.

## Algorithms

1. Tabular Q-learning
2. Deep Q-Network (DQN)
3. Double DQN
4. Dueling Double DQN
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

Evaluate locally available trained checkpoints (writes to `results/eval/` by default and does **not** overwrite published docs):

```bash
python3 evaluate_agents.py --episodes 25000 --seed 42
```

To republish portfolio artifacts after training the published neural set:

```bash
python3 evaluate_agents.py --episodes 25000 --seed 42 --output-dir docs/results
```

## How to Train

Each trainer accepts `--episodes` and `--seed`, and saves under `agents/<package>/results/` (gitignored). Neural agents default to the shared protocol in `config.py` (`NEURAL_TRAINING_EPISODES`, batch size, ε decay, target updates, and `train_updates_per_episode`), including the hand-then-shoe curriculum (`--no-curriculum` to disable):

```bash
python3 -m agents.q_learning_simple.train_q_learning_agent --seed 42
python3 -m agents.deep_q_learning.train_deep_q_learning_agent --seed 42
python3 -m agents.double_q_network_learning.train_double_q_network_learning_agent --seed 42
python3 -m agents.dueling_dqn.train_dueling_dqn_agent --seed 42
python3 -m agents.prioritized_replay.train_dueling_dqn_prioritized_agent --seed 42
```

Expected artifacts:

| Agent | Checkpoint |
|---|---|
| Q-learning | `agents/q_learning_simple/results/q_table.json` |
| DQN | `agents/deep_q_learning/results/deep_q_learning_model.pt` |
| Double DQN | `agents/double_q_network_learning/results/double_q_network_model.pt` |
| Dueling Double DQN | `agents/dueling_dqn/results/dueling_dqn_model.pt` |
| Dueling Double DQN + PER | `agents/prioritized_replay/results/dueling_dqn_prioritized_model.pt` |

Full training runs are long (tens to hundreds of thousands of episodes). Use a smaller `--episodes` value for smoke tests.

## Project Structure

```text
├── agents/                 # Rule, tabular, and neural agents
│   └── common.py           # Shared encoding, transitions, train helpers
├── docs/results/           # Published benchmark outputs
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
