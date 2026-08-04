# Blackjack Reinforcement Learning Bot

## Project Summary

This project is a Blackjack simulation and reinforcement learning environment built to explore how different AI agents make decisions under uncertainty. It compares a traditional rule-based player with several learning-based approaches, ranging from tabular Q-learning to more advanced deep reinforcement learning architectures.

Unlike many introductory Blackjack projects, this environment includes a persistent finite card shoe, card-count information, splitting, doubling down, natural Blackjack payouts, and multiple simultaneous player hands. These mechanics create a larger and more realistic decision space for the agents to learn from.

## The Problem

Blackjack is simple to understand but difficult to play optimally. A player must make sequential decisions using incomplete information while accounting for probabilities, changing deck composition, and the dealer's visible card.

The project investigates a broader machine learning question: how do different reinforcement learning techniques perform as an environment becomes more realistic and its state space becomes more complex?

## How It Works

The game engine manages the deck, player hands, dealer behavior, legal actions, and rewards. On each turn, an agent receives a structured game state containing information such as:

- The player's current hand value
- The dealer's visible card
- Whether the player has a usable ace
- Whether doubling or splitting is legal
- Split-hand context
- The number of each card value remaining in the shoe

Neural agents encode that state into a shared 19-feature vector. The agent then chooses one of four actions: **hit**, **stand**, **double**, or **split**. At the end of the round, it receives a positive, negative, or neutral reward based on the result and the amount wagered.

Default rules: 2 decks, reshuffle below 26 cards, dealer stands on all 17s (S17), DAS and re-splits allowed, no hitting split aces.

## AI Approaches

The repository explores several strategies:

- **Rule-based agent:** A basic-strategy-inspired benchmark used as a reliable baseline.
- **Q-learning:** A tabular agent that learns action values through repeated play.
- **Deep Q-Network (DQN):** Uses a neural network and experience replay to handle a larger state space.
- **Double DQN:** Reduces the tendency of standard DQN models to overestimate action values.
- **Dueling Double DQN:** Separately estimates state value and action advantage, with Double DQN targets.
- **Prioritized experience replay:** Trains more frequently on transitions from which the agent has the most to learn.

## Technical Highlights

- Custom Blackjack environment written in Python
- Configurable, persistent multi-deck shoe with realistic card frequencies
- Exact remaining-card count vector for card-aware decision-making
- Support for splits, re-splits, double-down decisions, and split-hand rewards
- Legal-action masking to prevent agents from selecting invalid moves
- Shared neural state encoding and training helpers in `agents/common.py`
- PyTorch neural networks, replay buffers, and target-network synchronization
- Deterministic evaluation through configurable random seeds
- Seeded training entrypoints with checkpoints under `agents/*/results/`
- Automated test suite covering game rules, deck behavior, and agent decisions
- Packaged with `pyproject.toml` and GitHub Actions CI on Python 3.10/3.11

## What Makes It Interesting

The main challenge is not simply teaching an agent when to hit or stand. Adding finite-deck information and multiple hands greatly expands the number of possible states. This makes the project a useful demonstration of why simple tabular methods struggle as complexity increases, and why neural-network-based methods and training-stability techniques become valuable.

The project also provides a clear comparison between human-designed heuristics and policies learned from experience. Because all agents use the same environment and reward system, their behavior and performance can be evaluated consistently.

## Current Status

The core game environment and rule-based benchmark are fully functional. The repository contains implementations of the major reinforcement learning agents, reproducible evaluation tooling, versioned benchmark results, and an automated test suite. Trained checkpoints are kept outside version control because of their size. The environment currently models a two-deck shoe by default and reshuffles only after reaching a configurable cut-card threshold.

The saved neural checkpoints were evaluated alongside the rule-based baseline over 25,000 seeded rounds per agent. The rule baseline achieved the strongest average reward. Rather than hiding this negative result, the project documents it as evidence that a more complex architecture does not automatically overcome a large state space or imperfect state and reward design. Those checkpoints used unequal training budgets; neural trainers now share a common episode count and hyperparameter schedule so architecture comparisons are not confounded by compute. Full results are available in [`docs/results`](docs/results).

## Project by the Numbers

### Approximately how many lines of code?

The repository contains approximately **6,470 lines of Python**, including tests. About **3,500 lines** are game, agent, training, and evaluation code, while roughly **2,970 lines** are tests. These figures count physical lines, so they include comments, docstrings, and blank lines.

### How many training episodes were run?

Configured neural training is **800,000 episodes** (four agents × 200,000 under the shared protocol), plus tabular Q-learning at 50,000 by default. Lifetime experiment totals including earlier unequal runs are higher. These figures exclude evaluation-only simulations.

### How many reinforcement learning algorithms were implemented?

The project ultimately implemented **five RL variants**:

1. Tabular Q-learning
2. Deep Q-Network (DQN)
3. Double DQN
4. Dueling Double DQN
5. Dueling Double DQN with prioritized experience replay

The rule-based strategy agent is an additional benchmark, but it is not counted as a reinforcement learning algorithm.

### How many times was the environment or state representation redesigned?

The environment and state representation went through approximately **four major redesigns** after the initial prototype. The project evolved from a simple Blackjack environment to support splits and doubles, then multi-hand state and reward tracking, finite-shoe card-count features, and finally a persistent multi-deck shoe with realistic card frequencies and threshold-based reshuffling. Smaller fixes and tuning changes are not included in this count.

## Future Improvements

Possible next steps include:

- A web interface where users can play against or observe trained agents
- Betting and bankroll-management strategies
- PPO or actor-critic agents
- More configurable casino rules and shoe penetration
- Explainable decisions showing why an agent selected a particular action
- Automated experiment tracking and learning-curve comparison
- Closing the performance gap with the rule baseline through state/reward redesign
- Two-phase curriculum (hand features → shoe-aware state) is now the default neural training path; further tuning may still help

## Technology Stack

- Python
- PyTorch
- Reinforcement learning
- Deep Q-learning
- pytest

## Short Application Description

> A realistic Blackjack reinforcement learning environment that compares rule-based strategy, Q-learning, and several deep Q-network architectures. The simulator supports finite multi-deck card counting, splits, doubles, and legal-action masking, providing a controlled way to study how AI decision-making changes as the state space becomes more complex.
