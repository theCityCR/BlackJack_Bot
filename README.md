# Blackjack Reinforcement Learning Bot

A reinforcement learning project exploring how different AI agents learn to play Blackjack under increasingly realistic game conditions.

The project began with a simple hit/stand Blackjack environment and evolved into a full training framework supporting:

* Q-learning
* Deep Q-learning (DQN)
* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* Finite-deck card counting
* Splits and doubles

The main focus of the project was understanding how environment design, state representation, and neural-network architecture affect learning performance in reinforcement learning systems.

---

# Overview

Most beginner Blackjack RL projects use an infinite deck and only allow hit/stand actions. This project instead models a much more realistic environment:

* Finite randomized shoe
* Multiple player hands after splits
* Double-down decisions
* Natural blackjack payouts
* Dealer blackjack checks before player actions
* Remaining-card count information

The goal was to study whether reinforcement learning agents could learn stronger strategies when given access to deck composition and more complex game mechanics. 

---

# Main Features

## Blackjack Environment

The environment supports:

* Hit
* Stand
* Double
* Split
* Split-hand tracking
* Natural blackjack payouts (3:2)
* Configurable rules

The game engine exposes a structured public `GameState` object to agents for training and evaluation. 

---

## Finite Randomized Shoe

Instead of using an infinite deck, each round generates a fresh randomized 26-card shoe.

Each card is sampled from a real Blackjack card distribution, allowing agents to use remaining-card information during training. This introduces deck-composition effects and lightweight card-counting behavior into the learning process. 

Agents receive a remaining-card count vector in the format:

```python id="w77bko"
[A, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

# Agents

## Rule-Based Agent

Implemented a basic-strategy-inspired benchmark agent supporting:

* Hard totals
* Soft totals
* Splits
* Doubles

Used as a baseline for comparing learned policies against traditional Blackjack heuristics. 

---

## Q-Learning Agent

Built a tabular Q-learning agent with support for:

* Split hands
* Per-hand rewards
* Epsilon-greedy exploration
* Legal-action masking

The project explored how traditional Q-learning scales poorly as the state space grows due to finite-deck information and multi-hand gameplay. 

Core update rule:

Q(s,a) \leftarrow Q(s,a)+\alpha\left(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right)

---

## Deep Q-Learning (DQN)

Implemented a neural-network-based reinforcement learning agent using:

* Experience replay
* Target networks
* Epsilon decay
* Legal-action masking

The DQN agent encoded Blackjack game states into compact numerical feature vectors for training. 

---

## Double DQN

Extended the DQN implementation with Double DQN to reduce Q-value overestimation.

Features included:

* Separate online and target networks
* Replay buffers
* Gradient clipping
* Stable target updates

y=r+\gamma Q_{target}\left(s',\arg\max_a Q_{main}(s',a)\right)

---

## Dueling DQN + Prioritized Replay

Built a Dueling DQN architecture with prioritized experience replay to improve learning stability and sample efficiency.

The architecture separates:

* State-value estimation
* Action-advantage estimation

and combines them into final Q-values:

Q(s,a)=V(s)+A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a')

The prioritized replay buffer samples important transitions more frequently using TD-error-based priorities.

---

# Technical Highlights

* Designed a modular Blackjack simulation environment from scratch in Python
* Implemented reinforcement learning agents using both tabular and deep-learning approaches
* Built replay buffers, target-network synchronization, and legal-action masking systems
* Encoded finite-deck card-count information into neural-network state representations
* Added support for split-hand reward attribution to improve training quality
* Trained and evaluated agents over hundreds of thousands of simulated games
* Used PyTorch for neural-network training and optimization

---

# Example Project Structure

```text id="69my8v"
project/
│
├── agents/
│   ├── rule_agent/
│   ├── q_learning/
│   ├── deep_q_learning/
│   ├── double_q_network_learning/
│   ├── dueling_dqn/
│   └── prioritized_replay/
│
├── training/
├── tests/
├── results/
│
├── cards.py
├── game.py
├── config.py
└── README.md
```

---

# Technologies Used

* Python
* PyTorch
* Reinforcement Learning
* Deep Q-Learning
* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* pytest

---

# Future Improvements

Potential future extensions include:

* PPO / Actor-Critic methods
* Betting optimization strategies
* Monte Carlo Tree Search
* Real casino shoe penetration simulation
* Training visualization dashboards
* Human-vs-agent interface

---

# Why I Built This

I built this project to better understand how reinforcement learning systems behave in environments with:

* Large state spaces
* Sparse rewards
* Delayed feedback
* Imperfect information
* Sequential decision-making

It also gave me experience implementing RL algorithms directly rather than relying on high-level frameworks.
