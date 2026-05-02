"""
Train the Q-learning Blackjack agent.

Supports:
- Hit
- Stand
- Double
- Split
"""

import json
import os
from collections import Counter, defaultdict

from config import (
    NUM_TRAINING_EPISODES,
    EVALUATION_EPISODES,
    Q_TABLE_PATH,
)
from game import Action, BlackjackGame
from agents.q_learning_agent import QLearningAgent


PRINT_EVERY = 50_000
MIN_VISITS_FOR_CONFIDENCE = 500


def reward_bucket(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def train() -> QLearningAgent:
    game = BlackjackGame()
    agent = QLearningAgent()

    agent.visit_counts = defaultdict(int)
    original_learn = agent.learn

    def learn_with_visit_count(
        state,
        action,
        reward,
        next_state,
        done,
        next_available_actions=None,
    ):
        agent.visit_counts[state.as_tuple()] += 1
        original_learn(
            state,
            action,
            reward,
            next_state,
            done,
            next_available_actions,
        )

    agent.learn = learn_with_visit_count

    recent_results = Counter()

    for episode in range(1, NUM_TRAINING_EPISODES + 1):
        reward = agent.train_one_episode(game)
        recent_results[reward_bucket(reward)] += 1

        if episode % PRINT_EVERY == 0:
            total = sum(recent_results.values())

            print(f"Episode {episode}")
            print(f"Win rate:   {recent_results['win'] / total:.3f}")
            print(f"Loss rate:  {recent_results['loss'] / total:.3f}")
            print(f"Draw rate:  {recent_results['draw'] / total:.3f}")
            print(f"Epsilon:    {agent.epsilon:.4f}")
            print()

            recent_results.clear()

    return agent


def evaluate(agent: QLearningAgent, num_episodes: int = EVALUATION_EPISODES):
    game = BlackjackGame()
    results = Counter()

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        reward = agent.play_episode(game)
        results[reward_bucket(reward)] += 1

    agent.epsilon = original_epsilon

    total = sum(results.values())

    print("Evaluation:")
    print(f"Win rate:   {results['win'] / total:.3f}")
    print(f"Loss rate:  {results['loss'] / total:.3f}")
    print(f"Draw rate:  {results['draw'] / total:.3f}")
    print()


def get_action_values(agent: QLearningAgent, state_key):
    values = agent.q_table[state_key]
    return {
        action: values[action]
        for action in Action
    }


def best_action_with_margin(agent: QLearningAgent, state_key, legal_actions):
    values = get_action_values(agent, state_key)

    best_value = max(values[action] for action in legal_actions)
    best_actions = [
        action for action in legal_actions
        if values[action] == best_value
    ]

    best_action = best_actions[0]

    other_values = [
        values[action]
        for action in legal_actions
        if action != best_action
    ]

    margin = best_value - max(other_values) if other_values else 0.0
    return best_action, margin


def possible_legal_actions(state_key):
    (
        player_value,
        dealer_upcard,
        usable_ace,
        can_double,
        can_split,
        is_split_hand,
        active_hand_index,
        num_hands,
    ) = state_key

    if player_value >= 21:
        return [Action.STAND]

    actions = [Action.HIT, Action.STAND]

    if can_double:
        actions.append(Action.DOUBLE)

    if can_split:
        actions.append(Action.SPLIT)

    return actions


def print_state_visit_summary(agent: QLearningAgent):
    visit_counts = list(agent.visit_counts.values())

    if not visit_counts:
        print("No visited states.")
        return

    print("State visit summary:")
    print(f"Visited states: {len(visit_counts)}")
    print(f"Min visits:     {min(visit_counts)}")
    print(f"Max visits:     {max(visit_counts)}")
    print()


def print_visited_policy(agent: QLearningAgent, usable_ace: bool):
    title = "soft totals" if usable_ace else "hard totals"
    print(f"Learned policy for visited {title}:")
    print("Format: ACTION(visits)")
    print()

    grouped = defaultdict(list)

    for state_key, visits in agent.visit_counts.items():
        (
            player_value,
            dealer_upcard,
            state_usable_ace,
            can_double,
            can_split,
            is_split_hand,
            active_hand_index,
            num_hands,
        ) = state_key

        if state_usable_ace != usable_ace:
            continue

        if not 12 <= player_value <= 21:
            continue

        # Print only the most common/simple one-hand states.
        if is_split_hand:
            continue
        if active_hand_index != 0 or num_hands != 1:
            continue

        grouped[(player_value, dealer_upcard)].append((state_key, visits))

    for player_value in range(12, 22):
        print(f"Player value {player_value}:")

        for dealer_upcard in range(1, 11):
            candidates = grouped.get((player_value, dealer_upcard), [])

            if not candidates:
                print(f"  Dealer {dealer_upcard}: NOT VISITED")
                continue

            state_key, visits = max(candidates, key=lambda item: item[1])
            legal_actions = possible_legal_actions(state_key)

            action, margin = best_action_with_margin(
                agent,
                state_key,
                legal_actions,
            )

            print(
                f"  Dealer {dealer_upcard}: "
                f"{action.name:<6} "
                f"(visits={visits:<6}, margin={margin:.4f}, state={state_key})"
            )

        print()


def save_q_table(agent: QLearningAgent, path: str = Q_TABLE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serializable = {}

    for state_key, action_values in agent.q_table.items():
        serializable[str(state_key)] = {
            action.name: value
            for action, value in action_values.items()
        }

    with open(path, "w") as file:
        json.dump(serializable, file, indent=2)

    print(f"Saved Q-table to {path}")
    print()


def evaluate_rewards(agent: QLearningAgent, num_games: int = 100_000):
    """
    Test the trained agent by simulating many games without learning.

    Prints average reward and reward distribution.
    """
    game = BlackjackGame()

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_reward = 0.0
    reward_counts = defaultdict(int)

    for _ in range(num_games):
        reward = agent.play_episode(game)
        total_reward += reward
        reward_counts[reward] += 1

    agent.epsilon = original_epsilon

    print(f"Tested on {num_games} games")
    print(f"Average reward per game: {total_reward / num_games:.4f}")
    print()
    print("Reward distribution:")

    for reward in sorted(reward_counts):
        count = reward_counts[reward]
        percentage = count / num_games * 100
        print(f"  reward {reward:>5}: {count:>8} ({percentage:>6.2f}%)")

    print()


if __name__ == "__main__":
    trained_agent = train()

    print("Training complete.")
    print()

    evaluate_rewards(trained_agent, num_games=100_000)

    print_state_visit_summary(trained_agent)

    print_visited_policy(trained_agent, usable_ace=False)
    print_visited_policy(trained_agent, usable_ace=True)

    save_q_table(trained_agent)