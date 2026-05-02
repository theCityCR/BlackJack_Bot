import json
import os
import random
from collections import Counter, defaultdict

from agents.q_learning_agent import QLearningAgent
from config import Q_TABLE_PATH
from game import Action, BlackjackGame


TOTAL_EPISODES = 3_000_000
EVALUATION_EPISODES = 500_000
PRINT_EVERY = 50_000

UNCOMMON_START_RATE = 0.20

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999995


def random_card():
    return random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


def reward_bucket(reward):
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def build_uncommon_starts():
    starts = []
    dealer_upcards = list(range(1, 11))

    # Pair starts for split learning.
    for card in range(1, 11):
        for dealer_upcard in dealer_upcards:
            starts.append(([card, card], [dealer_upcard, random_card()]))

    # Soft totals: A,2 through A,9.
    for second_card in range(2, 10):
        for dealer_upcard in dealer_upcards:
            starts.append(([1, second_card], [dealer_upcard, random_card()]))

    # Hard double totals: 9, 10, 11.
    hard_double_hands = [
        [2, 7], [3, 6], [4, 5],
        [2, 8], [3, 7], [4, 6],
        [2, 9], [3, 8], [4, 7], [5, 6],
    ]

    for player_cards in hard_double_hands:
        for dealer_upcard in dealer_upcards:
            starts.append((player_cards, [dealer_upcard, random_card()]))

    return starts


def train_one_forced_start_episode(agent, game, player_cards, dealer_cards):
    state = game.reset_with_cards(player_cards, dealer_cards)

    if state is None:
        agent._decay_epsilon()
        return game.round_reward

    transitions = []
    done = False
    round_reward = 0.0

    while not done:
        hand_index = game.active_hand_index
        available_actions = game.available_actions()
        action = agent.choose_action(state, available_actions)

        next_state, round_reward, done = game.step(action)

        transitions.append(
            {
                "hand_index": hand_index,
                "state": state,
                "action": action,
                "next_state": next_state,
            }
        )

        state = next_state

    hand_rewards = game.hand_rewards

    for index, transition in enumerate(transitions):
        state = transition["state"]
        action = transition["action"]
        next_state = transition["next_state"]
        hand_index = transition["hand_index"]

        hand_reward = hand_rewards[hand_index]

        is_last_transition = index == len(transitions) - 1
        next_is_different_hand = (
            not is_last_transition
            and transitions[index + 1]["hand_index"] != hand_index
        )

        terminal_for_this_hand = is_last_transition or next_is_different_hand

        if terminal_for_this_hand:
            agent.learn(state, action, hand_reward, None, True, [])
        else:
            agent.learn(state, action, 0.0, next_state, False, None)

    agent._decay_epsilon()
    return round_reward


def train():
    agent = QLearningAgent(
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    )

    normal_game = BlackjackGame()
    forced_game = BlackjackGame()
    starts = build_uncommon_starts()

    recent_results = Counter()

    for episode in range(1, TOTAL_EPISODES + 1):
        if random.random() < UNCOMMON_START_RATE:
            player_cards, dealer_cards = random.choice(starts)
            reward = train_one_forced_start_episode(
                agent,
                forced_game,
                player_cards,
                dealer_cards,
            )
        else:
            reward = agent.train_one_episode(normal_game)

        recent_results[reward_bucket(reward)] += 1

        if episode % PRINT_EVERY == 0:
            print_progress(episode, recent_results, agent)
            recent_results.clear()

    return agent


def print_progress(episode, recent_results, agent):
    total = sum(recent_results.values())

    print(f"Episode {episode}")
    print(f"Win rate:   {recent_results['win'] / total:.3f}")
    print(f"Loss rate:  {recent_results['loss'] / total:.3f}")
    print(f"Draw rate:  {recent_results['draw'] / total:.3f}")
    print(f"Epsilon:    {agent.epsilon:.4f}")
    print()


def legal_actions_from_state_key(state_key):
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


def best_action_and_margin(agent, state_key):
    values = agent.q_table[state_key]
    legal_actions = legal_actions_from_state_key(state_key)

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


def print_all_policy(agent):
    print("Learned policy for all visited states")
    print("Format: ACTION | visits | margin | state")
    print()

    state_keys = sorted(
        agent.q_table.keys(),
        key=lambda key: (
            key[5],  # is_split_hand
            key[7],  # num_hands
            key[6],  # active_hand_index
            key[2],  # usable_ace
            key[0],  # player_value
            key[1],  # dealer_upcard
            key[3],  # can_double
            key[4],  # can_split
        ),
    )

    for state_key in state_keys:
        action, margin = best_action_and_margin(agent, state_key)

        print(
            f"{action.name:<6} | "
            f"margin={margin:>7.4f} | "
            f"state={state_key}"
        )

    print()


def print_table_policy(agent, usable_ace, can_double, can_split, is_split_hand):
    title = (
        f"usable_ace={usable_ace}, "
        f"can_double={can_double}, "
        f"can_split={can_split}, "
        f"is_split_hand={is_split_hand}"
    )

    print(title)
    print()

    for player_value in range(4, 22):
        row = []

        for dealer_upcard in range(1, 11):
            matching_keys = [
                key for key in agent.q_table.keys()
                if key[0] == player_value
                and key[1] == dealer_upcard
                and key[2] == usable_ace
                and key[3] == can_double
                and key[4] == can_split
                and key[5] == is_split_hand
            ]

            if not matching_keys:
                row.append("----")
                continue

            state_key = matching_keys[0]
            action, _ = best_action_and_margin(agent, state_key)
            row.append(action.name[:4])

        print(f"{player_value:>2}: " + " ".join(f"{x:>4}" for x in row))

    print()


def print_policy_summary_tables(agent):
    print("Policy summary tables")
    print("Dealer columns: A 2 3 4 5 6 7 8 9 10")
    print()

    print_table_policy(agent, usable_ace=False, can_double=True, can_split=False, is_split_hand=False)
    print_table_policy(agent, usable_ace=False, can_double=False, can_split=False, is_split_hand=False)
    print_table_policy(agent, usable_ace=True, can_double=True, can_split=False, is_split_hand=False)
    print_table_policy(agent, usable_ace=True, can_double=False, can_split=False, is_split_hand=False)
    print_table_policy(agent, usable_ace=False, can_double=True, can_split=True, is_split_hand=False)
    print_table_policy(agent, usable_ace=True, can_double=True, can_split=True, is_split_hand=False)


def evaluate_rewards(agent, num_games=EVALUATION_EPISODES):
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


def save_q_table(agent, path=Q_TABLE_PATH):
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


if __name__ == "__main__":
    trained_agent = train()

    print("Training complete.")
    print()

    print_policy_summary_tables(trained_agent)

    # This may be very long, but you asked for all policy.
    print_all_policy(trained_agent)

    evaluate_rewards(trained_agent)

    save_q_table(trained_agent)