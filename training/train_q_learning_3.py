import json
import os
import random
from collections import Counter, defaultdict

from agents.q_learning_agent import QLearningAgent
from config import NUM_TRAINING_EPISODES, Q_TABLE_PATH
from game import Action, BlackjackGame


NORMAL_TRAINING_EPISODES = NUM_TRAINING_EPISODES
UNCOMMON_START_EPISODES = 300_000
EVALUATION_EPISODES = 100_000
PRINT_EVERY = 50_000


def reward_bucket(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def build_uncommon_starts():
    starts = []

    dealer_upcards = list(range(1, 11))

    # Pair starts: useful for split learning.
    for card in range(1, 11):
        for dealer_upcard in dealer_upcards:
            starts.append(([card, card], [dealer_upcard, random_hidden_card()]))

    # Soft totals: A,2 through A,9.
    for second_card in range(2, 10):
        for dealer_upcard in dealer_upcards:
            starts.append(([1, second_card], [dealer_upcard, random_hidden_card()]))

    # Hard double totals: 9, 10, 11.
    hard_double_hands = [
        [2, 7], [3, 6], [4, 5],
        [2, 8], [3, 7], [4, 6],
        [2, 9], [3, 8], [4, 7], [5, 6],
    ]

    for player_cards in hard_double_hands:
        for dealer_upcard in dealer_upcards:
            starts.append((player_cards, [dealer_upcard, random_hidden_card()]))

    return starts


def random_hidden_card():
    return random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


def train_normal(agent: QLearningAgent):
    game = BlackjackGame()
    recent_results = Counter()

    print("Normal training")
    print()

    for episode in range(1, NORMAL_TRAINING_EPISODES + 1):
        reward = agent.train_one_episode(game)
        recent_results[reward_bucket(reward)] += 1

        if episode % PRINT_EVERY == 0:
            print_progress(episode, recent_results, agent)
            recent_results.clear()


def train_uncommon_starts(agent: QLearningAgent):
    game = BlackjackGame()
    starts = build_uncommon_starts()
    recent_results = Counter()

    print("Uncommon-start training")
    print()

    for episode in range(1, UNCOMMON_START_EPISODES + 1):
        player_cards, dealer_cards = random.choice(starts)

        reward = train_one_forced_start_episode(
            agent,
            game,
            player_cards,
            dealer_cards,
        )

        recent_results[reward_bucket(reward)] += 1

        if episode % PRINT_EVERY == 0:
            print_progress(episode, recent_results, agent)
            recent_results.clear()


def train_one_forced_start_episode(
    agent: QLearningAgent,
    game: BlackjackGame,
    player_cards,
    dealer_cards,
) -> float:
    """
    Same idea as QLearningAgent.train_one_episode, except the initial cards
    are forced instead of random.
    """
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
            agent.learn(
                state=state,
                action=action,
                reward=hand_reward,
                next_state=None,
                done=True,
                next_available_actions=[],
            )
        else:
            agent.learn(
                state=state,
                action=action,
                reward=0.0,
                next_state=next_state,
                done=False,
                next_available_actions=None,
            )

    agent._decay_epsilon()
    return round_reward


def evaluate_rewards(agent: QLearningAgent, num_games: int = EVALUATION_EPISODES):
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


def print_progress(episode, recent_results, agent):
    total = sum(recent_results.values())

    print(f"Episode {episode}")
    print(f"Win rate:   {recent_results['win'] / total:.3f}")
    print(f"Loss rate:  {recent_results['loss'] / total:.3f}")
    print(f"Draw rate:  {recent_results['draw'] / total:.3f}")
    print(f"Epsilon:    {agent.epsilon:.4f}")
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


def train_mixed(agent: QLearningAgent):
    normal_game = BlackjackGame()
    forced_game = BlackjackGame()
    starts = build_uncommon_starts()

    recent_results = Counter()

    total_episodes = NORMAL_TRAINING_EPISODES + UNCOMMON_START_EPISODES

    for episode in range(1, total_episodes + 1):
        if random.random() < 0.20:
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


def train():
    agent = QLearningAgent()
    train_mixed(agent)
    return agent


if __name__ == "__main__":
    trained_agent = train()

    print("Training complete.")
    print()

    evaluate_rewards(trained_agent, num_games=500_000)

    save_q_table(trained_agent)