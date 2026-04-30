from collections import defaultdict

from game import Action, BlackjackGame
from agents.q_learning_agent import QLearningAgent


NUM_EPISODES = 2_000_000
EVALUATION_EPISODES = 100_000
PRINT_EVERY = 100_000
MIN_VISITS_FOR_CONFIDENCE = 1_000


def evaluate(agent: QLearningAgent, num_episodes: int = EVALUATION_EPISODES):
    game = BlackjackGame()

    wins = losses = draws = 0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        reward = agent.play_episode(game)

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

    agent.epsilon = original_epsilon

    total = wins + losses + draws

    print("Evaluation:")
    print(f"Win rate:  {wins / total:.3f}")
    print(f"Loss rate: {losses / total:.3f}")
    print(f"Draw rate: {draws / total:.3f}")
    print()


def get_action_values(agent: QLearningAgent, state_key):
    action_values = agent.q_table[state_key]
    return action_values[Action.HIT], action_values[Action.STAND]


def best_action_with_margin(agent: QLearningAgent, state_key):
    hit_q, stand_q = get_action_values(agent, state_key)

    if hit_q > stand_q:
        return Action.HIT, hit_q - stand_q

    return Action.STAND, stand_q - hit_q


def print_policy(agent: QLearningAgent, usable_ace: bool):
    title = "with usable ace" if usable_ace else "without usable ace"
    print(f"Learned policy {title}:")
    print("Format: ACTION(visits, margin)")
    print()

    for player_value in range(12, 22):
        print(f"Player value {player_value}:")

        for dealer_upcard in range(1, 11):
            state_key = (player_value, dealer_upcard, usable_ace)
            action, margin = best_action_with_margin(agent, state_key)
            visits = agent.visit_counts[state_key]

            if visits < MIN_VISITS_FOR_CONFIDENCE:
                confidence = "LOW"
            else:
                confidence = "OK"

            print(
                f"  Dealer {dealer_upcard}: "
                f"{action.name:<5} "
                f"(visits={visits:<6}, margin={margin:.4f}, {confidence})"
            )

        print()


def print_q_values(agent: QLearningAgent, player_value: int, dealer_upcard: int, usable_ace: bool):
    state_key = (player_value, dealer_upcard, usable_ace)
    hit_q, stand_q = get_action_values(agent, state_key)
    visits = agent.visit_counts[state_key]

    print(f"State {state_key}:")
    print(f"  visits: {visits}")
    print(f"  HIT:    {hit_q:.4f}")
    print(f"  STAND:  {stand_q:.4f}")
    print()


def print_state_visit_summary(agent: QLearningAgent):
    hard_counts = []
    soft_counts = []

    for player_value in range(12, 22):
        for dealer_upcard in range(1, 11):
            hard_counts.append(agent.visit_counts[(player_value, dealer_upcard, False)])
            soft_counts.append(agent.visit_counts[(player_value, dealer_upcard, True)])

    print("State visit summary:")
    print(f"Hard states min visits: {min(hard_counts)}")
    print(f"Hard states max visits: {max(hard_counts)}")
    print(f"Soft states min visits: {min(soft_counts)}")
    print(f"Soft states max visits: {max(soft_counts)}")
    print()


def train():
    game = BlackjackGame()

    agent = QLearningAgent(
        epsilon_decay=0.999995,
    )

    agent.visit_counts = defaultdict(int)

    original_learn = agent.learn

    def learn_with_visit_count(state, action, reward, next_state, done):
        agent.visit_counts[state.as_tuple()] += 1
        original_learn(state, action, reward, next_state, done)

    agent.learn = learn_with_visit_count

    wins = losses = draws = 0

    for episode in range(NUM_EPISODES):
        reward = agent.train_one_episode(game)

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

        if (episode + 1) % PRINT_EVERY == 0:
            total = wins + losses + draws

            print(f"Episode {episode + 1}")
            print(f"Training win rate:  {wins / total:.3f}")
            print(f"Training loss rate: {losses / total:.3f}")
            print(f"Training draw rate: {draws / total:.3f}")
            print(f"Epsilon:            {agent.epsilon:.4f}")
            print()

            wins = losses = draws = 0

    return agent


if __name__ == "__main__":
    trained_agent = train()

    print("Training complete.")
    print()

    evaluate(trained_agent)
    print_state_visit_summary(trained_agent)

    print_policy(trained_agent, usable_ace=False)
    print_policy(trained_agent, usable_ace=True)

    print("Selected Q-value checks:")
    print_q_values(trained_agent, 13, 6, False)
    print_q_values(trained_agent, 16, 10, False)
    print_q_values(trained_agent, 17, 1, False)
    print_q_values(trained_agent, 18, 6, True)
    print_q_values(trained_agent, 19, 10, True)