from game import BlackjackGame
from agents.q_learning_agent import QLearningAgent


NUM_EPISODES = 2_000_000
EVALUATION_EPISODES = 100_000
PRINT_EVERY = 100_000


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


def print_policy(agent: QLearningAgent, usable_ace: bool = False):
    title = "with usable ace" if usable_ace else "without usable ace"
    print(f"Learned policy {title}:")

    for player_value in range(12, 22):
        print(f"\nPlayer value {player_value}:")

        for dealer_upcard in range(1, 11):
            state = (player_value, dealer_upcard, usable_ace)
            action_values = agent.q_table[state]
            best_action = max(action_values, key=action_values.get)

            print(f"Dealer {dealer_upcard}: {best_action.name}")


def print_q_values(agent: QLearningAgent, player_value: int, dealer_upcard: int, usable_ace: bool = False):
    state = (player_value, dealer_upcard, usable_ace)
    action_values = agent.q_table[state]

    print(f"Q-values for state {state}:")
    print(f"  HIT:   {action_values.get(next(a for a in action_values if a.name == 'HIT')):.4f}")
    print(f"  STAND: {action_values.get(next(a for a in action_values if a.name == 'STAND')):.4f}")
    print()


def train():
    game = BlackjackGame()

    agent = QLearningAgent(
        epsilon_decay=0.999995,
    )

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

    print_policy(trained_agent, usable_ace=False)
    print()
    print_policy(trained_agent, usable_ace=True)
    print()

    print("Suspicious state Q-value checks:")
    print_q_values(trained_agent, 13, 6, False)
    print_q_values(trained_agent, 16, 10, False)
    print_q_values(trained_agent, 17, 1, False)