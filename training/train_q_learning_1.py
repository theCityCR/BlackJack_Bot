from config import NUM_TRAINING_EPISODES
from game import BlackjackGame
from agents.q_learning_agent import QLearningAgent


def train():
    game = BlackjackGame()
    agent = QLearningAgent()

    wins = 0
    losses = 0
    draws = 0

    for episode in range(NUM_TRAINING_EPISODES):
        reward = agent.train_one_episode(game)

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

        if (episode + 1) % 50_000 == 0:
            total = wins + losses + draws
            print(f"Episode {episode + 1}")
            print(f"Win rate:  {wins / total:.3f}")
            print(f"Loss rate: {losses / total:.3f}")
            print(f"Draw rate: {draws / total:.3f}")
            print(f"Epsilon:   {agent.epsilon:.4f}")
            print()

            wins = 0
            losses = 0
            draws = 0

    return agent


if __name__ == "__main__":
    trained_agent = train()

    print("Training complete.")
    print("Example learned policy:")

    for player_value in range(12, 22):
        print(f"\nPlayer value {player_value}:")
        for dealer_upcard in range(1, 11):
            state = (player_value, dealer_upcard, False)
            action_values = trained_agent.q_table[state]
            best_action = max(action_values, key=action_values.get)
            print(f"Dealer {dealer_upcard}: {best_action.name}")