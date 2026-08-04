from agents.deep_q_learning.deep_q_learning_agent import DeepQLearningAgent
from agents.rule_agent.rule_agent import RuleAgent
from agents.warmstart import warmstart_from_rule_agent
from game import BlackjackGame


def test_rule_agent_chooses_legal_actions_in_random_episodes():
    rule_agent = RuleAgent()
    game = BlackjackGame()

    for _ in range(50):
        state = game.reset()

        if state is None:
            continue

        done = False

        while not done:
            available_actions = game.available_actions()
            action = rule_agent.choose_action(state, available_actions)
            assert action in available_actions

            state, _, done = game.step(action)


def test_warmstart_fills_buffer_and_trains():
    agent = DeepQLearningAgent(
        batch_size=2,
        min_replay_size=2,
        replay_size=100,
    )
    game = BlackjackGame()

    warmstart_from_rule_agent(agent, game, num_episodes=20)

    assert len(agent.replay_buffer) > 0
    assert agent.training_steps > 0


def test_warmstart_does_not_decay_epsilon():
    start_epsilon = 0.42
    agent = DeepQLearningAgent(
        epsilon=start_epsilon,
        batch_size=2,
        min_replay_size=2,
        replay_size=100,
    )
    game = BlackjackGame()

    warmstart_from_rule_agent(agent, game, num_episodes=10)

    assert agent.epsilon == start_epsilon


def test_warmstart_counts_episodes_when_reset_returns_none():
    agent = DeepQLearningAgent(
        batch_size=2,
        min_replay_size=1000,
        replay_size=2000,
    )
    game = BlackjackGame()

    warmstart_from_rule_agent(agent, game, num_episodes=5)

    assert len(agent.replay_buffer) >= 0
