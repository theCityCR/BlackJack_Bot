from evaluate_agents import evaluate_agent
from agents.rule_agent.rule_agent import RuleAgent


def test_evaluation_is_reproducible():
    first = evaluate_agent("rule", RuleAgent(), episodes=100, seed=7)
    second = evaluate_agent("rule", RuleAgent(), episodes=100, seed=7)

    assert first == second
    assert first["episodes"] == 100
    assert first["win_rate"] + first["loss_rate"] + first["draw_rate"] == 1.0
