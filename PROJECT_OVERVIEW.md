# Project overview

Finite-shoe Blackjack RL study. Start with:

- [docs/paper.md](docs/paper.md) — research question, methods, results
- [AGENTS.md](AGENTS.md) — layout and conventions for coding agents
- [README.md](README.md) — quick start

## Scale (approximate)

- Five RL variants + rule baseline
- Shared 200k-episode neural protocol (`config/protocol.py`)
- Primary ablation subject: Double DQN (`agents/double_dqn.py`)

## Future work

Re-run hand-only gap-close under paired per-episode eval and replace the provisional §5.3 table. Then: variable betting / bankroll, richer rule charts, configurable penetration, and optional actor-critic methods—after flat-bet play is near the rule baseline.
