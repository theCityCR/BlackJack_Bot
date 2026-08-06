# Project overview

Finite-shoe Blackjack RL study. Start with:

- [docs/paper.md](docs/paper.md) — research question, methods, results
- [AGENTS.md](AGENTS.md) — layout and conventions for coding agents
- [README.md](README.md) — quick start

## Scale (approximate)

- Five RL variants + verified 2-deck S17 DAS rule baseline
- Shared 200k-episode neural protocol (`config/protocol.py`)
- Primary ablation subject: Double DQN (`agents/double_dqn.py`)
- Multi-seed CLI scaffolding via `--seeds` (matrix not yet published)

## Future work

1. Re-run hand-only gap-close under paired eval and replace provisional §5.3.
2. Re-eval the rule baseline (and warm-start clones) after the chart tighten; published −0.0103 rows are historical.
3. Run multi-seed A–D / gap-close with `--seeds` and publish aggregates.
4. Variable betting / bankroll — see [docs/design_variable_betting.md](docs/design_variable_betting.md) (design only until flat-bet ≈ rule).
5. Configurable penetration and optional actor-critic methods as later follow-ons.
