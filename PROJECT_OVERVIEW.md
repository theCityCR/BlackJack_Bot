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

1. ~~Re-run hand-only gap-close under paired eval and replace provisional §5.3.~~ Done (`docs/results/gap_close_results.json`).
2. ~~Re-eval the rule baseline after the chart tighten.~~ Done (−0.0034 paired; `docs/results/rule_baseline_paired.json`). Warm-start clones in the equalized ablation still use the historical chart until A–D are re-run.
3. Run multi-seed A–D / gap-close with `--seeds` and publish aggregates.
4. Variable betting / bankroll — see [docs/design_variable_betting.md](docs/design_variable_betting.md) (design only until flat-bet ≈ rule).
5. Configurable penetration and optional actor-critic methods as later follow-ons.
