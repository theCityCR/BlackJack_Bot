# Project overview

Finite-shoe Blackjack RL study. Start with:

- [docs/paper.md](docs/paper.md) — research question, methods, results
- [AGENTS.md](AGENTS.md) — layout and conventions for coding agents
- [README.md](README.md) — quick start

## Scale (approximate)

- Five RL variants + verified 2-deck S17 DAS rule baseline
- Shared 200k-episode neural protocol (`config/protocol.py`)
- Primary ablation subject: Double DQN (`agents/double_dqn.py`)
- Multi-seed A–D / gap-close aggregates published (§5.4; seeds 42, 43, 44)

## Future work

1. ~~Re-run hand-only gap-close under paired eval and replace provisional §5.3.~~ Done (`docs/results/gap_close_results.json`).
2. ~~Re-eval the rule baseline after the chart tighten.~~ Done (−0.0034 paired; `docs/results/rule_baseline_paired.json`). Multi-seed ablation (§5.4) re-ran A–D with the tightened chart for warm-start; the historical single-seed §5.2 D row still reflects the pre-tighten chart.
3. ~~Run multi-seed A–D / gap-close with `--seeds` and publish aggregates.~~ Done (`docs/results/multi_seed_ablation_results.json`, `docs/results/multi_seed_gap_close_results.json`).
4. ~~Variable betting / bankroll~~ — rule + Hi-Lo spread shipped; multi-seed 100k paired aggregate published ([§5.5](docs/paper.md), `docs/results/multi_seed_variable_betting_results.json`). Optional: bankroll / RoR reporting.
5. Configurable penetration and optional actor-critic methods as later follow-ons.
