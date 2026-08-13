# Agent navigation map

Finite-shoe Blackjack RL study. Human writeup: [docs/paper.md](docs/paper.md).
Storefront: [README.md](README.md).

## Layout

| Path | Role |
|---|---|
| `game.py` / `cards.py` | Environment (S17, DAS, 2-deck shoe, count vector) |
| `config/rules.py` | Casino rules + rewards |
| `config/protocol.py` | Shared neural training / ablation knobs |
| `config/tabular.py` | Tabular Q hyperparameters |
| `agents/rule.py` | Basic-strategy baseline |
| `agents/counting.py` / `betting.py` / `spread_rule.py` | Hi-Lo true count + stake schedule + rule play |
| `agents/bankroll.py` | Bankroll path + trip risk-of-ruin for spread demos |
| `agents/tabular_q.py` | Tabular Q-learning |
| `agents/dqn.py` / `double_dqn.py` / `dueling.py` / `prioritized.py` | Neural agents (study surface = net + `train_step`) |
| `agents/neural_base.py` | Shared ε-greedy / episode / remember loop |
| `agents/policy_base.py` / `reinforce.py` / `a2c.py` / `ppo.py` | Bet+play policy-gradient agents (on-policy) |
| `agents/pg_warmstart.py` / `train_pg_cli.py` | Spread-rule CE warm-start + PG train CLIs |
| `agents/common.py` | `encode_state` / `encode_shoe`, training loop, checkpoints |
| `agents/episode.py` | Per-hand reward attribution |
| `agents/networks.py` / `replay.py` | Shared Dueling net, `BetPlayActorCritic`, PER buffer |
| `config/pg.py` | Policy-gradient hyperparameters (separate from DQN protocol) |
| `agents/train_*.py` | Thin CLIs |
| `scripts/run_ablation_double_dqn.py` | Ablation conditions A–D |
| `scripts/run_variable_betting_eval.py` | Paired flat vs Hi-Lo spread + rule (`--seeds`; optional `--pg-agent`) |
| `scripts/run_penetration_sweep.py` | Spread EV vs reshuffle cut / dealt penetration (§5.6) |
| `scripts/run_pg_spread_bakeoff.py` | PG vs flat/spread rule bake-off (§5.7) |
| `evaluate_agents.py` | Seeded eval of available checkpoints |
| `docs/results/` | **Published** benchmarks — do not overwrite casually |

Checkpoints: `agents/results/<agent_name>/` (gitignored).

## Commands

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 main.py --episodes 1000 --seed 42
python3 -m agents.train_double_dqn --seed 42
python3 -m agents.train_reinforce --seed 42
python3 -m agents.train_a2c --seed 42
python3 -m agents.train_ppo --seed 42
python3 scripts/run_ablation_double_dqn.py --smoke
python3 scripts/run_variable_betting_eval.py --smoke
python3 scripts/run_variable_betting_eval.py --smoke --bankroll
python3 scripts/run_variable_betting_eval.py --smoke \
  --pg-agent a2c --pg-checkpoint agents/results/a2c/a2c_bet_play_model.pt
python3 scripts/run_penetration_sweep.py --smoke
python3 scripts/run_pg_spread_bakeoff.py --smoke
python3 evaluate_agents.py --episodes 25000 --seed 42
```
## Conventions for agents

- Prefer editing shared infra (`neural_base`, `policy_base`, `common`, `episode`) over copy-pasting into one agent.
- Keep Q-target math / loss / architecture differences intentional per agent file.
- Ablation default subject is **Double DQN**.
- Do not casually edit published artifacts under `docs/results/`.
