"""Shared CLI helpers for bet+play policy-gradient trainers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from agents.cli_seeds import add_seed_arguments, seed_artifact_dir, seeds_from_args
from agents.common import (
    agent_results_path,
    evaluate_greedy,
    load_policy_checkpoint,
    print_distribution,
    save_policy_checkpoint,
    set_seed,
)
from agents.learning_curves import LearningCurveLogger
from agents.pg_warmstart import warmstart_from_spread_rule
from config import (
    PG_BET_ENTROPY_COEF,
    PG_BET_FOCUS_ARTIFACT_SUBDIR,
    PG_BET_FOCUS_BET_ENTROPY_COEF,
    PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES,
    PG_BET_FOCUS_FINAL_EVAL_EPISODES,
    PG_BET_FOCUS_FREEZE_PLAY,
    PG_BET_FOCUS_PLAY_ENTROPY_COEF,
    PG_BET_FOCUS_PRINT_INTERVAL,
    PG_BET_FOCUS_TEACHER_BET_CE_COEF,
    PG_BET_FOCUS_TRAINING_EPISODES,
    PG_BET_FOCUS_WARMSTART_EPISODES,
    PG_CHECKPOINT_EVAL_EPISODES,
    PG_FINAL_EVAL_EPISODES,
    PG_FREEZE_PLAY,
    PG_LEARNING_CURVE_ENABLED,
    PG_LEARNING_CURVE_FILENAME,
    PG_PLAY_ENTROPY_COEF,
    PG_PRINT_INTERVAL,
    PG_TEACHER_BET_CE_COEF,
    PG_TRAINING_EPISODES,
    PG_WARMSTART_ENABLED,
    PG_WARMSTART_EPISODES,
    RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
)
from game import BlackjackGame


def build_pg_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes", type=int, default=None)
    add_seed_arguments(parser)
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Skip SpreadRuleAgent behavior cloning before RL",
    )
    parser.add_argument(
        "--warmstart-episodes",
        type=int,
        default=None,
        help=(
            "Behavior-cloning episodes before RL "
            f"(default {PG_WARMSTART_EPISODES}; "
            f"{PG_BET_FOCUS_WARMSTART_EPISODES} with --bet-focus)."
        ),
    )
    parser.add_argument(
        "--bet-focus",
        action="store_true",
        help=(
            "Stake-retention preset: freeze rule play, higher bet entropy, "
            "teacher bet CE, longer warm-start / train budget, leaner probes."
        ),
    )
    parser.add_argument(
        "--freeze-play",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train bet head only; rule chart plays (default: on with --bet-focus).",
    )
    parser.add_argument(
        "--bet-entropy-coef",
        type=float,
        default=None,
        help=f"Entropy bonus on the bet head (default {PG_BET_ENTROPY_COEF}).",
    )
    parser.add_argument(
        "--play-entropy-coef",
        type=float,
        default=None,
        help=f"Entropy bonus on the play head (default {PG_PLAY_ENTROPY_COEF}).",
    )
    parser.add_argument(
        "--teacher-bet-ce-coef",
        type=float,
        default=None,
        help=(
            "CE pull toward SpreadRule bet during RL "
            f"(default {PG_TEACHER_BET_CE_COEF})."
        ),
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=None,
        help=(
            "Greedy probe every N train episodes "
            f"(default {PG_PRINT_INTERVAL}; "
            f"{PG_BET_FOCUS_PRINT_INTERVAL} with --bet-focus)."
        ),
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=None,
        help=(
            "Greedy episodes per mid-run probe "
            f"(default {PG_CHECKPOINT_EVAL_EPISODES}; "
            f"{PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES} with --bet-focus)."
        ),
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=None,
        help=(
            "Greedy episodes after training (0 skips). "
            f"Default {PG_FINAL_EVAL_EPISODES}; "
            f"{PG_BET_FOCUS_FINAL_EVAL_EPISODES} with --bet-focus."
        ),
    )
    parser.add_argument(
        "--artifact-subdir",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under agents/results/<agent>/ "
            f"(default none; '{PG_BET_FOCUS_ARTIFACT_SUBDIR}' with --bet-focus)."
        ),
    )
    parser.add_argument(
        "--reshuffle-threshold",
        type=int,
        default=None,
        help=(
            "Cut card: reshuffle when remaining cards ≤ this value "
            f"(default {RESHUFFLE_WHEN_CARDS_REMAINING_BELOW} from config)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, mps, cuda). Default: CUDA if available else CPU.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "torch.set_num_threads(N). Default 1 so parallel agent trains "
            "do not oversubscribe cores."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from an existing checkpoint in the artifact dir "
            "(skips warm-start; trains remaining episodes to --episodes)."
        ),
    )
    return parser


def resolve_pg_train_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve episode budget and agent kwargs from CLI flags."""
    bet_focus = bool(args.bet_focus)
    freeze_play = (
        PG_BET_FOCUS_FREEZE_PLAY
        if args.freeze_play is None and bet_focus
        else (PG_FREEZE_PLAY if args.freeze_play is None else bool(args.freeze_play))
    )
    bet_entropy = (
        PG_BET_FOCUS_BET_ENTROPY_COEF
        if args.bet_entropy_coef is None and bet_focus
        else (
            PG_BET_ENTROPY_COEF
            if args.bet_entropy_coef is None
            else float(args.bet_entropy_coef)
        )
    )
    play_entropy = (
        PG_BET_FOCUS_PLAY_ENTROPY_COEF
        if args.play_entropy_coef is None and bet_focus
        else (
            PG_PLAY_ENTROPY_COEF
            if args.play_entropy_coef is None
            else float(args.play_entropy_coef)
        )
    )
    teacher_ce = (
        PG_BET_FOCUS_TEACHER_BET_CE_COEF
        if args.teacher_bet_ce_coef is None and bet_focus
        else (
            PG_TEACHER_BET_CE_COEF
            if args.teacher_bet_ce_coef is None
            else float(args.teacher_bet_ce_coef)
        )
    )
    if args.episodes is not None:
        episodes = int(args.episodes)
    elif bet_focus:
        episodes = PG_BET_FOCUS_TRAINING_EPISODES
    else:
        episodes = PG_TRAINING_EPISODES

    if args.warmstart_episodes is not None:
        warmstart_episodes = int(args.warmstart_episodes)
    elif bet_focus:
        warmstart_episodes = PG_BET_FOCUS_WARMSTART_EPISODES
    else:
        warmstart_episodes = PG_WARMSTART_EPISODES

    if args.print_interval is not None:
        print_interval = int(args.print_interval)
    elif bet_focus:
        print_interval = PG_BET_FOCUS_PRINT_INTERVAL
    else:
        print_interval = PG_PRINT_INTERVAL

    if args.checkpoint_eval_episodes is not None:
        checkpoint_eval_episodes = int(args.checkpoint_eval_episodes)
    elif bet_focus:
        checkpoint_eval_episodes = PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES
    else:
        checkpoint_eval_episodes = PG_CHECKPOINT_EVAL_EPISODES

    if args.final_eval_episodes is not None:
        final_eval_episodes = int(args.final_eval_episodes)
    elif bet_focus:
        final_eval_episodes = PG_BET_FOCUS_FINAL_EVAL_EPISODES
    else:
        final_eval_episodes = PG_FINAL_EVAL_EPISODES

    if args.artifact_subdir is not None:
        artifact_subdir = str(args.artifact_subdir).strip() or None
    elif bet_focus:
        artifact_subdir = PG_BET_FOCUS_ARTIFACT_SUBDIR
    else:
        artifact_subdir = None

    return {
        "episodes": episodes,
        "warmstart_episodes": warmstart_episodes,
        "print_interval": print_interval,
        "checkpoint_eval_episodes": checkpoint_eval_episodes,
        "final_eval_episodes": final_eval_episodes,
        "artifact_subdir": artifact_subdir,
        "bet_focus": bet_focus,
        "agent_kwargs": {
            "freeze_play": freeze_play,
            "bet_entropy_coef": bet_entropy,
            "play_entropy_coef": play_entropy,
            "teacher_bet_ce_coef": teacher_ce,
        },
    }


def run_pg_training_loop(
    agent: Any,
    game: BlackjackGame,
    num_episodes: int,
    *,
    print_interval: int = PG_PRINT_INTERVAL,
    checkpoint_eval_episodes: int = PG_CHECKPOINT_EVAL_EPISODES,
    warmstart: bool | None = None,
    warmstart_episodes: int | None = None,
    learning_curve_path: Path | str | None = None,
    start_episode: int = 0,
    checkpoint_path: Path | str | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
) -> Any:
    """Train a bet+play PG agent with periodic greedy probes.

    When ``checkpoint_path`` is set, the model is saved at each probe (and at
    the end) with ``episodes_completed`` so ``--resume`` can continue later.
    ``start_episode`` is the number of RL episodes already finished (0 = fresh).
    """
    if start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {start_episode}")
    if start_episode > num_episodes:
        raise ValueError(
            f"start_episode ({start_episode}) exceeds num_episodes ({num_episodes})"
        )

    do_warmstart = PG_WARMSTART_ENABLED if warmstart is None else warmstart
    ws_episodes = (
        PG_WARMSTART_EPISODES if warmstart_episodes is None else warmstart_episodes
    )
    if start_episode == 0 and do_warmstart and ws_episodes > 0:
        warmstart_from_spread_rule(agent, game, ws_episodes)

    curve_logger = None
    if learning_curve_path is not None and PG_LEARNING_CURVE_ENABLED:
        curve_logger = LearningCurveLogger(learning_curve_path)

    def _save_progress(episodes_completed: int) -> None:
        if checkpoint_path is None:
            return
        extra = {
            "episodes_completed": int(episodes_completed),
            "target_episodes": int(num_episodes),
        }
        if checkpoint_extra:
            extra.update(checkpoint_extra)
        save_policy_checkpoint(agent, checkpoint_path, extra=extra)

    if start_episode >= num_episodes:
        print(
            f"Resume: already at {start_episode}/{num_episodes} episodes; nothing to do"
        )
        return agent

    agent.model.train()
    running = 0.0
    window = 0
    probe_every = max(1, int(print_interval))
    for episode in range(start_episode + 1, num_episodes + 1):
        reward = agent.train_one_episode(game)
        running += reward
        window += 1

        if episode % probe_every == 0 or episode == num_episodes:
            mean_train = running / max(1, window)
            running = 0.0
            window = 0
            if checkpoint_eval_episodes > 0:
                eval_mean, _ = evaluate_greedy(
                    agent,
                    checkpoint_eval_episodes,
                    reshuffle_threshold=game.reshuffle_threshold,
                )
            else:
                eval_mean = float("nan")
            print(
                f"Episode {episode}/{num_episodes}  "
                f"train_window={mean_train:.4f}  "
                f"greedy_eval={eval_mean:.4f}  "
                f"steps={agent.training_steps}"
            )
            if curve_logger is not None and checkpoint_eval_episodes > 0:
                curve_logger.append(
                    episode=episode,
                    training_steps=agent.training_steps,
                    eval_reward=eval_mean,
                    epsilon=0.0,
                    shoe_features_on=True,
                )
            _save_progress(episode)
            print(f"Checkpoint saved ({episode}/{num_episodes}): {checkpoint_path}")

    # Flush partial PPO rollout if any.
    if hasattr(agent, "update_from_rollout") and getattr(
        agent, "_episodes_in_rollout", 0
    ):
        agent.update_from_rollout()
        agent.clear_rollout()
        _save_progress(num_episodes)

    return agent


def run_pg_train_main(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    description: str,
) -> None:
    parser = build_pg_arg_parser(description)
    args = parser.parse_args()
    try:
        seeds = seeds_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    settings = resolve_pg_train_settings(args)
    multi = len(seeds) > 1
    aggregate: list[dict[str, Any]] = []
    agent_base = agent_results_path(agent_name, model_filename).parent
    if settings["artifact_subdir"]:
        agent_base = agent_base / settings["artifact_subdir"]

    for seed in seeds:
        set_seed(seed)
        out_dir = seed_artifact_dir(agent_base, seed, multi=multi)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / model_filename
        curve_path = out_dir / PG_LEARNING_CURVE_FILENAME

        factory_kwargs: dict[str, Any] = dict(settings["agent_kwargs"])
        if args.device is not None:
            factory_kwargs["device"] = args.device

        game = BlackjackGame(reshuffle_threshold=args.reshuffle_threshold)
        start_episode = 0
        resume_requested = bool(args.resume)
        if resume_requested and model_path.is_file():
            agent, payload = load_policy_checkpoint(
                agent_factory, model_path, **factory_kwargs
            )
            start_episode = int(payload.get("episodes_completed", 0))
            print(
                f"\n=== Resume seed={seed} from {model_path} "
                f"at episode {start_episode}/{settings['episodes']} ==="
            )
        elif resume_requested:
            print(
                f"\n=== Resume requested but no checkpoint at {model_path}; "
                "starting fresh ==="
            )
            agent = agent_factory(**factory_kwargs)
        else:
            agent = agent_factory(**factory_kwargs)
            print(f"\n=== Train seed={seed} artifacts={out_dir} ===")

        checkpoint_extra = {
            "freeze_play": bool(agent.freeze_play),
            "use_rule_play": bool(agent.use_rule_play),
            "bet_entropy_coef": float(agent.bet_entropy_coef),
            "play_entropy_coef": float(agent.play_entropy_coef),
            "teacher_bet_ce_coef": float(agent.teacher_bet_ce_coef),
            "bet_focus": settings["bet_focus"],
        }
        print(
            f"bet_focus={settings['bet_focus']}  "
            f"freeze_play={factory_kwargs['freeze_play']}  "
            f"bet_entropy={factory_kwargs['bet_entropy_coef']}  "
            f"teacher_ce={factory_kwargs['teacher_bet_ce_coef']}  "
            f"warmstart_episodes="
            f"{0 if (args.no_warmstart or start_episode > 0) else settings['warmstart_episodes']}  "
            f"print_interval={settings['print_interval']}  "
            f"checkpoint_eval={settings['checkpoint_eval_episodes']}  "
            f"final_eval={settings['final_eval_episodes']}  "
            f"start_episode={start_episode}"
        )
        run_pg_training_loop(
            agent,
            game,
            settings["episodes"],
            print_interval=settings["print_interval"],
            checkpoint_eval_episodes=settings["checkpoint_eval_episodes"],
            warmstart=not args.no_warmstart and start_episode == 0,
            warmstart_episodes=settings["warmstart_episodes"],
            learning_curve_path=curve_path,
            start_episode=start_episode,
            checkpoint_path=model_path,
            checkpoint_extra=checkpoint_extra,
        )
        print(f"Saved checkpoint: {model_path}")

        mean_reward = None
        distribution: dict[str, Any] = {}
        if settings["final_eval_episodes"] > 0:
            mean_reward, distribution = evaluate_greedy(
                agent,
                settings["final_eval_episodes"],
                seed=seed,
                reshuffle_threshold=args.reshuffle_threshold,
            )
            print(
                f"Final greedy eval ({settings['final_eval_episodes']} eps): "
                f"{mean_reward:.6f}"
            )
            print_distribution(distribution)

        summary = {
            "agent": agent_name,
            "seed": seed,
            "episodes": settings["episodes"],
            "episodes_completed": settings["episodes"],
            "resumed_from_episode": start_episode,
            "final_eval_episodes": settings["final_eval_episodes"],
            "mean_reward": mean_reward,
            "distribution": dict(distribution),
            "checkpoint": str(model_path),
            "warmstart": not args.no_warmstart and start_episode == 0,
            "warmstart_episodes": (
                0
                if args.no_warmstart or start_episode > 0
                else settings["warmstart_episodes"]
            ),
            "bet_focus": settings["bet_focus"],
            "freeze_play": bool(agent.freeze_play),
            "bet_entropy_coef": float(agent.bet_entropy_coef),
            "play_entropy_coef": float(agent.play_entropy_coef),
            "teacher_bet_ce_coef": float(agent.teacher_bet_ce_coef),
            "print_interval": settings["print_interval"],
            "checkpoint_eval_episodes": settings["checkpoint_eval_episodes"],
            "reshuffle_threshold": game.reshuffle_threshold,
        }
        summary_path = out_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        aggregate.append(summary)

    if multi:
        agg_path = agent_base / "multi_seed_train_summary.json"
        agg_path.write_text(json.dumps({"runs": aggregate}, indent=2) + "\n")
        print(f"Wrote multi-seed summary: {agg_path}")


def train_pg_agent(
    *,
    agent_name: str,
    model_filename: str,
    agent_factory: Callable[..., Any],
    num_episodes: int = PG_TRAINING_EPISODES,
    warmstart: bool | None = None,
    warmstart_episodes: int | None = None,
    learning_curve_path: Path | str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    reshuffle_threshold: int | None = None,
) -> Any:
    """Programmatic trainer used by tests and notebooks."""
    game = BlackjackGame(reshuffle_threshold=reshuffle_threshold)
    agent = agent_factory(**(agent_kwargs or {}))
    curve = learning_curve_path
    if curve is None:
        curve = agent_results_path(agent_name, PG_LEARNING_CURVE_FILENAME)
    run_pg_training_loop(
        agent,
        game,
        num_episodes,
        warmstart=warmstart,
        warmstart_episodes=warmstart_episodes,
        learning_curve_path=curve,
    )
    save_policy_checkpoint(
        agent,
        agent_results_path(agent_name, model_filename),
        extra={
            "freeze_play": bool(agent.freeze_play),
            "use_rule_play": bool(agent.use_rule_play),
            "bet_entropy_coef": float(agent.bet_entropy_coef),
            "play_entropy_coef": float(agent.play_entropy_coef),
            "teacher_bet_ce_coef": float(agent.teacher_bet_ce_coef),
        },
    )
    return agent
