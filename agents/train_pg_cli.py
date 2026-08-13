"""Shared CLI helpers for bet+play policy-gradient trainers."""

from __future__ import annotations

import argparse
import json
import signal
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
    PG_UNFREEZE_ARTIFACT_SUBDIR,
    PG_UNFREEZE_BET_ENTROPY_COEF,
    PG_UNFREEZE_CHECKPOINT_EVAL_EPISODES,
    PG_UNFREEZE_FINAL_EVAL_EPISODES,
    PG_UNFREEZE_FREEZE_PLAY,
    PG_UNFREEZE_INIT_SUBDIR,
    PG_UNFREEZE_PLAY_ENTROPY_COEF,
    PG_UNFREEZE_PRINT_INTERVAL,
    PG_UNFREEZE_TEACHER_BET_CE_COEF,
    PG_UNFREEZE_TRAINING_EPISODES,
    PG_UNFREEZE_WARMSTART_EPISODES,
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
            f"{PG_BET_FOCUS_WARMSTART_EPISODES} with --bet-focus; "
            f"{PG_UNFREEZE_WARMSTART_EPISODES} with --unfreeze)."
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
        "--unfreeze",
        action="store_true",
        help=(
            "Phase-2 preset: load bet_focus weights, thaw play, keep teacher "
            "bet CE; checkpoints under agents/results/<agent>/unfreeze/."
        ),
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "Checkpoint to initialize from when starting a fresh --unfreeze "
            "run (default: agents/results/<agent>/bet_focus/<model>)."
        ),
    )
    parser.add_argument(
        "--freeze-play",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Train bet head only; rule chart plays "
            "(default: on with --bet-focus, off with --unfreeze)."
        ),
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
            f"{PG_BET_FOCUS_PRINT_INTERVAL} with --bet-focus; "
            f"{PG_UNFREEZE_PRINT_INTERVAL} with --unfreeze)."
        ),
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=None,
        help=(
            "Greedy episodes per mid-run probe "
            f"(default {PG_CHECKPOINT_EVAL_EPISODES}; "
            f"{PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES} with --bet-focus; "
            f"{PG_UNFREEZE_CHECKPOINT_EVAL_EPISODES} with --unfreeze)."
        ),
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=None,
        help=(
            "Greedy episodes after training (0 skips). "
            f"Default {PG_FINAL_EVAL_EPISODES}; "
            f"{PG_BET_FOCUS_FINAL_EVAL_EPISODES} with --bet-focus; "
            f"{PG_UNFREEZE_FINAL_EVAL_EPISODES} with --unfreeze."
        ),
    )
    parser.add_argument(
        "--artifact-subdir",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under agents/results/<agent>/ "
            f"(default none; '{PG_BET_FOCUS_ARTIFACT_SUBDIR}' with --bet-focus; "
            f"'{PG_UNFREEZE_ARTIFACT_SUBDIR}' with --unfreeze)."
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
    unfreeze = bool(getattr(args, "unfreeze", False))
    if bet_focus and unfreeze:
        raise ValueError("--bet-focus and --unfreeze are mutually exclusive")

    if args.freeze_play is None:
        if bet_focus:
            freeze_play = PG_BET_FOCUS_FREEZE_PLAY
        elif unfreeze:
            freeze_play = PG_UNFREEZE_FREEZE_PLAY
        else:
            freeze_play = PG_FREEZE_PLAY
    else:
        freeze_play = bool(args.freeze_play)

    if args.bet_entropy_coef is None:
        if bet_focus:
            bet_entropy = PG_BET_FOCUS_BET_ENTROPY_COEF
        elif unfreeze:
            bet_entropy = PG_UNFREEZE_BET_ENTROPY_COEF
        else:
            bet_entropy = PG_BET_ENTROPY_COEF
    else:
        bet_entropy = float(args.bet_entropy_coef)

    if args.play_entropy_coef is None:
        if bet_focus:
            play_entropy = PG_BET_FOCUS_PLAY_ENTROPY_COEF
        elif unfreeze:
            play_entropy = PG_UNFREEZE_PLAY_ENTROPY_COEF
        else:
            play_entropy = PG_PLAY_ENTROPY_COEF
    else:
        play_entropy = float(args.play_entropy_coef)

    if args.teacher_bet_ce_coef is None:
        if bet_focus:
            teacher_ce = PG_BET_FOCUS_TEACHER_BET_CE_COEF
        elif unfreeze:
            teacher_ce = PG_UNFREEZE_TEACHER_BET_CE_COEF
        else:
            teacher_ce = PG_TEACHER_BET_CE_COEF
    else:
        teacher_ce = float(args.teacher_bet_ce_coef)

    if args.episodes is not None:
        episodes = int(args.episodes)
    elif bet_focus:
        episodes = PG_BET_FOCUS_TRAINING_EPISODES
    elif unfreeze:
        episodes = PG_UNFREEZE_TRAINING_EPISODES
    else:
        episodes = PG_TRAINING_EPISODES

    if args.warmstart_episodes is not None:
        warmstart_episodes = int(args.warmstart_episodes)
    elif bet_focus:
        warmstart_episodes = PG_BET_FOCUS_WARMSTART_EPISODES
    elif unfreeze:
        warmstart_episodes = PG_UNFREEZE_WARMSTART_EPISODES
    else:
        warmstart_episodes = PG_WARMSTART_EPISODES

    if args.print_interval is not None:
        print_interval = int(args.print_interval)
    elif bet_focus:
        print_interval = PG_BET_FOCUS_PRINT_INTERVAL
    elif unfreeze:
        print_interval = PG_UNFREEZE_PRINT_INTERVAL
    else:
        print_interval = PG_PRINT_INTERVAL

    if args.checkpoint_eval_episodes is not None:
        checkpoint_eval_episodes = int(args.checkpoint_eval_episodes)
    elif bet_focus:
        checkpoint_eval_episodes = PG_BET_FOCUS_CHECKPOINT_EVAL_EPISODES
    elif unfreeze:
        checkpoint_eval_episodes = PG_UNFREEZE_CHECKPOINT_EVAL_EPISODES
    else:
        checkpoint_eval_episodes = PG_CHECKPOINT_EVAL_EPISODES

    if args.final_eval_episodes is not None:
        final_eval_episodes = int(args.final_eval_episodes)
    elif bet_focus:
        final_eval_episodes = PG_BET_FOCUS_FINAL_EVAL_EPISODES
    elif unfreeze:
        final_eval_episodes = PG_UNFREEZE_FINAL_EVAL_EPISODES
    else:
        final_eval_episodes = PG_FINAL_EVAL_EPISODES

    if args.artifact_subdir is not None:
        artifact_subdir = str(args.artifact_subdir).strip() or None
    elif bet_focus:
        artifact_subdir = PG_BET_FOCUS_ARTIFACT_SUBDIR
    elif unfreeze:
        artifact_subdir = PG_UNFREEZE_ARTIFACT_SUBDIR
    else:
        artifact_subdir = None

    init_from = getattr(args, "init_from", None)
    init_from_path = str(init_from).strip() if init_from else None

    return {
        "episodes": episodes,
        "warmstart_episodes": warmstart_episodes,
        "print_interval": print_interval,
        "checkpoint_eval_episodes": checkpoint_eval_episodes,
        "final_eval_episodes": final_eval_episodes,
        "artifact_subdir": artifact_subdir,
        "bet_focus": bet_focus,
        "unfreeze": unfreeze,
        "init_from": init_from_path,
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
) -> tuple[Any, int]:
    """Train a bet+play PG agent with periodic greedy probes.

    When ``checkpoint_path`` is set, the model is saved at each probe (and at
    the end) with ``episodes_completed`` so ``--resume`` can continue later.
    ``start_episode`` is the number of RL episodes already finished (0 = fresh).

    SIGINT / SIGTERM finish the current episode, save a checkpoint, and return
    so a shutdown or ``kill`` does not lose progress.

    Returns ``(agent, episodes_completed)``.
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

    stop_requested = {"value": False}
    previous_handlers: dict[int, Any] = {}

    def _request_stop(signum: int, _frame: Any) -> None:
        stop_requested["value"] = True
        print(
            f"\nReceived signal {signum}; "
            "finishing current episode then saving checkpoint for --resume…"
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            # Signals unavailable (e.g. non-main thread); periodic probes still save.
            pass

    if start_episode >= num_episodes:
        print(
            f"Resume: already at {start_episode}/{num_episodes} episodes; nothing to do"
        )
        return agent, start_episode

    agent.model.train()
    running = 0.0
    window = 0
    probe_every = max(1, int(print_interval))
    paused_early = False
    last_episode = start_episode
    try:
        for episode in range(start_episode + 1, num_episodes + 1):
            reward = agent.train_one_episode(game)
            running += reward
            window += 1
            last_episode = episode

            should_probe = episode % probe_every == 0 or episode == num_episodes
            if should_probe:
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

            if stop_requested["value"]:
                if hasattr(agent, "update_from_rollout") and getattr(
                    agent, "_episodes_in_rollout", 0
                ):
                    agent.update_from_rollout()
                    agent.clear_rollout()
                _save_progress(episode)
                print(
                    f"Paused at episode {episode}/{num_episodes}. "
                    f"Resume with --resume (checkpoint: {checkpoint_path})"
                )
                paused_early = True
                break

        if not paused_early:
            # Flush partial PPO rollout if any.
            if hasattr(agent, "update_from_rollout") and getattr(
                agent, "_episodes_in_rollout", 0
            ):
                agent.update_from_rollout()
                agent.clear_rollout()
                _save_progress(num_episodes)
                last_episode = num_episodes
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    return agent, last_episode


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

    try:
        settings = resolve_pg_train_settings(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

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
        initialized_from: str | None = None

        if model_path.is_file() and (
            resume_requested or settings["unfreeze"]
        ):
            agent, payload = load_policy_checkpoint(
                agent_factory, model_path, **factory_kwargs
            )
            start_episode = int(payload.get("episodes_completed", 0))
            if settings["unfreeze"] and not resume_requested:
                print(
                    f"\n=== Found existing unfreeze checkpoint {model_path}; "
                    f"continuing at episode {start_episode}/"
                    f"{settings['episodes']} (pass a fresh --artifact-subdir "
                    "to restart from bet_focus) ==="
                )
            else:
                print(
                    f"\n=== Resume seed={seed} from {model_path} "
                    f"at episode {start_episode}/{settings['episodes']} ==="
                )
        elif settings["unfreeze"]:
            init_path = (
                Path(settings["init_from"])
                if settings["init_from"]
                else (
                    agent_results_path(agent_name, model_filename).parent
                    / PG_UNFREEZE_INIT_SUBDIR
                    / model_filename
                )
            )
            if not init_path.is_file():
                parser.error(
                    f"--unfreeze requires an init checkpoint at {init_path} "
                    "(train --bet-focus first, or pass --init-from)"
                )
            agent, _payload = load_policy_checkpoint(
                agent_factory, init_path, **factory_kwargs
            )
            start_episode = 0
            initialized_from = str(init_path)
            print(
                f"\n=== Unfreeze seed={seed} init_from={init_path} "
                f"artifacts={out_dir} ==="
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
            "unfreeze": settings["unfreeze"],
        }
        if initialized_from:
            checkpoint_extra["initialized_from"] = initialized_from
        print(
            f"bet_focus={settings['bet_focus']}  "
            f"unfreeze={settings['unfreeze']}  "
            f"freeze_play={factory_kwargs['freeze_play']}  "
            f"bet_entropy={factory_kwargs['bet_entropy_coef']}  "
            f"play_entropy={factory_kwargs['play_entropy_coef']}  "
            f"teacher_ce={factory_kwargs['teacher_bet_ce_coef']}  "
            f"warmstart_episodes="
            f"{0 if (args.no_warmstart or start_episode > 0 or settings['unfreeze']) else settings['warmstart_episodes']}  "
            f"print_interval={settings['print_interval']}  "
            f"checkpoint_eval={settings['checkpoint_eval_episodes']}  "
            f"final_eval={settings['final_eval_episodes']}  "
            f"start_episode={start_episode}"
        )
        agent, episodes_completed = run_pg_training_loop(
            agent,
            game,
            settings["episodes"],
            print_interval=settings["print_interval"],
            checkpoint_eval_episodes=settings["checkpoint_eval_episodes"],
            warmstart=(
                not args.no_warmstart
                and start_episode == 0
                and not settings["unfreeze"]
            ),
            warmstart_episodes=settings["warmstart_episodes"],
            learning_curve_path=curve_path,
            start_episode=start_episode,
            checkpoint_path=model_path,
            checkpoint_extra=checkpoint_extra,
        )
        print(f"Saved checkpoint: {model_path}")
        finished = episodes_completed >= settings["episodes"]

        mean_reward = None
        distribution: dict[str, Any] = {}
        if finished and settings["final_eval_episodes"] > 0:
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
        elif not finished:
            print(
                f"Training incomplete ({episodes_completed}/"
                f"{settings['episodes']}); skipping final eval. "
                "Re-run with --resume to continue."
            )

        summary = {
            "agent": agent_name,
            "seed": seed,
            "episodes": settings["episodes"],
            "episodes_completed": episodes_completed,
            "finished": finished,
            "resumed_from_episode": start_episode,
            "final_eval_episodes": settings["final_eval_episodes"],
            "mean_reward": mean_reward,
            "distribution": dict(distribution),
            "checkpoint": str(model_path),
            "warmstart": (
                not args.no_warmstart
                and start_episode == 0
                and not settings["unfreeze"]
            ),
            "warmstart_episodes": (
                0
                if args.no_warmstart or start_episode > 0 or settings["unfreeze"]
                else settings["warmstart_episodes"]
            ),
            "bet_focus": settings["bet_focus"],
            "unfreeze": settings["unfreeze"],
            "initialized_from": initialized_from,
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

    if any(not row.get("finished", True) for row in aggregate):
        raise SystemExit(2)


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
