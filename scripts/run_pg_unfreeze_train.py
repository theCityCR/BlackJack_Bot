#!/usr/bin/env python3
"""Launch unfreeze-after-bet-focus PG trains for reinforce / a2c / ppo.

Each agent starts from ``agents/results/<agent>/bet_focus/`` weights, thaws
the play head, and writes mid-run checkpoints under
``agents/results/<agent>/unfreeze/``.

By default ``--detach`` starts each trainer in its own session and returns
immediately so a Cursor/agent shell abort does not kill the children.
SIGINT/SIGTERM on a trainer finishes the current episode and saves; continue
later with ``--resume`` (or simply re-run this script — existing unfreeze
checkpoints auto-continue).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("reinforce", "a2c", "ppo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--agents",
        default=",".join(AGENTS),
        help="Comma-separated subset of reinforce,a2c,ppo",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "agents" / "results",
        help="Directory for per-agent train logs and pid files",
    )
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start trainers in new sessions and exit (default: on). "
            "Disable with --no-detach to wait for completion."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to each trainer (continue from saved checkpoints).",
    )
    args = parser.parse_args()

    names = [p.strip() for p in args.agents.split(",") if p.strip()]
    for name in names:
        if name not in AGENTS:
            parser.error(f"unknown agent {name!r}")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["OMP_NUM_THREADS"] = "1"

    procs: list[tuple[str, subprocess.Popen[str], Path]] = []
    pid_payload: dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "resume": bool(args.resume),
        "detach": bool(args.detach),
        "mode": "unfreeze",
        "agents": {},
    }
    for name in names:
        mode = "a" if args.resume else "w"
        log_path = args.log_dir / f"pg_unfreeze_{name}.log"
        cmd = [
            sys.executable,
            "-m",
            f"agents.train_{name}",
            "--unfreeze",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--torch-threads",
            "1",
        ]
        if args.resume:
            cmd.append("--resume")
        log_file = log_path.open(mode, encoding="utf-8")
        if args.resume:
            log_file.write(
                f"\n===== RESUME {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n"
            )
            log_file.flush()
        print(f"start {name} -> {log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        procs.append((name, proc, log_path))
        pid_payload["agents"][name] = {
            "pid": proc.pid,
            "log": str(log_path),
            "checkpoint": str(
                ROOT
                / "agents"
                / "results"
                / name
                / "unfreeze"
                / f"{name}_bet_play_model.pt"
            ),
            "init_from": str(
                ROOT
                / "agents"
                / "results"
                / name
                / "bet_focus"
                / f"{name}_bet_play_model.pt"
            ),
        }
        time.sleep(0.5)

    pid_path = args.log_dir / "pg_unfreeze_pids.json"
    pid_path.write_text(json.dumps(pid_payload, indent=2) + "\n")
    print(f"Wrote {pid_path}")

    if args.detach:
        print(
            "Detached: trainers are in independent sessions. "
            "Safe to close Cursor or shut down after a probe save "
            "(every 25k episodes, or immediately on SIGTERM). "
            "Monitor agents/results/pg_unfreeze_*.log; "
            "continue with: python3 scripts/run_pg_unfreeze_train.py --resume"
        )
        return

    failures = 0
    for name, proc, log_path in procs:
        code = proc.wait()
        print(f"{name} exit={code} log={log_path}")
        if code != 0:
            failures += 1

    if failures:
        raise SystemExit(1)
    print("all unfreeze trains finished")


if __name__ == "__main__":
    main()
