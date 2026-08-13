#!/usr/bin/env python3
"""Launch bet-focus PG trains for reinforce / a2c / ppo in parallel.

Each agent is single-threaded (OMP/torch threads = 1) so three processes can
saturate three cores without fighting. Artifacts land under
``agents/results/<agent>/bet_focus/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

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
        help="Directory for per-agent train logs",
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
    for name in names:
        log_path = args.log_dir / f"pg_bet_focus_{name}.log"
        cmd = [
            sys.executable,
            "-m",
            f"agents.train_{name}",
            "--bet-focus",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--torch-threads",
            "1",
        ]
        log_file = log_path.open("w", encoding="utf-8")
        print(f"start {name} -> {log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((name, proc, log_path))
        # Stagger slightly so first prints do not interleave on a shared disk.
        time.sleep(0.5)

    failures = 0
    for name, proc, log_path in procs:
        code = proc.wait()
        print(f"{name} exit={code} log={log_path}")
        if code != 0:
            failures += 1

    if failures:
        raise SystemExit(1)
    print("all bet-focus trains finished")


if __name__ == "__main__":
    main()
