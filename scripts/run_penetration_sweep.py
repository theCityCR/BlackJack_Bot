#!/usr/bin/env python3
"""Sweep reshuffle cut cards and compare rule flat vs Hi-Lo spread EV.

Lower ``--thresholds`` ⇒ deeper dealt penetration ⇒ more time at extreme
true counts. Default study cut remains 26 (≈75% on a 2-deck shoe).

Does not modify published flat-bet or §5.5 artifacts under docs/results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cards import dealt_penetration, shoe_size
from config import NUM_DECKS, RESHUFFLE_WHEN_CARDS_REMAINING_BELOW
import scripts.run_variable_betting_eval as vb_eval

DEFAULT_THRESHOLDS = (13, 26, 39, 52)
DEFAULT_OUTPUT = Path("agents/results/variable_betting/penetration_sweep.json")
SMOKE_THRESHOLDS = (26, 52)


def parse_thresholds(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("thresholds must be a non-empty comma-separated list")
    values = [int(p) for p in parts]
    size = shoe_size(NUM_DECKS)
    for value in values:
        if not 0 <= value <= size:
            raise ValueError(
                f"threshold {value} outside shoe size 0..{size} ({NUM_DECKS} decks)"
            )
    return values


def sweep_thresholds(
    thresholds: list[int],
    *,
    episodes: int,
    seed: int,
    rounds_per_shoe: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cut in thresholds:
        summary = vb_eval.run_comparison(
            episodes,
            seed,
            rounds_per_shoe=rounds_per_shoe,
            reshuffle_threshold=cut,
        )
        vb_eval.print_comparison(summary)
        print("---")
        row = vb_eval.compact_run_row(summary)
        rows.append(row)

    return {
        "seeds": [seed],
        "episodes": episodes,
        "rounds_per_shoe": rounds_per_shoe,
        "num_decks": NUM_DECKS,
        "default_reshuffle_threshold": RESHUFFLE_WHEN_CARDS_REMAINING_BELOW,
        "thresholds": thresholds,
        "runs": rows,
        "artifact_note": (
            "Penetration sweep for rule + Hi-Lo spread. "
            "Copy to docs/results/ only when publishing; "
            "does not modify published study tables."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rounds-per-shoe",
        type=int,
        default=vb_eval.DEFAULT_ROUNDS_PER_SHOE,
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help=(
            "Comma-separated remaining-card cut cards to sweep "
            f"(default {','.join(str(t) for t in DEFAULT_THRESHOLDS)})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"JSON path (default {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short CI sweep (500 episodes, thresholds 26 and 52)",
    )
    args = parser.parse_args()

    try:
        thresholds = (
            list(SMOKE_THRESHOLDS)
            if args.smoke
            else parse_thresholds(args.thresholds)
        )
    except ValueError as exc:
        parser.error(str(exc))

    episodes = 500 if args.smoke else args.episodes
    payload = sweep_thresholds(
        thresholds,
        episodes=episodes,
        seed=args.seed,
        rounds_per_shoe=args.rounds_per_shoe,
    )
    payload["smoke"] = bool(args.smoke)

    out = args.output or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")
    print(
        "cut  pen     flat EV   spread EV  delta EV"
    )
    for row in payload["runs"]:
        cut = row["reshuffle_threshold"]
        pen = dealt_penetration(cut, NUM_DECKS)
        print(
            f"{cut:>3}  {pen:5.1%}  "
            f"{row['flat_average_reward']:+8.4f}  "
            f"{row['spread_average_reward']:+9.4f}  "
            f"{row['delta_average_reward']:+8.4f}"
        )


if __name__ == "__main__":
    main()
