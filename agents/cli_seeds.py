"""Shared CLI helpers for single- and multi-seed experiment scaffolding."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any


def add_seed_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--seed`` and ``--seeds`` flags (``--seeds`` wins when both set)."""
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when --seeds is omitted (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated RNG seeds (e.g. 42,43,44). When set, overrides "
            "--seed. Multiple seeds write under seed_<n>/ subdirectories."
        ),
    )


def parse_seeds(
    *,
    seed: int = 42,
    seeds: str | None = None,
) -> list[int]:
    """Return a non-empty seed list from ``--seed`` / ``--seeds``.

    If ``seeds`` is provided it wins; otherwise ``[seed]``.
    """
    if seeds is None:
        return [seed]

    parts = [part.strip() for part in seeds.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--seeds must be a non-empty comma-separated list of integers")

    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise ValueError(
                f"--seeds entries must be integers (got {part!r})"
            ) from exc
    if not parsed:
        raise ValueError("--seeds must include at least one integer")
    return parsed


def seeds_from_args(args: argparse.Namespace) -> list[int]:
    """Parse seeds from an argparse namespace that used :func:`add_seed_arguments`."""
    return parse_seeds(seed=args.seed, seeds=args.seeds)


def seed_artifact_dir(base: Path, seed: int, *, multi: bool) -> Path:
    """Legacy path for a single seed; ``base/seed_<n>`` when multi-seed."""
    if not multi:
        return base
    return base / f"seed_{seed}"


def metric_stats(values: list[float]) -> dict[str, float]:
    """Mean / sample std / min / max for a metric across seeds.

    With a single value, ``std`` is 0.0 (no population variance to estimate).
    """
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    mean = statistics.fmean(values)
    if n == 1:
        std = 0.0
    else:
        std = statistics.stdev(values)
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "n": n,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _stats_for_keys(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    return {
        key: metric_stats([float(row[key]) for row in rows])
        for key in keys
    }


def summarize_ablation_runs(
    rows: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Per-condition mean/std over ablation result rows from multiple seeds."""
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        condition_id = row["condition_id"]
        by_condition.setdefault(condition_id, []).append(row)

    conditions: list[dict[str, Any]] = []
    for condition_id, condition_rows in by_condition.items():
        label = condition_rows[0].get("label", condition_id)
        metrics = _stats_for_keys(
            condition_rows,
            ["average_reward", "win_rate", "loss_rate", "draw_rate"],
        )
        conditions.append(
            {
                "condition_id": condition_id,
                "label": label,
                "n_seeds": len(condition_rows),
                **metrics,
            }
        )

    return {
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "conditions": conditions,
    }


def summarize_gap_close_runs(
    rows: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Mean/std over gap-close multi-seed run rows."""
    metrics = _stats_for_keys(
        rows,
        ["agent_average_reward", "rule_average_reward", "gap"],
    )
    return {
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        **metrics,
    }


def summarize_benchmark_runs(
    aggregate: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Per-agent mean/std of ``average_reward`` across evaluate_agents runs."""
    by_agent: dict[str, list[float]] = {}
    for run in aggregate:
        for result in run.get("results", []):
            name = result["agent"]
            by_agent.setdefault(name, []).append(float(result["average_reward"]))

    agents = [
        {
            "agent": name,
            "average_reward": metric_stats(values),
        }
        for name, values in by_agent.items()
    ]
    return {
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "agents": agents,
    }

