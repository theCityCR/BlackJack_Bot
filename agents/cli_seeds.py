"""Shared CLI helpers for single- and multi-seed experiment scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path


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
