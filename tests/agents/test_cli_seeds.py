"""Tests for multi-seed CLI scaffolding helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agents.cli_seeds import (
    add_seed_arguments,
    parse_seeds,
    seed_artifact_dir,
    seeds_from_args,
)


def test_parse_seeds_defaults_to_single_seed():
    assert parse_seeds(seed=42) == [42]
    assert parse_seeds(seed=7, seeds=None) == [7]


def test_parse_seeds_comma_list_overrides_seed():
    assert parse_seeds(seed=42, seeds="10, 20,30") == [10, 20, 30]


def test_parse_seeds_rejects_empty_and_non_int():
    with pytest.raises(ValueError, match="non-empty"):
        parse_seeds(seeds="")
    with pytest.raises(ValueError, match="non-empty"):
        parse_seeds(seeds="42,")
    with pytest.raises(ValueError, match="integers"):
        parse_seeds(seeds="42,x")


def test_seed_artifact_dir_keeps_legacy_path_for_single_seed(tmp_path: Path):
    assert seed_artifact_dir(tmp_path, 42, multi=False) == tmp_path
    assert seed_artifact_dir(tmp_path, 42, multi=True) == tmp_path / "seed_42"


def test_seeds_from_args_prefers_seeds_flag():
    parser = argparse.ArgumentParser()
    add_seed_arguments(parser)
    args = parser.parse_args(["--seed", "1", "--seeds", "2,3"])
    assert seeds_from_args(args) == [2, 3]
