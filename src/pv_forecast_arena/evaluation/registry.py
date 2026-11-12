"""Experiment registry: one row per run and variant, appended to results/experiments.csv."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path("results/experiments.csv")


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def git_state() -> tuple[str, bool]:
    """Short commit hash and whether the working tree has uncommitted changes.

    The registry file itself is ignored: earlier runs of a batch append to it before later ones start.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", ".", f":!{REGISTRY_PATH.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return commit, dirty


def new_run_id(name: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{name}"


def append(rows: list[dict], path: Path = REGISTRY_PATH) -> pd.DataFrame:
    """Append rows to the registry (creating it), keeping the union of columns."""
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new
