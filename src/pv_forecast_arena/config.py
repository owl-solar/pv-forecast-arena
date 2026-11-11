"""Load YAML configs; experiment configs extend configs/base.yaml."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("extends", None)
    if base:
        parent = load_config(path.parent / base)
        parent.update(cfg)
        cfg = parent
    return cfg
