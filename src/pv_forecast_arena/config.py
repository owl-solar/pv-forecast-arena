"""Load YAML configs; experiment configs extend configs/base.yaml."""

from pathlib import Path

import yaml


def _merge(parent: dict, child: dict) -> dict:
    """Child keys win; nested mappings merge, so a run can override a single `evaluation` key."""
    out = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("extends", None)
    if base:
        parent = load_config(path.parent / base)
        cfg = _merge(parent, cfg)
    return cfg
