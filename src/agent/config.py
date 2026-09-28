"""Project settings.

Settings live in ``config/settings.yaml`` (committed defaults) and
``config/runtime.yaml`` (workspace-specific values written by
``notebooks/00_setup.py``). The runtime file wins where both define a key.

When the agent is served, the same dictionary is passed to MLflow as the
``model_config``, so the served agent and the notebooks read identical values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Settings:
    """Thin wrapper around the settings dictionary with naming helpers."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def catalog(self) -> str:
        return self.data["catalog"]

    @property
    def schema(self) -> str:
        return self.data["schema"]

    def fq(self, name: str) -> str:
        """Fully qualified Unity Catalog name for an object in the project schema."""
        return f"{self.catalog}.{self.schema}.{name}"

    def table(self, key: str) -> str:
        """Fully qualified name of a table defined under ``tables:``."""
        return self.fq(self.data["tables"][key])

    @property
    def silver_tables(self) -> list[str]:
        """Short names of the tables the Text2SQL tool may read."""
        t = self.data["tables"]
        return [t["transactions"], t["customers"], t["franchises"], t["suppliers"]]

    @property
    def index_name(self) -> str:
        return self.fq(self.data["ai_search"]["index_name"])

    @property
    def model_name(self) -> str:
        return self.fq(self.data["serving"]["model_name"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def load_settings(config_dir: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load committed defaults, then runtime values, then explicit overrides."""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    with open(config_dir / "settings.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runtime_path = config_dir / "runtime.yaml"
    if runtime_path.exists():
        with open(runtime_path, encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings(data)


def write_runtime_values(values: dict[str, Any], config_dir: str | Path | None = None) -> Path:
    """Merge ``values`` into ``config/runtime.yaml`` (used by the setup notebook)."""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    path = config_dir / "runtime.yaml"
    existing: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_deep_merge(existing, values), f, sort_keys=False)
    return path
