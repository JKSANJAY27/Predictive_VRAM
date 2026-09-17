"""
Configuration loader for model and runtime baseline settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Safely load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_default_config_path(filename: str) -> Path:
    """Resolve default config path relative to workspace root."""
    root = Path(__file__).resolve().parent.parent.parent
    return root / "configs" / filename
