"""
SINGLE source of truth for all parameters.
"""
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {path}, using empty defaults")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
