"""Configuration utilities for the project."""

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    # Assumes this file is at src/utils/config.py
    return Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default configs/config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_data_path(subpath: str = "") -> Path:
    """Get path to data directory.

    Args:
        subpath: Optional subdirectory within data/

    Returns:
        Path object
    """
    return get_project_root() / "data" / subpath


def get_results_path(subpath: str = "") -> Path:
    """Get path to results directory.

    Args:
        subpath: Optional subdirectory within results/

    Returns:
        Path object
    """
    return get_project_root() / "results" / subpath


# Load config on import for convenience
CONFIG = load_config()
