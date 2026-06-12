"""Configuration loading (Phase 0).

Loads secrets from ``.env`` (via python-dotenv) and non-secret settings from
``config/settings.yaml``. Secrets never live in the YAML; settings never live in
``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo root = three levels up from this file (src/ict_bot/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"
ENV_PATH = REPO_ROOT / ".env"


def load_env(env_path: Path = ENV_PATH) -> None:
    """Load environment variables from ``.env`` if present (no-op if missing)."""
    load_dotenv(env_path)


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Parse ``config/settings.yaml`` into a dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
