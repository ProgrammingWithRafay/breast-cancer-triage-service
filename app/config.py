"""Service configuration, read only from environment variables."""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


@dataclass(frozen=True)
class Settings:
    model_path: Path
    decision_threshold: float
    log_level: str
    model_version: str | None  # None -> use model_version from the model card
    git_sha: str


def _parse_threshold(raw: str) -> float:
    """Parse DECISION_THRESHOLD and require a finite value in [0, 1]."""
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(
            f"DECISION_THRESHOLD must be a number between 0 and 1, got {raw!r}"
        ) from None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"DECISION_THRESHOLD must be between 0 and 1, got {raw!r}")
    return value


def _parse_log_level(raw: str) -> str:
    """Normalise LOG_LEVEL and require a standard logging level name."""
    level = raw.strip().upper()
    if not isinstance(logging.getLevelName(level), int):
        raise ConfigError(
            f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL, got {raw!r}"
        )
    return level


def load_settings() -> Settings:
    """Build Settings from the environment; raises ConfigError on bad values."""
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "artifacts/model.joblib")),
        decision_threshold=_parse_threshold(os.getenv("DECISION_THRESHOLD", "0.5")),
        log_level=_parse_log_level(os.getenv("LOG_LEVEL", "INFO")),
        model_version=os.getenv("MODEL_VERSION") or None,
        git_sha=os.getenv("GIT_SHA", "unknown"),
    )
