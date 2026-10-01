"""Small shared helpers for validated environment configuration."""

import math
import os


def positive_seconds(name: str, default: float) -> float:
    """Read a finite, positive duration, rejecting invalid configuration early."""
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number greater than zero") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value
