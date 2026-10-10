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


def http_timeout(kind: str = "inference") -> tuple[float, float]:
    """Requests connect/read-idle budgets, not a total execution deadline."""
    defaults = {"inference": (5, 120), "preflight": (5, 15), "heartbeat": (5, 15)}
    connect, read = defaults[kind]
    prefix = f"COLABCLUSTER_{kind.upper()}"
    return (positive_seconds(prefix + "_CONNECT_TIMEOUT", connect),
            positive_seconds(prefix + "_READ_TIMEOUT", read))
