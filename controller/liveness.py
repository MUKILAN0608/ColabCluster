"""Lightweight liveness monitor owned by the FastAPI application lifespan."""

import asyncio

from controller.registry import WorkerRegistry


async def monitor_workers(registry: WorkerRegistry, timeout: float) -> None:
    """Remove stale workers at most every five seconds until cancelled."""
    interval = min(5.0, timeout / 2)
    while True:
        await asyncio.sleep(interval)
        registry.expire_stale_workers(timeout)
