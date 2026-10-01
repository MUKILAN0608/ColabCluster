"""Periodic heartbeats with one attempt per interval and no re-registration."""

import logging
import time

from common.config import positive_seconds
from worker.client import WorkerClient, WorkerClientError
from worker.worker import Worker


def get_heartbeat_interval() -> float:
    """Read the heartbeat period, defaulting to ten seconds."""
    return positive_seconds("COLABCLUSTER_HEARTBEAT_INTERVAL", 10)


def run_heartbeat_loop(worker: Worker, client: WorkerClient, interval: float) -> None:
    """Continue after temporary HTTP errors; Ctrl+C exits to caller cleanup."""
    logger = logging.getLogger(__name__)
    while True:
        try:
            client.heartbeat(worker)
            logger.info("[Heartbeat] Sent successfully: %s", worker.worker_id)
        except WorkerClientError as exc:
            logger.warning("[Heartbeat] %s; retrying on next interval", exc)
        time.sleep(interval)
