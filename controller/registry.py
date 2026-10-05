"""Thread-safe, process-local worker storage."""

from datetime import datetime, timezone
from collections import Counter
from math import fsum
from threading import Lock
import logging

from common.schemas import ClusterSummary, WorkerInfo, WorkerRegistration, WorkerStatus


class DuplicateWorkerError(Exception):
    """A worker ID is already registered."""


class WorkerNotReadyError(Exception):
    """The selected worker is already busy or not ready."""


class UnknownWorkerError(Exception):
    """A worker ID is not registered."""


class WorkerRegistry:
    """Store workers in memory and return copies to protect internal state."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = Lock()

    def register_worker(self, registration: WorkerRegistration) -> WorkerInfo:
        """Register a unique worker with controller-generated timestamps."""
        with self._lock:
            if registration.worker_id in self._workers:
                raise DuplicateWorkerError(registration.worker_id)
            now = datetime.now(timezone.utc)
            worker = WorkerInfo(**registration.model_dump(), registered_at=now, last_seen=now)
            self._workers[worker.worker_id] = worker
            return worker.model_copy(deep=True)

    def get_worker(self, worker_id: str) -> WorkerInfo:
        """Return a worker, or raise UnknownWorkerError."""
        with self._lock:
            return self._require_worker(worker_id).model_copy(deep=True)

    def get_workers(self, timeout: float | None = None) -> list[WorkerInfo]:
        """Return an active snapshot, optionally expiring records before the read."""
        with self._lock:
            if timeout is not None:
                self._expire_locked(timeout, datetime.now(timezone.utc))
            return [worker.model_copy(deep=True) for worker in sorted(self._workers.values(), key=lambda item: item.worker_id)]

    def get_cluster_summary(self, timeout: float) -> ClusterSummary:
        """Aggregate a single locked snapshot, excluding CPU-only GPU labels."""
        workers = self.get_workers(timeout=timeout)
        gpu_workers = [worker for worker in workers if worker.gpu.casefold() != "none"]
        return ClusterSummary(
            total_workers=len(workers),
            ready_workers=sum(worker.status == WorkerStatus.READY for worker in workers),
            busy_workers=sum(worker.status == WorkerStatus.BUSY for worker in workers),
            total_gpu_memory=fsum(worker.gpu_memory for worker in gpu_workers),
            gpus=dict(sorted(Counter(worker.gpu for worker in gpu_workers).items())),
            workers=workers,
        )

    def update_heartbeat(
        self, worker_id: str, status: WorkerStatus | None = None,
        gpu_utilization: float | None = None, gpu_memory_used: float | None = None,
    ) -> WorkerInfo:
        """Refresh an existing worker's last-seen timestamp."""
        with self._lock:
            worker = self._require_worker(worker_id)
            worker.last_seen = datetime.now(timezone.utc)
            if status is not None:
                worker.status = status
            worker.gpu_utilization = gpu_utilization
            worker.gpu_memory_used = gpu_memory_used
            logging.getLogger(__name__).info("[Registry] Heartbeat received: %s", worker_id)
            return worker.model_copy(deep=True)

    def expire_stale_workers(self, timeout: float, now: datetime | None = None) -> list[str]:
        """Atomically remove expired records and return their IDs."""
        current = now if now is not None else datetime.now(timezone.utc)
        with self._lock:
            return self._expire_locked(timeout, current)

    def _expire_locked(self, timeout: float, current: datetime) -> list[str]:
        """Delete inactive records while the caller holds the registry lock."""
        expired = [
            worker_id for worker_id, worker in self._workers.items()
            if worker.status == WorkerStatus.OFFLINE
            or (current - worker.last_seen).total_seconds() > timeout
        ]
        for worker_id in expired:
            del self._workers[worker_id]
            logging.getLogger(__name__).warning(
                "[Registry] Worker expired and removed: %s", worker_id
            )
        return expired

    def unregister_worker(self, worker_id: str) -> None:
        """Remove an existing worker, or raise UnknownWorkerError."""
        with self._lock:
            self._require_worker(worker_id)
            del self._workers[worker_id]

    def _require_worker(self, worker_id: str) -> WorkerInfo:
        """Look up internal state while the caller holds the lock."""
        try:
            return self._workers[worker_id]
        except KeyError:
            raise UnknownWorkerError(worker_id) from None


    def begin_nn_test(self, worker_id: str, timeout: float) -> tuple[WorkerInfo, WorkerInfo]:
        """Reserve a ready worker and expose BUSY immediately to dashboard polls."""
        with self._lock:
            self._expire_locked(timeout, datetime.now(timezone.utc))
            worker = self._require_worker(worker_id)
            if worker.status != WorkerStatus.READY:
                raise WorkerNotReadyError(worker_id)
            worker.status = WorkerStatus.BUSY
            return worker.model_copy(deep=True), worker

    def finish_nn_test(self, worker_id: str, original: WorkerInfo) -> None:
        """Restore only the same registration, never recreate an expired record."""
        with self._lock:
            current = self._workers.get(worker_id)
            if current is original and current.status == WorkerStatus.BUSY:
                current.status = WorkerStatus.READY
