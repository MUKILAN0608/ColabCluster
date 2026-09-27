"""Thread-safe, process-local worker storage."""

from datetime import datetime, timezone
from threading import Lock

from common.schemas import WorkerInfo, WorkerRegistration, WorkerStatus


class DuplicateWorkerError(Exception):
    """A worker ID is already registered."""


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

    def get_workers(self) -> list[WorkerInfo]:
        """Return a snapshot of registered workers."""
        with self._lock:
            return [worker.model_copy(deep=True) for worker in self._workers.values()]

    def update_heartbeat(self, worker_id: str, status: WorkerStatus | None = None) -> WorkerInfo:
        """Refresh an existing worker's last-seen timestamp."""
        with self._lock:
            worker = self._require_worker(worker_id)
            worker.last_seen = datetime.now(timezone.utc)
            if status is not None:
                worker.status = status
            return worker.model_copy(deep=True)

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
