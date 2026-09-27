"""Thread-safe, process-local worker storage."""

from datetime import datetime, timezone
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field


class WorkerRegistration(BaseModel):
    """Worker metadata; GPU memory is expressed in GiB."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    worker_id: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    gpu_memory: float = Field(ge=0, allow_inf_nan=False)
    status: str = Field(min_length=1)


class Worker(WorkerRegistration):
    """Registered metadata and controller-generated UTC timestamps."""

    registered_at: datetime
    last_seen: datetime


class DuplicateWorkerError(Exception):
    """A worker ID is already registered."""


class UnknownWorkerError(Exception):
    """A worker ID is not registered."""


class WorkerRegistry:
    """Store workers in memory and return copies to protect internal state."""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = Lock()

    def register_worker(self, registration: WorkerRegistration) -> Worker:
        """Register a unique worker with controller-generated timestamps."""
        with self._lock:
            if registration.worker_id in self._workers:
                raise DuplicateWorkerError(registration.worker_id)
            now = datetime.now(timezone.utc)
            worker = Worker(**registration.model_dump(), registered_at=now, last_seen=now)
            self._workers[worker.worker_id] = worker
            return worker.model_copy()

    def get_worker(self, worker_id: str) -> Worker:
        """Return a worker, or raise UnknownWorkerError."""
        with self._lock:
            return self._require_worker(worker_id).model_copy()

    def get_workers(self) -> list[Worker]:
        """Return a snapshot of registered workers."""
        with self._lock:
            return [worker.model_copy() for worker in self._workers.values()]

    def update_heartbeat(self, worker_id: str) -> Worker:
        """Refresh an existing worker's last-seen timestamp."""
        with self._lock:
            worker = self._require_worker(worker_id)
            worker.last_seen = datetime.now(timezone.utc)
            return worker.model_copy()

    def unregister_worker(self, worker_id: str) -> None:
        """Remove an existing worker, or raise UnknownWorkerError."""
        with self._lock:
            self._require_worker(worker_id)
            del self._workers[worker_id]

    def _require_worker(self, worker_id: str) -> Worker:
        """Look up internal state while the caller holds the lock."""
        try:
            return self._workers[worker_id]
        except KeyError:
            raise UnknownWorkerError(worker_id) from None
