"""Worker protocol models; all memory quantities are expressed in GiB."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class WorkerStatus(str, Enum):
    """Supported worker states, serialized as lowercase strings."""

    READY = "ready"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class ProtocolModel(BaseModel):
    """Common validation for protocol messages."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class WorkerRegistration(ProtocolModel):
    """Advertise worker identity and optional resource information."""

    worker_id: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    gpu_memory: float = Field(ge=0, allow_inf_nan=False)
    status: WorkerStatus
    cpu_count: int | None = Field(default=None, ge=0)
    ram_total: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cuda_available: bool | None = None
    metadata: dict[str, Any] | None = None


class WorkerInfo(WorkerRegistration):
    """Controller record with timezone-aware receipt times."""

    registered_at: AwareDatetime
    last_seen: AwareDatetime


class WorkerRegistrationResponse(ProtocolModel):
    """Registration acknowledgement, retaining the Step 2 worker payload."""

    success: bool
    worker_id: str = Field(min_length=1)
    message: str
    worker: WorkerInfo | None = None


class WorkerHeartbeat(ProtocolModel):
    """State and telemetry; defaults support legacy ID-only requests."""

    worker_id: str = Field(min_length=1)
    status: WorkerStatus = WorkerStatus.READY
    gpu_utilization: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    gpu_memory_used: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkerHeartbeatResponse(ProtocolModel):
    """Acknowledgement with authoritative controller receipt time."""

    success: bool
    worker_id: str = Field(min_length=1)
    message: str
    server_timestamp: AwareDatetime
    worker: WorkerInfo | None = None


class WorkerUnregister(ProtocolModel):
    """Request removal, optionally explaining why the worker is leaving."""

    worker_id: str = Field(min_length=1)
    reason: str | None = None


class WorkerUnregisterResponse(ProtocolModel):
    """Acknowledgement that a worker was removed."""

    success: bool
    worker_id: str = Field(min_length=1)
    message: str


class HealthResponse(ProtocolModel):
    """Controller health information."""

    status: Literal["healthy"] = "healthy"
    service: Literal["colabcluster-controller"] = "colabcluster-controller"
    version: Literal["0.1.0"] = "0.1.0"


class WorkersResponse(ProtocolModel):
    """Snapshot of registered workers."""

    workers: list[WorkerInfo]


class HardwareInfo(ProtocolModel):
    """Local hardware snapshot; RAM and GPU memory are measured in GiB."""

    model_config = ConfigDict(frozen=True)

    cpu_count: int = Field(ge=1)
    ram_total: float = Field(gt=0, allow_inf_nan=False)
    gpu: str = Field(min_length=1)
    gpu_memory: float = Field(ge=0, allow_inf_nan=False)
    cuda_available: bool
    torch_version: str | None
    python_version: str
    platform: str


class LocalWorkerInfo(ProtocolModel):
    """Local identity and hardware, independent of controller registration."""

    model_config = ConfigDict(frozen=True)

    worker_id: str = Field(min_length=1)
    hardware: HardwareInfo
