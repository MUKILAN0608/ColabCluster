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
    gpu_utilization: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    gpu_memory_used: float | None = Field(default=None, ge=0, allow_inf_nan=False)


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


class ClusterSummary(ProtocolModel):
    """One active snapshot; memory is aggregate GiB across separate GPUs."""

    total_workers: int = Field(ge=0)
    ready_workers: int = Field(ge=0)
    busy_workers: int = Field(ge=0)
    total_gpu_memory: float = Field(ge=0, allow_inf_nan=False)
    gpus: dict[str, int]
    workers: list[WorkerInfo]


class GpuTestRequest(ProtocolModel):
    """Bounded CUDA matrix multiplication diagnostic."""

    matrix_size: int = Field(default=4096, ge=256, le=8192, strict=True)
    iterations: int = Field(default=20, ge=1, le=100, strict=True)


class GpuTestResponse(ProtocolModel):
    """Actual worker benchmark measurements; memory is measured in GiB."""

    worker_id: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    cuda_available: bool
    cuda_version: str
    torch_version: str
    matrix_size: int = Field(ge=256, le=8192)
    iterations: int = Field(ge=1, le=100)
    total_time_seconds: float = Field(gt=0, allow_inf_nan=False)
    average_time_ms: float = Field(gt=0, allow_inf_nan=False)
    gpu_memory_allocated_gb: float = Field(ge=0, allow_inf_nan=False)
    result_shape: tuple[int, int]


class NnTestRequest(ProtocolModel):
    """An empty request selects the fixed diagnostic; no custom code or options."""


class NnTestResponse(ProtocolModel):
    """CUDA-only SmallMLP inference measurements, excluding network latency."""

    worker_id: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    cuda_version: str
    torch_version: str
    device: Literal["cuda:0"]
    model: Literal["SmallMLP"]
    input_shape: tuple[Literal[128], Literal[784]]
    batch_size: Literal[128]
    forward_passes: Literal[100]
    total_gpu_time_ms: float = Field(gt=0, allow_inf_nan=False)
    average_inference_ms: float = Field(gt=0, allow_inf_nan=False)
    peak_memory_mb: float = Field(ge=0, allow_inf_nan=False)
    output_shape: tuple[Literal[128], Literal[10]]
    status: Literal["passed"]


class CnnTestRequest(ProtocolModel):
    """Empty request for the fixed SmallCNN diagnostic; no custom parameters."""


class CnnTestResponse(ProtocolModel):
    """Actual remote CUDA inference measurements for the predefined SmallCNN."""

    worker_id: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    cuda_version: str
    torch_version: str
    device: Literal["cuda:0"]
    model: Literal["SmallCNN"]
    input_shape: tuple[Literal[32], Literal[3], Literal[32], Literal[32]]
    batch_size: Literal[32]
    forward_passes: Literal[100]
    total_gpu_time_ms: float = Field(gt=0, allow_inf_nan=False)
    average_inference_ms: float = Field(gt=0, allow_inf_nan=False)
    throughput_images_per_second: float = Field(gt=0, allow_inf_nan=False)
    peak_memory_mb: float = Field(ge=0, allow_inf_nan=False)
    output_shape: tuple[Literal[32], Literal[10]]
    status: Literal["passed"]


class InferenceBatchRequest(ProtocolModel):
    """One of two fixed synthetic partitions; no tensors or executable input."""

    partition: Literal[0, 1]


class InferenceBatchResponse(CnnTestResponse):
    """One timed forward over 32 distinct synthetic samples."""

    partition: Literal[0, 1]
    gpu: Literal["Tesla T4"]
    cuda_available: Literal[True]
    forward_passes: Literal[1]


class TwoWorkerRequest(ProtocolModel):
    """Select the fixed pair explicitly, with defaults for the verified setup."""

    worker_ids: tuple[Literal["COLAB-GPU-TEST"], Literal["COLAB-GPU-TEST-2"]] = (
        "COLAB-GPU-TEST", "COLAB-GPU-TEST-2")


class TwoWorkerResponse(ProtocolModel):
    status: Literal["passed"] = "passed"
    worker_count: Literal[2] = 2
    total_samples: Literal[64] = 64
    workers: tuple[InferenceBatchResponse, InferenceBatchResponse]
    parallel_wall_time_ms: float = Field(gt=0, allow_inf_nan=False)
    sum_worker_gpu_time_ms: float = Field(gt=0, allow_inf_nan=False)
    effective_throughput_images_per_second: float = Field(gt=0, allow_inf_nan=False)
    output_shapes: tuple[tuple[Literal[32], Literal[10]], tuple[Literal[32], Literal[10]]]
