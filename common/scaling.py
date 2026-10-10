"""Bounded workload-scaling protocol, separate from the fixed diagnostic APIs."""
from typing import Literal
from pydantic import Field, model_validator
from common.schemas import ProtocolModel, SingleInferenceResponse

SIZES = (64, 256, 1024, 4096)
WORKER_IDS = ("COLAB-GPU-TEST", "COLAB-GPU-TEST-2")


def partition_samples(total: int, count: int) -> list[int]:
    """Assign any remainder to the first worker; never lose samples."""
    if count not in (1, 2) or total < count:
        raise ValueError("Expected one/two workers and at least one sample per worker")
    base, remainder = divmod(total, count)
    return [base + (index < remainder) for index in range(count)]


class ScalingRequest(ProtocolModel):
    total_samples: Literal[64, 256, 1024, 4096]
    worker_count: Literal[1, 2]


class ScalingBatchRequest(ScalingRequest):
    partition: Literal[0, 1]

    @model_validator(mode="after")
    def valid_partition(self):
        if self.partition >= self.worker_count:
            raise ValueError("Partition must identify an assigned worker")
        return self

    @property
    def assigned_samples(self):
        return partition_samples(self.total_samples, self.worker_count)[self.partition]


class ScalingBatchResponse(SingleInferenceResponse):
    partition: Literal[0, 1]
    batch_size: int = Field(ge=1, le=4096, strict=True)
    input_shape: tuple[int, Literal[3], Literal[32], Literal[32]]
    output_shape: tuple[int, Literal[10]]

    @model_validator(mode="after")
    def matching_shapes(self):
        if self.input_shape[0] != self.batch_size or self.output_shape[0] != self.batch_size:
            raise ValueError("Input/output sample counts must match assigned batch")
        return self


class ScalingResponse(ScalingRequest):
    status: Literal["passed"] = "passed"
    workers: list[ScalingBatchResponse] = Field(min_length=1, max_length=2)
    wall_time_ms: float = Field(gt=0, allow_inf_nan=False)
    sum_worker_gpu_time_ms: float = Field(gt=0, allow_inf_nan=False)
    effective_throughput_images_per_second: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def correct_workload(self):
        if len(self.workers) != self.worker_count or len({w.worker_id for w in self.workers}) != self.worker_count:
            raise ValueError("Expected distinct assigned workers")
        counts = partition_samples(self.total_samples, self.worker_count)
        for index, (worker, count) in enumerate(zip(self.workers, counts)):
            if worker.partition != index or worker.batch_size != count:
                raise ValueError("Incorrect partition/sample count")
        return self
