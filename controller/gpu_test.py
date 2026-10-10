"""HTTP-only forwarding; this module never imports the worker or PyTorch."""

import requests
from controller.transport import post_execution
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter, ValidationError

from common.schemas import GpuTestRequest, GpuTestResponse, WorkerInfo


def forward_gpu_test(worker: WorkerInfo, payload: GpuTestRequest) -> GpuTestResponse:
    """Call only the selected worker's advertised fixed diagnostic endpoint."""
    raw_url = (worker.metadata or {}).get("diagnostic_url")
    try:
        url = TypeAdapter(HttpUrl).validate_python(raw_url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid diagnostic URL")
    except (ValidationError, ValueError):
        raise HTTPException(409, "Worker has no valid diagnostic_url; enable worker diagnostics") from None
    response = post_execution(worker, str(url).rstrip("/") + "/gpu-test", payload.model_dump())
    try:
        try:
            result = GpuTestResponse.model_validate(response.json())
        except (ValidationError, ValueError):
            raise HTTPException(502, "Worker returned an invalid benchmark result") from None
        if (result.worker_id != worker.worker_id or not result.cuda_available
                or result.matrix_size != payload.matrix_size or result.iterations != payload.iterations
                or result.result_shape != (payload.matrix_size, payload.matrix_size)):
            raise HTTPException(502, "Worker benchmark result does not match the request")
        return result
    finally:
        response.close()
