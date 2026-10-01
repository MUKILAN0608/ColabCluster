"""HTTP-only forwarding; this module never imports the worker or PyTorch."""

import requests
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
    try:
        response = requests.post(
            str(url).rstrip("/") + "/gpu-test", json=payload.model_dump(),
            timeout=(5, 120), allow_redirects=False,
        )
    except requests.Timeout:
        raise HTTPException(504, "Worker GPU diagnostic timed out; it may still be running") from None
    except requests.RequestException:
        raise HTTPException(502, "Unable to reach worker diagnostic endpoint") from None
    try:
        if response.status_code != 200:
            detail = "Worker diagnostic failed"
            try:
                detail = str(response.json().get("detail", detail))[:500]
            except (ValueError, AttributeError):
                pass
            status = response.status_code if response.status_code in (409, 503) else 502
            raise HTTPException(status, detail)
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
