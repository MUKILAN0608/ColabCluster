"""Forward the fixed CNN diagnostic over HTTP; never import PyTorch here."""

import requests
from controller.transport import post_execution
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter, ValidationError

from common.schemas import CnnTestResponse, WorkerInfo


def forward_cnn_test(worker: WorkerInfo) -> CnnTestResponse:
    """Send an empty request to the worker's predefined CNN endpoint."""
    try:
        url = TypeAdapter(HttpUrl).validate_python((worker.metadata or {}).get("diagnostic_url"))
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid URL")
    except (ValidationError, ValueError):
        raise HTTPException(409, "Worker has no valid diagnostic_url; enable worker diagnostics") from None
    response = post_execution(worker, str(url).rstrip("/") + "/cnn-test", {})
    try:
        try:
            result = CnnTestResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            raise HTTPException(502, "Worker returned an invalid CNN diagnostic result") from None
        if result.worker_id != worker.worker_id:
            raise HTTPException(502, "CNN result worker_id does not match the selected worker")
        return result
    finally:
        response.close()
