"""Forward the fixed NN diagnostic over HTTP; never import PyTorch here."""

import requests
from controller.transport import post_execution
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter, ValidationError

from common.schemas import NnTestResponse, WorkerInfo


def forward_nn_test(worker: WorkerInfo) -> NnTestResponse:
    """Send an empty request to the worker's predefined NN endpoint."""
    try:
        url = TypeAdapter(HttpUrl).validate_python((worker.metadata or {}).get("diagnostic_url"))
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid URL")
    except (ValidationError, ValueError):
        raise HTTPException(409, "Worker has no valid diagnostic_url; enable worker diagnostics") from None
    response = post_execution(worker, str(url).rstrip("/") + "/nn-test", {})
    try:
        try:
            result = NnTestResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            raise HTTPException(502, "Worker returned an invalid NN diagnostic result") from None
        if result.worker_id != worker.worker_id:
            raise HTTPException(502, "NN result worker_id does not match the selected worker")
        return result
    finally:
        response.close()
