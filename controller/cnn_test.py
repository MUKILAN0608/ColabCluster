"""Forward the fixed CNN diagnostic over HTTP; never import PyTorch here."""

import requests
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
    try:
        response = requests.post(str(url).rstrip("/") + "/cnn-test", json={},
                                 timeout=(5, 120), allow_redirects=False)
    except requests.Timeout:
        raise HTTPException(504, "CNN diagnostic timed out; it may still be running on the worker") from None
    except requests.RequestException:
        raise HTTPException(502, "Unable to reach worker diagnostic endpoint") from None
    try:
        if response.status_code != 200:
            detail = "Worker CNN diagnostic failed"
            try:
                detail = str(response.json().get("detail", detail))[:500]
            except (ValueError, AttributeError):
                pass
            code = response.status_code if response.status_code in (409, 503) else 502
            raise HTTPException(code, detail)
        try:
            result = CnnTestResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            raise HTTPException(502, "Worker returned an invalid CNN diagnostic result") from None
        if result.worker_id != worker.worker_id:
            raise HTTPException(502, "CNN result worker_id does not match the selected worker")
        return result
    finally:
        response.close()
