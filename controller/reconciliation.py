"""Read-only remote reconciliation; never resubmit or cancel GPU work."""
import requests
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter

from common.config import http_timeout


def reconcile(registry, worker_id):
    worker = registry.get_worker(worker_id)
    execution = (worker.metadata or {}).get("execution", {})
    if execution.get("state") != "unknown":
        return worker
    request_id = execution["request_id"]
    url = TypeAdapter(HttpUrl).validate_python(worker.metadata["diagnostic_url"])
    try:
        response = requests.get(str(url).rstrip("/") + f"/executions/{request_id}",
                                timeout=http_timeout("preflight"), allow_redirects=False)
    except requests.RequestException as exc:
        raise HTTPException(502, f"Reconciliation failed ({type(exc).__name__}); execution remains unknown: {exc}") from exc
    try:
        if response.status_code != 200:
            raise HTTPException(502, f"Reconciliation HTTP {response.status_code}; execution remains unknown")
        try:
            data = response.json()
            if (not isinstance(data, dict) or data.get("worker_id") != worker_id
                    or data.get("request_id") != request_id
                    or not isinstance(data.get("slot_available"), bool)
                    or data.get("state") not in ("running", "unknown", "completed", "failed")):
                raise ValueError("Invalid execution acknowledgement")
        except ValueError as exc:
            raise HTTPException(502, "Invalid reconciliation response; execution remains unknown") from exc
        return registry.reconcile_execution(worker_id, worker.registered_at, request_id,
                                            data["state"], data["slot_available"])
    finally:
        response.close()
