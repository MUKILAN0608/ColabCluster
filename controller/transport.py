"""One-shot GPU dispatch. A lost reply never proves that execution stopped."""
from uuid import uuid4

import requests
from fastapi import HTTPException

from common.config import http_timeout


def post_execution(worker, url, payload):
    timeout = http_timeout()
    execution = {"request_id": uuid4().hex, "state": "unknown"}
    worker.metadata = {**(worker.metadata or {}), "execution": execution}
    headers = {"X-ColabCluster-Request-ID": execution["request_id"]}
    try:
        response = requests.post(url, json=payload, timeout=timeout,
                                 allow_redirects=False, headers=headers)
    except requests.ConnectTimeout as exc:
        execution["state"] = "not_started"
        fail(worker, execution, 504, "connect_timeout", str(exc), timeout)
    except requests.ReadTimeout as exc:
        fail(worker, execution, 504, "read_timeout", str(exc), timeout)
    except requests.Timeout as exc:
        fail(worker, execution, 504, "transport_timeout", str(exc), timeout)
    except requests.RequestException as exc:
        fail(worker, execution, 502, "connection_error", str(exc), timeout)
    if response.status_code == 200:
        execution["state"] = "completed"
        return response
    try:
        detail = response.json()
        detail = detail.get("detail", "") if isinstance(detail, dict) else ""
        detail = str(detail)[:500] if isinstance(detail, (str, dict, list)) else ""
    except ValueError:
        detail = "Non-JSON remote error"
    finally:
        response.close()
    # A proxy can generate even a 503. Only a matching worker acknowledgement
    # proves this execution failed after leaving the GPU function.
    if (response.headers.get("X-ColabCluster-Request-ID") == execution["request_id"]
            and response.headers.get("X-ColabCluster-Execution-State") == "failed"):
        execution["state"] = "failed"
    code = response.status_code if response.status_code in (409, 503, 504) else 502
    fail(worker, execution, code, "worker_execution_error" if execution["state"] == "failed" else "remote_http_error",
         f"HTTP {response.status_code}: {detail}", timeout)


def fail(worker, execution, code, kind, detail, timeout):
    execution["error_kind"] = kind
    message = (f"{worker.worker_id}: {kind}; request_id={execution['request_id']}; "
               f"execution={execution['state']}; connect={timeout[0]:g}s read={timeout[1]:g}s; "
               f"{detail}")
    if execution["state"] == "unknown":
        message += "; remote work may continue; reconcile before dispatching again"
    raise HTTPException(code, message)
