"""Bounded HTTP requests for registration, heartbeat, and unregistration."""

import os
from urllib.parse import urlsplit

import requests
from pydantic import ValidationError

from common.schemas import (
    WorkerRegistration, WorkerRegistrationResponse, WorkerStatus,
    WorkerUnregister, WorkerUnregisterResponse,
    WorkerHeartbeat, WorkerHeartbeatResponse,
)
from worker.hardware import get_gpu_metrics
from worker.worker import Worker


class WorkerClientError(RuntimeError):
    """A controller request failed or returned an invalid acknowledgement."""


class WorkerClient:
    """Communicate with an explicitly configured controller over HTTP(S)."""

    def __init__(self, controller_url: str | None = None) -> None:
        configured = controller_url if controller_url is not None else os.getenv(
            "COLABCLUSTER_CONTROLLER_URL", "http://127.0.0.1:8000"
        )
        self.controller_url = configured.strip().rstrip("/")
        try:
            parsed = urlsplit(self.controller_url)
            valid = (
                parsed.scheme in {"http", "https"} and parsed.hostname
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
                and not any(character.isspace() for character in self.controller_url)
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Controller URL must be an HTTP(S) URL without credentials, query, or fragment")

    def register(self, worker: Worker) -> WorkerRegistrationResponse:
        """Send actual detected hardware once and validate the acknowledgement."""
        hardware = worker.get_hardware_info()
        payload = WorkerRegistration(
            worker_id=worker.worker_id, gpu=hardware.gpu, gpu_memory=hardware.gpu_memory,
            status=WorkerStatus.READY, cpu_count=hardware.cpu_count,
            ram_total=hardware.ram_total, cuda_available=hardware.cuda_available,
            metadata={
                "platform": hardware.platform, "python_version": hardware.python_version,
                "torch_version": hardware.torch_version,
            },
        )
        diagnostic_url = os.getenv("COLABCLUSTER_WORKER_URL", "").strip()
        if diagnostic_url:
            payload.metadata["diagnostic_url"] = diagnostic_url
        data = self._post("register", payload.model_dump(mode="json"), worker.worker_id)
        try:
            response = WorkerRegistrationResponse.model_validate(data)
        except ValidationError as exc:
            raise WorkerClientError("Controller returned an invalid registration response") from exc
        self._check_ack(response.success, response.worker_id, worker.worker_id, response.message)
        return response

    def heartbeat(self, worker: Worker) -> WorkerHeartbeatResponse:
        """Send state and optional metrics with a timezone-aware UTC timestamp."""
        utilization, memory_used = get_gpu_metrics(worker.get_hardware_info().cuda_available)
        payload = WorkerHeartbeat(
            worker_id=worker.worker_id, status=worker.status,
            gpu_utilization=utilization, gpu_memory_used=memory_used,
        )
        data = self._post("heartbeat", payload.model_dump(mode="json"), worker.worker_id)
        try:
            response = WorkerHeartbeatResponse.model_validate(data)
        except ValidationError as exc:
            raise WorkerClientError("Controller returned an invalid heartbeat response") from exc
        self._check_ack(response.success, response.worker_id, worker.worker_id, response.message)
        return response

    def unregister(self, worker_id: str, reason: str | None = None) -> WorkerUnregisterResponse:
        """Explicitly remove a worker; this is not automatic shutdown recovery."""
        payload = WorkerUnregister(worker_id=worker_id, reason=reason)
        data = self._post("unregister", payload.model_dump(mode="json"), payload.worker_id)
        try:
            response = WorkerUnregisterResponse.model_validate(data)
        except ValidationError as exc:
            raise WorkerClientError("Controller returned an invalid unregistration response") from exc
        self._check_ack(response.success, response.worker_id, payload.worker_id, response.message)
        return response

    @staticmethod
    def _check_ack(success: bool, actual_id: str, expected_id: str, message: str) -> None:
        """Reject unsuccessful or mismatched acknowledgements."""
        if not success:
            raise WorkerClientError(f"Controller rejected request: {message}")
        if actual_id != expected_id:
            raise WorkerClientError("Controller response worker_id does not match the request")

    def _post(self, operation: str, payload: dict, worker_id: str) -> object:
        """Make a bounded request and translate transport/protocol failures."""
        url = f"{self.controller_url}/workers/{operation}"
        try:
            response = requests.post(url, json=payload, timeout=(5, 15), allow_redirects=False)
        except requests.Timeout as exc:
            raise WorkerClientError(f"Connection/request timed out at {self.controller_url}") from exc
        except requests.ConnectionError as exc:
            raise WorkerClientError(f"Unable to connect to controller at {self.controller_url}") from exc
        except requests.RequestException as exc:
            raise WorkerClientError(f"Controller request failed at {self.controller_url}: {exc}") from exc
        try:
            if not 200 <= response.status_code < 300:
                if response.status_code == 409 and operation == "register":
                    raise WorkerClientError(f"Worker {worker_id} is already registered. (HTTP 409)")
                descriptions = {
                    400: "Controller rejected the request",
                    404: "Controller endpoint or worker was not found; check the controller URL",
                    422: "Controller rejected worker data validation",
                    500: "Controller encountered an internal error",
                }
                description = descriptions.get(response.status_code, "Unexpected controller response")
                raise WorkerClientError(f"{description} (HTTP {response.status_code}) at {url}")
            try:
                return response.json()
            except ValueError as exc:
                raise WorkerClientError("Controller returned invalid JSON") from exc
        finally:
            response.close()
