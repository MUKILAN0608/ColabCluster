"""Validate shared models and their controller integration."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.schemas import (
    WorkerHeartbeat, WorkerHeartbeatResponse, WorkerInfo, WorkerRegistration,
    WorkerRegistrationResponse, WorkerStatus, WorkerUnregister, WorkerUnregisterResponse,
)
from controller.main import create_app
from controller.registry import WorkerRegistry


def registration(**changes) -> WorkerRegistration:
    """Construct representative backend-independent resource metadata."""
    return WorkerRegistration.model_validate({
        "worker_id": "test", "gpu": "generic", "gpu_memory": 16, "status": "ready",
        "cpu_count": 4, "ram_total": 32, "cuda_available": True,
        "metadata": {"labels": {"region": "local"}}, **changes,
    })


def test_valid_registration() -> None:
    message = registration()
    assert message.status is WorkerStatus.READY
    assert WorkerRegistration.model_validate_json(message.model_dump_json()) == message
    minimal = WorkerRegistration(worker_id="minimal", gpu="generic", gpu_memory=0, status="ready")
    assert minimal.cpu_count is None
    assert minimal.metadata is None


@pytest.mark.parametrize("changes", [
    {"worker_id": " "}, {"gpu": " "}, {"gpu_memory": -1}, {"gpu_memory": float("inf")},
    {"cpu_count": -1}, {"cpu_count": 1.5}, {"ram_total": -1},
    {"ram_total": float("nan")}, {"status": "unknown"}, {"metadata": []},
])
def test_invalid_registration(changes) -> None:
    with pytest.raises(ValidationError):
        registration(**changes)


@pytest.mark.parametrize("status", list(WorkerStatus))
def test_status_values(status: WorkerStatus) -> None:
    assert registration(status=status.value).status is status


def test_valid_heartbeat() -> None:
    message = WorkerHeartbeat(
        worker_id="test", status=WorkerStatus.BUSY, gpu_utilization=100,
        gpu_memory_used=0, timestamp=datetime.now(timezone.utc),
    )
    assert WorkerHeartbeat.model_validate_json(message.model_dump_json()) == message
    assert WorkerHeartbeat(worker_id="test").timestamp.utcoffset() == timedelta(0)


@pytest.mark.parametrize("changes", [
    {"worker_id": ""}, {"status": "unknown"}, {"gpu_utilization": 101},
    {"gpu_utilization": -1}, {"gpu_utilization": float("nan")},
    {"gpu_memory_used": -1}, {"gpu_memory_used": float("inf")},
    {"timestamp": "invalid"}, {"timestamp": "2026-01-01T00:00:00"},
])
def test_invalid_heartbeat(changes) -> None:
    with pytest.raises(ValidationError):
        WorkerHeartbeat.model_validate({"worker_id": "test", **changes})


def test_valid_unregister() -> None:
    message = WorkerUnregister(worker_id="test", reason="shutdown")
    assert WorkerUnregister.model_validate_json(message.model_dump_json()) == message
    assert WorkerUnregister(worker_id="test").reason is None
    with pytest.raises(ValidationError):
        WorkerUnregister(worker_id=" ")


@pytest.mark.parametrize("field", ["registered_at", "last_seen"])
def test_worker_info_requires_aware_timestamps(field: str) -> None:
    values = {**registration().model_dump(), "registered_at": datetime.now(timezone.utc),
              "last_seen": datetime.now(timezone.utc)}
    values[field] = datetime(2026, 1, 1)
    with pytest.raises(ValidationError):
        WorkerInfo.model_validate(values)


def test_protocol_lifecycle() -> None:
    with TestClient(create_app()) as client:
        request = registration()
        response = client.post("/workers/register", json=request.model_dump(mode="json"))
        assert response.status_code == 201
        acknowledgement = WorkerRegistrationResponse.model_validate(response.json())
        assert acknowledgement.success and acknowledgement.worker_id == request.worker_id
        worker = WorkerInfo.model_validate(client.get("/workers").json()["workers"][0])
        assert worker.metadata == request.metadata
        assert worker.cpu_count == 4 and worker.ram_total == 32 and worker.cuda_available

        heartbeat = WorkerHeartbeat(
            worker_id=request.worker_id, status=WorkerStatus.BUSY,
            gpu_utilization=50, gpu_memory_used=8,
            timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        response = client.post("/workers/heartbeat", json=heartbeat.model_dump(mode="json"))
        assert response.status_code == 200
        ack = WorkerHeartbeatResponse.model_validate(response.json())
        assert ack.success and ack.worker_id == request.worker_id
        updated = WorkerInfo.model_validate(client.get("/workers").json()["workers"][0])
        assert updated.status is WorkerStatus.BUSY
        assert updated.last_seen == ack.server_timestamp
        assert updated.last_seen > heartbeat.timestamp
        assert updated.registered_at == worker.registered_at
        # A Step 2 client omitting status must not reset a busy worker to ready.
        response = client.post("/workers/heartbeat", json={"worker_id": request.worker_id})
        assert response.json()["worker"]["status"] == "busy"

        invalid = heartbeat.model_dump(mode="json") | {"gpu_utilization": 101}
        assert client.post("/workers/heartbeat", json=invalid).status_code == 422
        message = WorkerUnregister(worker_id=request.worker_id, reason="done")
        response = client.post("/workers/unregister", json=message.model_dump(mode="json"))
        assert response.status_code == 200
        ack = WorkerUnregisterResponse.model_validate(response.json())
        assert ack.success and ack.worker_id == request.worker_id
        assert client.get("/workers").json() == {"workers": []}


def test_nested_metadata_is_isolated() -> None:
    registry = WorkerRegistry()
    message = registration()
    result = registry.register_worker(message)
    message.metadata["labels"]["region"] = "input-mutation"
    result.metadata["labels"]["region"] = "response-mutation"
    registry.get_workers()[0].metadata["labels"]["region"] = "list-mutation"
    registry.get_worker("test").metadata["labels"]["region"] = "lookup-mutation"
    assert registry.get_worker("test").metadata == {"labels": {"region": "local"}}


def test_openapi_protocol_models() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    for route, model, code in [
        ("register", "WorkerRegistration", "201"),
        ("heartbeat", "WorkerHeartbeat", "200"),
        ("unregister", "WorkerUnregister", "200"),
    ]:
        operation = schema["paths"][f"/workers/{route}"]["post"]
        request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
        assert request_schema["$ref"] == f"#/components/schemas/{model}"
        response_schema = operation["responses"][code]["content"]["application/json"]["schema"]
        assert response_schema["$ref"] == f"#/components/schemas/{model}Response"
