"""Exercise the controller's API contract and registry isolation."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from controller.main import create_app
from common.schemas import WorkerRegistration
from controller.registry import UnknownWorkerError, WorkerRegistry


@pytest.fixture
def client():
    """Provide a fresh controller for each test."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def registration():
    """Sample metadata submitted manually, without a worker process."""
    return {"worker_id": "local-test", "gpu": "T4", "gpu_memory": 16, "status": "ready"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy", "service": "colabcluster-controller", "version": "0.1.0"
    }


def test_workers_initially_empty(client):
    response = client.get("/workers")
    assert response.status_code == 200
    assert response.json() == {"workers": []}


def test_registration(client, registration):
    response = client.post("/workers/register", json=registration)
    assert response.status_code == 201
    worker = response.json()["worker"]
    assert all(worker[key] == value for key, value in registration.items())
    assert datetime.fromisoformat(worker["registered_at"].replace("Z", "+00:00")).utcoffset() == timedelta(0)
    assert worker["last_seen"] == worker["registered_at"]
    assert client.get("/workers").json() == {"workers": [worker]}


def test_duplicate_registration(client, registration):
    original = client.post("/workers/register", json=registration).json()["worker"]
    response = client.post("/workers/register", json={**registration, "gpu": "L4"})
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]
    assert client.get("/workers").json() == {"workers": [original]}


def test_heartbeat_updates_timestamp(client, registration):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = start + timedelta(seconds=30)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = start
        original = client.post("/workers/register", json=registration).json()["worker"]
        clock.now.return_value = later
        response = client.post("/workers/heartbeat", json={"worker_id": "local-test"})
    assert response.status_code == 200
    worker = response.json()["worker"]
    assert worker["registered_at"] == original["registered_at"]
    assert datetime.fromisoformat(worker["last_seen"].replace("Z", "+00:00")) == later
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = later
        assert client.get("/workers").json() == {"workers": [worker]}


@pytest.mark.parametrize("endpoint", ["heartbeat", "unregister"])
def test_unknown_worker(client, endpoint):
    response = client.post(f"/workers/{endpoint}", json={"worker_id": "missing"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_unregister(client, registration):
    client.post("/workers/register", json=registration)
    response = client.post("/workers/unregister", json={"worker_id": "local-test"})
    assert response.status_code == 200
    assert response.json() == {
        "success": True, "message": "Worker unregistered", "worker_id": "local-test"
    }
    assert client.get("/workers").json() == {"workers": []}
    assert client.post("/workers/heartbeat", json={"worker_id": "local-test"}).status_code == 404
    assert client.post("/workers/register", json=registration).status_code == 201


@pytest.mark.parametrize("changes", [
    {"worker_id": " "}, {"gpu": ""}, {"gpu_memory": -1},
    {"gpu_memory": "invalid"}, {"gpu_memory": "Infinity"},
    {"status": " "}, {"registered_at": "2026-01-01"},
])
def test_invalid_registration(client, registration, changes):
    assert client.post("/workers/register", json={**registration, **changes}).status_code == 422
    assert client.get("/workers").json() == {"workers": []}


@pytest.mark.parametrize("endpoint", ["register", "heartbeat", "unregister"])
def test_missing_required_fields(client, endpoint):
    assert client.post(f"/workers/{endpoint}", json={}).status_code == 422


def test_docs(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "ColabCluster Controller"
    assert schema["info"]["version"] == "0.1.0"
    assert len(schema["paths"]) == 11


def test_app_instances_are_isolated(client, registration):
    client.post("/workers/register", json=registration)
    with TestClient(create_app()) as other:
        assert other.get("/workers").json() == {"workers": []}


def test_registry_returns_copies(registration):
    registry = WorkerRegistry()
    registered = registry.register_worker(WorkerRegistration(**registration))
    registered.status = "changed"
    registry.get_worker("local-test").status = "changed"
    registry.get_workers()[0].status = "changed"
    assert registry.get_worker("local-test").status == "ready"
    registry.unregister_worker("local-test")
    with pytest.raises(UnknownWorkerError):
        registry.get_worker("local-test")
