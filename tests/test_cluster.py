"""Independent multi-worker state and consistent cluster aggregation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from common.schemas import WorkerRegistration
from controller.main import create_app
from controller.registry import DuplicateWorkerError, WorkerRegistry


def payload(worker_id, **changes):
    return {"worker_id": worker_id, "gpu": "Tesla T4", "gpu_memory": 14.56,
            "status": "ready", "metadata": {"owner": worker_id}, **changes}


@pytest.mark.parametrize("count", [1, 2, 5])
def test_multiple_workers_and_order(count):
    with TestClient(create_app()) as client:
        ids = [f"colab-{i}" for i in range(count)]
        for worker_id in reversed(ids):
            assert client.post("/workers/register", json=payload(worker_id)).status_code == 201
        workers = client.get("/workers").json()["workers"]
        assert [w["worker_id"] for w in workers] == ids
        summary = client.get("/cluster").json()
        assert summary["workers"] == workers
        assert summary["total_workers"] == summary["ready_workers"] == count
        assert summary["busy_workers"] == 0
        assert summary["gpus"] == {"Tesla T4": count}
        assert summary["total_gpu_memory"] == pytest.approx(14.56 * count)


def test_empty_and_mixed_cluster():
    with TestClient(create_app()) as client:
        assert client.get("/cluster").json() == {
            "total_workers": 0, "ready_workers": 0, "busy_workers": 0,
            "total_gpu_memory": 0, "gpus": {}, "workers": [],
        }
        for data in [payload("a"), payload("b", gpu="L4", gpu_memory=24, status="busy"),
                     payload("cpu", gpu="None", gpu_memory=0, status="error"),
                     payload("offline", status="offline")]:
            assert client.post("/workers/register", json=data).status_code == 201
        result = client.get("/cluster").json()
        assert result["total_workers"] == 3
        assert result["ready_workers"] == result["busy_workers"] == 1
        assert result["gpus"] == {"L4": 1, "Tesla T4": 1}
        assert result["total_gpu_memory"] == pytest.approx(38.56)
        assert client.post("/workers/unregister", json={"worker_id": "b"}).status_code == 200
        assert client.get("/cluster").json()["total_workers"] == 2


def test_independent_heartbeats_expiry_and_duplicate():
    start = datetime.now(timezone.utc)
    with TestClient(create_app()) as client, patch("controller.registry.datetime") as clock:
        clock.now.return_value = start
        for worker_id in ["a", "b"]:
            client.post("/workers/register", json=payload(worker_id))
        before = client.get("/workers").json()["workers"]
        assert client.post("/workers/register", json=payload("a", gpu="replacement")).status_code == 409
        clock.now.return_value = start + timedelta(seconds=20)
        client.post("/workers/heartbeat", json={"worker_id": "a", "status": "busy",
                    "gpu_utilization": 50, "gpu_memory_used": 4})
        current = client.get("/workers").json()["workers"]
        assert current[1] == before[1]
        assert current[0]["last_seen"] != before[0]["last_seen"]
        assert current[0]["metadata"] == before[0]["metadata"]
        assert current[0]["gpu"] == "Tesla T4"
        assert current[0]["gpu_utilization"] == 50
        assert current[0]["gpu_memory_used"] == 4
        clock.now.return_value = start + timedelta(seconds=31)
        summary = client.get("/cluster").json()
        assert summary["total_workers"] == summary["busy_workers"] == 1
        assert summary["ready_workers"] == 0
        assert [w["worker_id"] for w in summary["workers"]] == ["a"]
        assert summary["workers"] == client.get("/workers").json()["workers"]
        assert client.post("/workers/register", json=payload("b")).status_code == 201
        assert client.get("/cluster").json()["total_workers"] == 2


def test_concurrent_duplicate_registration_cannot_overwrite():
    registry = WorkerRegistry()
    def register(index):
        try:
            registry.register_worker(WorkerRegistration(**payload("same", metadata={"index": index})))
            return index
        except DuplicateWorkerError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(register, range(20)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert registry.get_worker("same").metadata == {"index": winners[0]}


def test_cluster_openapi_schema():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/cluster"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/ClusterSummary")
