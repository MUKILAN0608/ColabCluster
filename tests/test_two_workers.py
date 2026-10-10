"""Two independent runtimes, using in-process HTTP and simulated GPU results."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from common.schemas import WorkerStatus
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setenv("COLABCLUSTER_WORKER_TIMEOUT", "30")
    with TestClient(create_app()) as client:
        for name, memory in (("a", 14.56), ("b", 14.57)):
            response = client.post("/workers/register", json={
                "worker_id": name, "gpu": "Tesla T4", "gpu_memory": memory,
                "cuda_available": True, "status": "ready",
                "metadata": {"diagnostic_url": f"https://worker-{name}.example",
                             "platform": "Linux", "runtime": name},
            })
            assert response.status_code == 201
        yield client


def snapshot(client):
    response = client.get("/workers")
    assert response.status_code == 200
    workers = response.json()["workers"]
    summary = client.get("/cluster").json()
    assert summary["workers"] == workers
    assert summary["total_workers"] == len(workers)
    assert summary["ready_workers"] == sum(w["status"] == "ready" for w in workers)
    assert summary["busy_workers"] == sum(w["status"] == "busy" for w in workers)
    return {w["worker_id"]: w for w in workers}


def test_two_worker_identity_metadata_and_bidirectional_heartbeats(pair):
    initial = snapshot(pair)
    assert set(initial) == {"a", "b"}
    assert pair.get("/cluster").json()["gpus"] == {"Tesla T4": 2}
    assert pair.get("/cluster").json()["total_gpu_memory"] == pytest.approx(29.13)
    for name, memory in (("a", 14.56), ("b", 14.57)):
        assert initial[name]["gpu"] == "Tesla T4"
        assert initial[name]["gpu_memory"] == memory
        assert initial[name]["cuda_available"] is True
        assert initial[name]["metadata"] == {
            "diagnostic_url": f"https://worker-{name}.example",
            "platform": "Linux", "runtime": name,
        }
    start = datetime.now(timezone.utc)
    with patch("controller.registry.datetime") as clock:
        for tick, (name, status) in enumerate(
            (("a", "busy"), ("a", "ready"), ("b", "busy"), ("b", "ready")), 1
        ):
            clock.now.return_value = start + timedelta(seconds=tick)
            before = snapshot(pair)
            response = pair.post("/workers/heartbeat", json={
                "worker_id": name, "status": status,
                "gpu_utilization": tick * 10, "gpu_memory_used": tick,
            })
            assert response.status_code == 200
            after = snapshot(pair)
            other = "b" if name == "a" else "a"
            assert after[other] == before[other]
            assert after[other]["status"] == "ready"
            assert after[name]["status"] == status
            assert after[name]["last_seen"] != before[name]["last_seen"]
            assert after[name]["gpu_utilization"] == tick * 10
            assert after[name]["gpu_memory_used"] == tick
            assert after[name]["metadata"] == initial[name]["metadata"]
    assert all(w["status"] == "ready" for w in snapshot(pair).values())


@pytest.mark.parametrize("removal", ["unregister", "stale"])
def test_removing_a_preserves_b(pair, removal):
    start = datetime.now(timezone.utc)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = start + timedelta(seconds=20)
        assert pair.post("/workers/heartbeat", json={"worker_id": "b"}).status_code == 200
        before = snapshot(pair)["b"]
        if removal == "unregister":
            assert pair.post("/workers/unregister", json={"worker_id": "a"}).status_code == 200
        else:
            clock.now.return_value = start + timedelta(seconds=31)
            assert pair.app.state.registry.expire_stale_workers(30) == ["a"]
        assert snapshot(pair) == {"b": before}


@pytest.mark.parametrize("kind", ["nn", "cnn"])
def test_diagnostics_route_to_independent_worker_apps(pair, monkeypatch, kind):
    workers = {name: SimpleNamespace(worker_id=name, status=WorkerStatus.READY)
               for name in ("a", "b")}
    selected = []

    def execute(name):
        worker = workers[name]
        other = "b" if name == "a" else "a"
        selected.append(name)
        assert worker.status == WorkerStatus.BUSY
        assert workers[other].status == WorkerStatus.READY
        current = snapshot(pair)
        assert current[name]["status"] == "busy"
        assert current[other]["status"] == "ready"
        cnn = kind == "cnn"
        result = {
            "worker_id": name, "gpu": "Tesla T4", "cuda_version": "simulated",
            "torch_version": "simulated", "device": "cuda:0", "status": "passed",
            "model": "SmallCNN" if cnn else "SmallMLP",
            "input_shape": [32, 3, 32, 32] if cnn else [128, 784],
            "output_shape": [32, 10] if cnn else [128, 10],
            "batch_size": 32 if cnn else 128, "forward_passes": 100,
            "total_gpu_time_ms": 100, "average_inference_ms": 1,
            "peak_memory_mb": 8,
        }
        if cnn:
            result["throughput_images_per_second"] = 32000
        return result

    monkeypatch.setattr(f"worker.diagnostics.run_{kind}_test", execute)
    with TestClient(create_diagnostic_app(workers["a"])) as a, \
            TestClient(create_diagnostic_app(workers["b"])) as b:
        destinations = {f"https://worker-a.example/{kind}-test": a,
                        f"https://worker-b.example/{kind}-test": b}

        def forward(url, **kwargs):
            headers = kwargs.pop("headers")
            assert len(headers["X-ColabCluster-Request-ID"]) == 32
            assert kwargs == {"json": {}, "timeout": (5, 120), "allow_redirects": False}
            return destinations[url].post(f"/{kind}-test", json={}, headers=headers)

        monkeypatch.setattr(f"controller.{kind}_test.requests.post", forward)
        for name in ("a", "b"):
            before = snapshot(pair)
            response = pair.post(f"/workers/{name}/{kind}-test", json={})
            assert response.status_code == 200, response.text
            assert response.json()["worker_id"] == name
            after = snapshot(pair)
            execution = after[name]["metadata"].pop("execution")
            assert execution["state"] == "completed"
            assert len(execution["request_id"]) == 32
            assert after == before
            assert all(w.status == WorkerStatus.READY for w in workers.values())
    assert selected == ["a", "b"]
