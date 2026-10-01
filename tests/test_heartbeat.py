"""Heartbeat, telemetry, and liveness tests with controlled time and no GPU."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from common.schemas import HardwareInfo, WorkerHeartbeat, WorkerRegistration, WorkerStatus
from controller import liveness
from controller.main import create_app
from controller.registry import UnknownWorkerError, WorkerRegistry
from worker import hardware, heartbeat
from worker.client import WorkerClient, WorkerClientError
from worker.worker import Worker


def make_registry(status=WorkerStatus.READY):
    registry = WorkerRegistry()
    worker = registry.register_worker(WorkerRegistration(
        worker_id="test", gpu="None", gpu_memory=0, status=status,
    ))
    return registry, worker


@pytest.mark.parametrize("status", [WorkerStatus.READY, WorkerStatus.BUSY, WorkerStatus.ERROR])
def test_staleness_boundary_and_fresh_registration(status):
    registry, original = make_registry(status)
    assert registry.expire_stale_workers(30, original.last_seen + timedelta(seconds=30)) == []
    assert registry.get_worker("test").status == status
    later = original.last_seen + timedelta(seconds=31)
    assert registry.expire_stale_workers(30, later) == ["test"]
    assert registry.get_workers() == []
    assert registry.expire_stale_workers(30, later) == []
    with pytest.raises(UnknownWorkerError):
        registry.update_heartbeat("test", status=status)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = later
        revived = registry.register_worker(WorkerRegistration(
            worker_id="test", gpu="None", gpu_memory=0, status=WorkerStatus.READY,
        ))
    assert revived.status == WorkerStatus.READY
    assert revived.registered_at == revived.last_seen == later
    assert registry.expire_stale_workers(30, later) == []


@pytest.mark.parametrize("status", ["ready", "busy"])
@pytest.mark.parametrize("metrics", [
    {"gpu_utilization": 73.5, "gpu_memory_used": 4.0},
    {"gpu_utilization": None, "gpu_memory_used": None},
])
def test_api_heartbeat_metrics_and_status(status, metrics):
    with TestClient(create_app()) as client:
        original = client.post("/workers/register", json={
            "worker_id": "test", "gpu": "test", "gpu_memory": 8, "status": status,
        }).json()["worker"]
        response = client.post("/workers/heartbeat", json={"worker_id": "test", **metrics})
        assert response.status_code == 200
        result = response.json()
        worker = result["worker"]
        assert worker["status"] == status
        assert worker["last_seen"] == result["server_timestamp"]
        assert worker["registered_at"] == original["registered_at"]
        for key, value in metrics.items():
            assert worker[key] == value
        assert client.get("/workers").json()["workers"][0] == worker
        cleared = client.post("/workers/heartbeat", json={"worker_id": "test"}).json()["worker"]
        assert cleared["gpu_utilization"] is None
        assert cleared["gpu_memory_used"] is None


def test_lifespan_owns_monitor(monkeypatch):
    monkeypatch.setenv("COLABCLUSTER_WORKER_TIMEOUT", "45")
    app = create_app()
    assert app.state.worker_timeout == 45
    with TestClient(app):
        task = app.state.liveness_task
        assert not task.done()
    assert task.done() and task.cancelled()


def test_monitor_runs_registry_check_without_real_wait(monkeypatch):
    registry, original = make_registry()
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError()])
    monkeypatch.setattr(liveness.asyncio, "sleep", sleep)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = original.last_seen + timedelta(seconds=31)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(liveness.monitor_workers(registry, 30))
    assert registry.get_workers() == []
    assert sleep.await_args_list[0].args == (5.0,)


def test_interval_and_timeout_configuration(monkeypatch):
    monkeypatch.delenv("COLABCLUSTER_HEARTBEAT_INTERVAL", raising=False)
    monkeypatch.delenv("COLABCLUSTER_WORKER_TIMEOUT", raising=False)
    assert heartbeat.get_heartbeat_interval() == 10
    assert create_app().state.worker_timeout == 30
    monkeypatch.setenv("COLABCLUSTER_HEARTBEAT_INTERVAL", "2.5")
    assert heartbeat.get_heartbeat_interval() == 2.5


@pytest.mark.parametrize("value", ["", "oops", "0", "-1", "nan", "inf"])
def test_invalid_durations(monkeypatch, value):
    monkeypatch.setenv("COLABCLUSTER_HEARTBEAT_INTERVAL", value)
    monkeypatch.setenv("COLABCLUSTER_WORKER_TIMEOUT", value)
    with pytest.raises(ValueError, match="COLABCLUSTER_HEARTBEAT_INTERVAL"):
        heartbeat.get_heartbeat_interval()
    with pytest.raises(ValueError, match="COLABCLUSTER_WORKER_TIMEOUT"):
        create_app()


def test_loop_retries_next_interval(monkeypatch, caplog):
    worker = SimpleNamespace(worker_id="test")
    client = Mock()
    client.heartbeat.side_effect = [WorkerClientError("offline"), Mock()]
    sleep = Mock(side_effect=[None, KeyboardInterrupt()])
    monkeypatch.setattr(heartbeat.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        heartbeat.run_heartbeat_loop(worker, client, 2.5)
    assert client.heartbeat.call_count == 2
    assert sleep.call_count == 2
    assert sleep.call_args.args == (2.5,)
    assert "retrying on next interval" in caplog.text
    client.register.assert_not_called()


@pytest.mark.parametrize("metrics", [(50.0, 3.0), (None, None)])
def test_client_heartbeat_payload(monkeypatch, metrics):
    monkeypatch.setattr("worker.worker.get_hardware_info", lambda: HardwareInfo(
        cpu_count=1, ram_total=1, gpu="test", gpu_memory=4, cuda_available=True,
        torch_version="test", python_version="test", platform="test",
    ))
    worker = Worker("test")
    worker.status = WorkerStatus.BUSY
    monkeypatch.setattr("worker.client.get_gpu_metrics", lambda available: metrics)
    client = WorkerClient()
    post = Mock(return_value={"success": True, "worker_id": "test", "message": "ok",
                              "server_timestamp": datetime.now(timezone.utc).isoformat()})
    monkeypatch.setattr(client, "_post", post)
    assert client.heartbeat(worker).success
    payload = WorkerHeartbeat.model_validate(post.call_args.args[1])
    assert post.call_args.args[0] == "heartbeat"
    assert payload.status == WorkerStatus.BUSY
    assert payload.timestamp.utcoffset() == timedelta(0)
    assert (payload.gpu_utilization, payload.gpu_memory_used) == metrics
    post.return_value = {}
    with pytest.raises(WorkerClientError, match="invalid heartbeat response"):
        client.heartbeat(worker)


def test_gpu_metrics_are_optional(monkeypatch):
    cuda = Mock()
    cuda.utilization.return_value = 50
    cuda.mem_get_info.return_value = (4 * 1024**3, 8 * 1024**3)
    importer = Mock(return_value=SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(hardware.importlib, "import_module", importer)
    assert hardware.get_gpu_metrics(False) == (None, None)
    importer.assert_not_called()
    assert hardware.get_gpu_metrics(True) == (50, 4)
    cuda.utilization.side_effect = RuntimeError("no NVML")
    assert hardware.get_gpu_metrics(True) == (None, 4)
    cuda.mem_get_info.side_effect = RuntimeError("no CUDA")
    assert hardware.get_gpu_metrics(True) == (None, None)
    importer.side_effect = ImportError("no torch")
    assert hardware.get_gpu_metrics(True) == (None, None)


def test_shutdown_unregister_failure_is_nonfatal(monkeypatch, caplog):
    from worker import colab_worker
    worker = SimpleNamespace(worker_id="test")
    client = Mock()
    client.register.return_value.model_dump_json.return_value = "{}"
    client.unregister.side_effect = WorkerClientError("controller down")
    monkeypatch.setattr(colab_worker, "Worker", lambda: worker)
    monkeypatch.setattr(colab_worker, "print_worker_info", Mock())
    monkeypatch.setattr(colab_worker, "WorkerClient", lambda: client)
    monkeypatch.setattr(colab_worker, "run_heartbeat_loop", Mock(side_effect=KeyboardInterrupt))
    assert colab_worker.main() == 0
    client.unregister.assert_called_once_with("test", reason="worker shutdown")
    assert "liveness timeout will apply" in caplog.text


def test_api_expired_worker_requires_fresh_registration():
    app = create_app()
    with TestClient(app) as client:
        payload = {"worker_id": "test", "gpu": "None", "gpu_memory": 0, "status": "ready"}
        assert client.post("/workers/register", json=payload).status_code == 201
        original = app.state.registry.get_worker("test")
        app.state.registry.expire_stale_workers(30, original.last_seen + timedelta(seconds=31))
        assert client.get("/workers").json() == {"workers": []}
        response = client.post("/workers/heartbeat", json={"worker_id": "test", "status": "ready"})
        assert response.status_code == 404
        assert client.post("/workers/unregister", json={"worker_id": "test"}).status_code == 404
        assert client.post("/workers/register", json=payload).status_code == 201
        workers = client.get("/workers").json()["workers"]
        assert len(workers) == 1 and workers[0]["status"] == "ready"


def test_mixed_workers_expire_only_stale_entries(caplog):
    registry = WorkerRegistry()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = start
        for worker_id in ("stale-a", "stale-b", "active-a", "active-b"):
            registry.register_worker(WorkerRegistration(
                worker_id=worker_id, gpu="None", gpu_memory=0, status="ready",
            ))
        clock.now.return_value = start + timedelta(seconds=20)
        registry.update_heartbeat("active-a", status=WorkerStatus.BUSY)
        registry.update_heartbeat("active-b")
    assert len(registry.get_workers()) == 4
    assert registry.expire_stale_workers(30, start + timedelta(seconds=31)) == ["stale-a", "stale-b"]
    assert {w.worker_id: w.status for w in registry.get_workers()} == {
        "active-a": WorkerStatus.BUSY, "active-b": WorkerStatus.READY,
    }
    assert "[Registry] Worker expired and removed: stale-a" in caplog.text


def test_api_lists_all_active_workers():
    with TestClient(create_app()) as client:
        for worker_id in ("first", "second"):
            assert client.post("/workers/register", json={
                "worker_id": worker_id, "gpu": "None", "gpu_memory": 0, "status": "ready",
            }).status_code == 201
        assert {w["worker_id"] for w in client.get("/workers").json()["workers"]} == {"first", "second"}
        assert client.post("/workers/unregister", json={"worker_id": "first"}).status_code == 200
        assert [w["worker_id"] for w in client.get("/workers").json()["workers"]] == ["second"]


def test_list_workers_expires_before_monitor_and_allows_reregistration():
    app = create_app()
    start = datetime.now(timezone.utc)
    with TestClient(app) as client, patch("controller.registry.datetime") as clock:
        clock.now.return_value = start
        payload = {"worker_id": "stale", "gpu": "None", "gpu_memory": 0, "status": "ready"}
        client.post("/workers/register", json=payload)
        clock.now.return_value = start + timedelta(seconds=20)
        client.post("/workers/register", json={**payload, "worker_id": "alive"})
        clock.now.return_value = start + timedelta(seconds=31)
        assert [w["worker_id"] for w in client.get("/workers").json()["workers"]] == ["alive"]
        with pytest.raises(UnknownWorkerError):
            app.state.registry.get_worker("stale")
        assert client.post("/workers/register", json=payload).status_code == 201
        assert len(client.get("/workers").json()["workers"]) == 2


def test_list_removes_explicit_offline_records():
    with TestClient(create_app()) as client:
        client.post("/workers/register", json={
            "worker_id": "offline", "gpu": "None", "gpu_memory": 0, "status": "offline",
        })
        assert client.get("/workers").json() == {"workers": []}
        assert client.post("/workers/heartbeat", json={"worker_id": "offline"}).status_code == 404
