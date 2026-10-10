"""Transport/lifecycle regressions with no external network or CUDA."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from socket import timeout as SocketTimeout
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from common.config import http_timeout
from common.schemas import WorkerStatus
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app
from test_single_worker_inference import register, result


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        register(client, "a")
        yield client


def record(client):
    return client.get("/workers").json()["workers"][0]


@pytest.mark.parametrize("kind,expected", [("inference", (5,120)), ("preflight", (5,15)), ("heartbeat", (5,15))])
def test_timeout_defaults_and_overrides(monkeypatch, kind, expected):
    assert http_timeout(kind) == expected
    monkeypatch.setenv(f"COLABCLUSTER_{kind.upper()}_CONNECT_TIMEOUT", "7.5")
    monkeypatch.setenv(f"COLABCLUSTER_{kind.upper()}_READ_TIMEOUT", "42")
    assert http_timeout(kind) == (7.5,42)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "text"])
def test_invalid_timeout_fails_before_startup(monkeypatch, value):
    monkeypatch.setenv("COLABCLUSTER_INFERENCE_READ_TIMEOUT", value)
    with pytest.raises(ValueError, match="COLABCLUSTER_INFERENCE_READ_TIMEOUT"):
        create_app()


def test_dispatch_propagates_config(client, monkeypatch):
    monkeypatch.setenv("COLABCLUSTER_INFERENCE_CONNECT_TIMEOUT", "9")
    monkeypatch.setenv("COLABCLUSTER_INFERENCE_READ_TIMEOUT", "101")
    reply = Mock(status_code=200)
    reply.json.return_value = result()
    post = Mock(return_value=reply)
    monkeypatch.setattr("controller.transport.requests.post", post)
    assert client.post("/inference/single-worker", json={}).status_code == 200
    assert post.call_args.kwargs["timeout"] == (9,101)
    assert post.call_args.kwargs["allow_redirects"] is False
    assert record(client)["metadata"]["execution"]["request_id"] == post.call_args.kwargs["headers"]["X-ColabCluster-Request-ID"]
    post.assert_called_once()


@pytest.mark.parametrize("error,code,kind,state", [
    (requests.ConnectTimeout("connect failed"),504,"connect_timeout","ready"),
    (requests.ReadTimeout("reply lost"),504,"read_timeout","error"),
    (requests.Timeout("ambiguous"),504,"transport_timeout","error"),
    (requests.ConnectionError("connection reset"),502,"connection_error","error"),
])
def test_transport_errors_are_distinct_and_never_retry(client, monkeypatch, error, code, kind, state):
    post = Mock(side_effect=error)
    monkeypatch.setattr("controller.transport.requests.post", post)
    response = client.post("/inference/single-worker", json={})
    assert response.status_code == code
    detail = response.json()["detail"]
    assert kind in detail and str(error) in detail and "request_id=" in detail
    assert "connect=5s read=120s" in detail
    assert record(client)["status"] == state
    post.assert_called_once()
    if state == "error":
        assert record(client)["metadata"]["execution"]["state"] == "unknown"
        assert client.post("/inference/single-worker", json={}).status_code == 409
        post.assert_called_once()


@pytest.mark.parametrize("http_status,api_status", [(504,504),(503,503),(500,502),(409,409),(302,502)])
def test_proxy_errors_do_not_release_worker(client, monkeypatch, http_status, api_status):
    reply = Mock(status_code=http_status, headers={})
    reply.json.return_value = {"detail":"upstream reply unavailable"}
    monkeypatch.setattr("controller.transport.requests.post", Mock(return_value=reply))
    response = client.post("/inference/single-worker", json={})
    assert response.status_code == api_status
    assert f"HTTP {http_status}" in response.json()["detail"]
    assert "upstream reply unavailable" in response.json()["detail"]
    assert record(client)["status"] == "error"
    reply.close.assert_called_once()


def quarantine(client, monkeypatch):
    monkeypatch.setattr("controller.transport.requests.post", Mock(side_effect=requests.ReadTimeout("late")))
    assert client.post("/inference/single-worker", json={}).status_code == 504
    return record(client)["metadata"]["execution"]["request_id"]


def test_ready_heartbeats_expiry_and_unregister_cannot_clear_unknown(client, monkeypatch):
    quarantine(client, monkeypatch)
    assert client.post("/workers/heartbeat", json={"worker_id":"a", "status":"ready"}).status_code == 200
    registry = client.app.state.registry
    registry.expire_stale_workers(30, datetime.now(timezone.utc)+timedelta(days=1))
    assert record(client)["status"] == "error"
    assert client.post("/workers/unregister", json={"worker_id":"a"}).status_code == 409
    assert client.post("/workers/register", json={"worker_id":"a", "gpu":"Tesla T4", "gpu_memory":14, "status":"ready"}).status_code == 409
    for route in ("gpu-test", "nn-test", "cnn-test"):
        assert client.post(f"/workers/a/{route}", json={}).status_code == 409


@pytest.mark.parametrize("state,available,expected", [
    ("running",False,"error"), ("unknown",True,"error"),
    ("completed",False,"error"), ("completed",True,"ready"), ("failed",True,"ready"),
])
def test_reconciliation_requires_matching_terminal_execution_and_free_slot(client, monkeypatch, state, available, expected):
    request_id = quarantine(client, monkeypatch)
    reply = Mock(status_code=200)
    reply.json.return_value = dict(worker_id="a", request_id=request_id, state=state, slot_available=available)
    get = Mock(return_value=reply)
    monkeypatch.setattr("controller.reconciliation.requests.get", get)
    response = client.post("/workers/a/reconcile")
    assert response.status_code == 200
    assert response.json()["status"] == expected
    get.assert_called_once_with(f"https://a.example/executions/{request_id}", timeout=(5,15), allow_redirects=False)
    reply.close.assert_called_once()


@pytest.mark.parametrize("failure", ["missing", "identity", "request_id", "json", "timeout", "state", "no_slot"])
def test_unconfirmed_reconciliation_remains_quarantined(client, monkeypatch, failure):
    request_id = quarantine(client, monkeypatch)
    data = dict(worker_id="a", request_id=request_id, state="completed", slot_available=True)
    if failure == "identity": data["worker_id"] = "other"
    if failure == "request_id": data["request_id"] = "old"
    if failure == "state": data["state"] = "idle"
    if failure == "no_slot": data.pop("slot_available")
    reply = Mock(status_code=404 if failure == "missing" else 200)
    reply.json.return_value = data
    if failure == "json": reply.json.side_effect = ValueError("broken")
    get = Mock(return_value=reply)
    if failure == "timeout": get.side_effect = requests.ReadTimeout("no reply")
    monkeypatch.setattr("controller.reconciliation.requests.get", get)
    assert client.post("/workers/a/reconcile").status_code == 502
    assert record(client)["status"] == "error"


def test_remote_execution_continues_after_lost_reply_then_reconciles(client, monkeypatch):
    worker = SimpleNamespace(worker_id="a", status=WorkerStatus.READY)
    entered, release = Event(), Event()
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        return result()
    monkeypatch.setattr("worker.diagnostics.run_inference", execute)
    with TestClient(create_diagnostic_app(worker)) as remote, ThreadPoolExecutor(1) as pool:
        tasks = []
        def post(url, **kwargs):
            tasks.append(pool.submit(remote.post, "/inference/single-worker", json=kwargs["json"], headers=kwargs["headers"]))
            assert entered.wait(5)
            raise requests.ReadTimeout("simulated caller lost reply")
        monkeypatch.setattr("controller.transport.requests.post", post)
        monkeypatch.setattr("controller.reconciliation.requests.get", lambda url, **kwargs: remote.get("/executions/"+url.rsplit("/",1)[1]))
        try:
            assert client.post("/inference/single-worker", json={}).status_code == 504
            request_id = record(client)["metadata"]["execution"]["request_id"]
            assert worker.status == WorkerStatus.BUSY
            assert client.post("/workers/a/reconcile").json()["status"] == "error"
            assert client.post("/inference/single-worker", json={}).status_code == 409
            assert remote.post("/inference/single-worker", json={}, headers={"X-ColabCluster-Request-ID":request_id}).status_code == 409
            assert remote.post("/nn-test", json={}).status_code == 409
        finally:
            release.set()
        assert tasks[0].result(timeout=5).status_code == 200
        assert client.post("/workers/a/reconcile").json()["status"] == "ready"
        assert remote.post("/inference/single-worker", json={}, headers={"X-ColabCluster-Request-ID":request_id}).status_code == 409
        assert len(calls) == 1


def test_worker_execution_error_is_correlated_and_restores_slot(client, monkeypatch):
    from worker.cnn_test import CnnTestError
    worker = SimpleNamespace(worker_id="a", status=WorkerStatus.READY)
    monkeypatch.setattr("worker.diagnostics.run_inference", Mock(side_effect=CnnTestError("CUDA allocation failed")))
    with TestClient(create_diagnostic_app(worker)) as remote:
        monkeypatch.setattr("controller.transport.requests.post", lambda url, **kwargs: remote.post(
            "/inference/single-worker", json=kwargs["json"], headers=kwargs["headers"]))
        response = client.post("/inference/single-worker", json={})
        assert response.status_code == 503
        assert "worker_execution_error" in response.json()["detail"]
        assert "CUDA allocation failed" in response.json()["detail"]
        assert record(client)["status"] == "ready"
        execution = record(client)["metadata"]["execution"]
        assert execution["state"] == "failed"
        assert remote.get("/executions/"+execution["request_id"]).json()["state"] == "failed"


def test_https_connect_phase_can_be_reported_as_read_timeout(monkeypatch):
    """Reproduce the installed urllib3 mechanism, not the historical network event."""
    from urllib3 import HTTPSConnectionPool
    from urllib3.exceptions import ReadTimeoutError
    from urllib3.util import Timeout
    pool = HTTPSConnectionPool("worker.example")
    conn = Mock(proxy=None)
    monkeypatch.setattr(pool, "_validate_conn", Mock(side_effect=SocketTimeout("TLS handshake stalled")))
    with pytest.raises(ReadTimeoutError, match="read timeout=5"):
        pool._make_request(conn, "GET", "/openapi.json", timeout=Timeout(connect=5, read=120))
    assert conn.timeout == 5


def test_preflight_and_heartbeat_have_independent_budgets(monkeypatch):
    from scripts.benchmark_scaling import get_json
    from worker.client import WorkerClient
    monkeypatch.setenv("COLABCLUSTER_PREFLIGHT_CONNECT_TIMEOUT", "8")
    monkeypatch.setenv("COLABCLUSTER_PREFLIGHT_READ_TIMEOUT", "19")
    monkeypatch.setenv("COLABCLUSTER_HEARTBEAT_CONNECT_TIMEOUT", "6")
    monkeypatch.setenv("COLABCLUSTER_HEARTBEAT_READ_TIMEOUT", "16")
    reply = Mock(status_code=200)
    reply.json.return_value = {"success":True}
    get, post = Mock(return_value=reply), Mock(return_value=reply)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    assert get_json("https://worker.example/openapi.json") == {"success":True}
    WorkerClient("http://controller.example")._post("heartbeat", {}, "a")
    assert get.call_args.kwargs["timeout"] == (8,19)
    assert post.call_args.kwargs["timeout"] == (6,16)
    get.side_effect = requests.ReadTimeout("read timeout=8")
    with pytest.raises(requests.ReadTimeout, match="Preflight GET.*connect=8s read=19s"):
        get_json("https://worker.example/openapi.json")


def test_external_caller_giving_up_does_not_release_controller_reservation(client, monkeypatch):
    entered, release = Event(), Event()
    def post(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        reply = Mock(status_code=200)
        reply.json.return_value = result()
        return reply
    monkeypatch.setattr("controller.transport.requests.post", post)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(client.post, "/inference/single-worker", json={})
        try:
            assert entered.wait(5)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.01)
            registry = client.app.state.registry
            assert registry.expire_stale_workers(30, datetime.now(timezone.utc)+timedelta(days=1)) == []
            assert record(client)["status"] == "busy"
            assert client.post("/workers/unregister", json={"worker_id":"a"}).status_code == 409
            assert client.post("/inference/single-worker", json={}).status_code == 409
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    assert record(client)["status"] == "ready"


def test_worker_ledger_never_evicts_duplicate_protection(monkeypatch):
    monkeypatch.setattr("worker.diagnostics.EXECUTION_HISTORY_LIMIT", 1)
    execute = Mock(return_value=result())
    monkeypatch.setattr("worker.diagnostics.run_inference", execute)
    worker = SimpleNamespace(worker_id="a", status=WorkerStatus.READY)
    with TestClient(create_diagnostic_app(worker)) as remote:
        first = {"X-ColabCluster-Request-ID":"a"*32}
        second = {"X-ColabCluster-Request-ID":"b"*32}
        assert remote.post("/inference/single-worker", json={}, headers=first).status_code == 200
        assert remote.post("/inference/single-worker", json={}, headers=second).status_code == 503
        assert remote.post("/inference/single-worker", json={}, headers=first).status_code == 409
        assert remote.get("/executions/"+"b"*32).json()["state"] == "unknown"
        execute.assert_called_once()


def test_old_cleanup_cannot_release_new_registration(client):
    """Retain the identity guard even though public unregister now rejects BUSY."""
    registry = client.app.state.registry
    dispatched, original = registry.begin_single_worker_inference(30)
    registry.finish_nn_test("a", original, dispatched)
    registry.unregister_worker("a")
    register(client, "a", status="error")
    registry.finish_nn_test("a", original, dispatched)
    assert record(client)["status"] == "error"


def test_reconciliation_rejects_another_execution(client, monkeypatch):
    from controller.registry import WorkerNotReadyError
    request_id = quarantine(client, monkeypatch)
    worker = client.app.state.registry.get_worker("a")
    with pytest.raises(WorkerNotReadyError):
        client.app.state.registry.reconcile_execution("a", worker.registered_at, "not-"+request_id, "completed", True)
    assert record(client)["status"] == "error"
