"""Offline HTTP-client and single-worker workflow checks."""

from unittest.mock import Mock

import pytest
import requests

from common.schemas import HardwareInfo, WorkerRegistrationResponse
from worker import colab_worker
from worker.client import WorkerClient, WorkerClientError
from worker.worker import Worker


@pytest.fixture
def worker(monkeypatch):
    hardware = HardwareInfo(cpu_count=4, ram_total=16, gpu="Test GPU", gpu_memory=8,
                            cuda_available=True, torch_version="test", python_version="3.10",
                            platform="Linux")
    monkeypatch.setattr("worker.worker.get_hardware_info", lambda: hardware)
    return Worker("TEST-01")


@pytest.fixture
def post(monkeypatch):
    response = Mock(status_code=201)
    response.json.return_value = {"success": True, "worker_id": "TEST-01", "message": "Registered"}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    return post


def test_payload_and_success(worker, post):
    client = WorkerClient("https://controller.example/base/")
    response = client.register(worker)
    assert isinstance(response, WorkerRegistrationResponse)
    assert response.success and response.worker_id == "TEST-01"
    post.assert_called_once_with(
        "https://controller.example/base/workers/register",
        json={"worker_id": "TEST-01", "gpu": "Test GPU", "gpu_memory": 8,
              "status": "ready", "cpu_count": 4, "ram_total": 16, "cuda_available": True,
              "metadata": {"platform": "Linux", "python_version": "3.10", "torch_version": "test"}},
        timeout=(5, 15), allow_redirects=False,
    )
    post.return_value.close.assert_called_once()


def test_url_configuration(monkeypatch):
    monkeypatch.delenv("COLABCLUSTER_CONTROLLER_URL", raising=False)
    assert WorkerClient().controller_url == "http://127.0.0.1:8000"
    monkeypatch.setenv("COLABCLUSTER_CONTROLLER_URL", "https://controller.example/")
    assert WorkerClient().controller_url == "https://controller.example"


@pytest.mark.parametrize("url", ["", "localhost:8000", "ftp://example.org", "https://u:p@example.org",
                                    "http://example.org?x=y", "http://example.org/#x", "http://a:bad"])
def test_invalid_url(url):
    with pytest.raises(ValueError):
        WorkerClient(url)


@pytest.mark.parametrize("error, match", [
    (requests.ConnectionError("refused"), "Unable to connect"),
    (requests.Timeout("slow"), "timed out"),
    (requests.RequestException("failed"), "request failed"),
])
def test_transport_errors(worker, post, error, match):
    post.side_effect = error
    with pytest.raises(WorkerClientError, match=match):
        WorkerClient().register(worker)
    post.assert_called_once()


@pytest.mark.parametrize("code", [400, 404, 409, 422, 500, 503, 302])
def test_http_errors(worker, post, code):
    post.return_value.status_code = code
    with pytest.raises(WorkerClientError, match="already registered" if code == 409 else f"HTTP {code}"):
        WorkerClient().register(worker)
    post.assert_called_once()
    post.return_value.close.assert_called_once()


@pytest.mark.parametrize("body", [None, {}, {"success": True},
    {"success": False, "worker_id": "TEST-01", "message": "denied"},
    {"success": True, "worker_id": "different", "message": "ok"},
])
def test_invalid_acknowledgements(worker, post, body):
    post.return_value.json.return_value = body
    with pytest.raises(WorkerClientError):
        WorkerClient().register(worker)


def test_invalid_json(worker, post):
    post.return_value.json.side_effect = ValueError("not json")
    with pytest.raises(WorkerClientError, match="invalid JSON"):
        WorkerClient().register(worker)


def test_unregister(post):
    post.return_value.status_code = 200
    response = WorkerClient().unregister("TEST-01", reason="test complete")
    assert response.success
    assert post.call_args.kwargs["json"] == {"worker_id": "TEST-01", "reason": "test complete"}
    assert post.call_args.args[0].endswith("/workers/unregister")


def test_registered_worker_starts_heartbeat_and_unregisters(worker, post, monkeypatch, capsys):
    monkeypatch.setattr(colab_worker, "Worker", lambda: worker)
    loop = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(colab_worker, "run_heartbeat_loop", loop)
    assert colab_worker.main() == 0
    assert post.call_count == 2
    assert post.call_args.args[0].endswith("/workers/unregister")
    loop.assert_called_once()
    assert "Registration successful" in capsys.readouterr().out


def test_failed_registration_exits(worker, post, monkeypatch, capsys):
    monkeypatch.setattr(colab_worker, "Worker", lambda: worker)
    post.side_effect = requests.ConnectionError("offline")
    loop = Mock()
    monkeypatch.setattr(colab_worker, "run_heartbeat_loop", loop)
    assert colab_worker.main() == 1
    loop.assert_not_called()
    assert "Unable to connect" in capsys.readouterr().err
