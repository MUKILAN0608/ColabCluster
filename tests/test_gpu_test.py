"""GPU diagnostics are tested with mocks; no CUDA operations run locally."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.schemas import GpuTestRequest, GpuTestResponse, WorkerStatus
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app
from worker.gpu_test import GpuTestError, run_gpu_test


def result():
    return dict(worker_id="gpu", gpu="Mock GPU", cuda_available=True, cuda_version="test",
                torch_version="test", matrix_size=256, iterations=2, total_time_seconds=0.2,
                average_time_ms=100, gpu_memory_allocated_gb=1, result_shape=[256, 256])


def test_defaults():
    assert GpuTestRequest().model_dump() == {"matrix_size": 4096, "iterations": 20}
    assert GpuTestResponse.model_validate(result()).result_shape == (256, 256)


@pytest.mark.parametrize("values", [dict(matrix_size=255), dict(matrix_size=8193),
    dict(iterations=0), dict(iterations=101), dict(matrix_size=256.5), dict(iterations=True)])
def test_invalid_request(values):
    with pytest.raises(ValidationError):
        GpuTestRequest(**values)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        c.post("/workers/register", json={"worker_id":"gpu", "gpu":"Mock GPU",
               "gpu_memory":16,"status":"ready", "cuda_available":True,
               "metadata":{"diagnostic_url":"https://worker.example"}})
        yield c


def test_forwarding(client, monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = result()
    post = Mock(return_value=response)
    monkeypatch.setattr("controller.gpu_test.requests.post", post)
    r = client.post("/workers/gpu/gpu-test", json={"matrix_size":256,"iterations":2})
    assert r.status_code == 200
    assert r.json()["worker_id"] == "gpu"
    post.assert_called_once_with("https://worker.example/gpu-test", json={"matrix_size":256,"iterations":2},
                                 timeout=(5,120), allow_redirects=False, headers=post.call_args.kwargs["headers"])
    assert len(post.call_args.kwargs["headers"]["X-ColabCluster-Request-ID"]) == 32
    response.close.assert_called_once()


def test_unknown_and_expired(client, monkeypatch):
    post = Mock()
    monkeypatch.setattr("controller.gpu_test.requests.post", post)
    assert client.post("/workers/missing/gpu-test", json={}).status_code == 404
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value = datetime.now(timezone.utc) + timedelta(seconds=31)
        assert client.post("/workers/gpu/gpu-test", json={}).status_code == 404
    post.assert_not_called()


@pytest.mark.parametrize("error, code", [(requests.Timeout(),504), (requests.ConnectionError(),502)])
def test_forward_errors(client, monkeypatch, error, code):
    monkeypatch.setattr("controller.gpu_test.requests.post", Mock(side_effect=error))
    assert client.post("/workers/gpu/gpu-test", json={}).status_code == code


@pytest.mark.parametrize("body", [{}, dict(result(), worker_id="other"),
    dict(result(), cuda_available=False), dict(result(), result_shape=[1,1])])
def test_bad_results(client, monkeypatch, body):
    response = Mock(status_code=200)
    response.json.return_value = body
    monkeypatch.setattr("controller.gpu_test.requests.post", Mock(return_value=response))
    assert client.post("/workers/gpu/gpu-test", json={"matrix_size":256,"iterations":2}).status_code == 502


def test_unconfigured_worker(client):
    client.post("/workers/register", json={"worker_id":"old", "gpu":"None","gpu_memory":0,"status":"ready"})
    assert client.post("/workers/old/gpu-test", json={}).status_code == 409


def test_no_torch_in_controller():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Controller tried importing torch')
sys.meta_path.insert(0, BlockTorch())
from controller.main import create_app
assert create_app()
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_missing_torch_and_cuda(monkeypatch):
    importer = Mock(side_effect=ImportError("absent"))
    monkeypatch.setattr("worker.gpu_test.importlib.import_module", importer)
    with pytest.raises(GpuTestError, match="PyTorch is unavailable"):
        run_gpu_test("gpu", GpuTestRequest())
    importer.side_effect = None
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    importer.return_value = torch
    with pytest.raises(GpuTestError, match="CUDA is unavailable"):
        run_gpu_test("gpu", GpuTestRequest())
    torch.randn.assert_not_called()


def test_mock_cuda_benchmark(monkeypatch):
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_name.return_value = "Mock GPU"
    torch.cuda.memory_allocated.return_value = 1024**3
    torch.version.cuda = "test"
    torch.__version__ = "test"
    torch.mm.return_value.shape = (256,256)
    from worker import gpu_test as module
    monkeypatch.setattr(module.time, "perf_counter", Mock(side_effect=[1,1.2]))
    monkeypatch.setattr(module.importlib, "import_module", Mock(return_value=torch))
    response = run_gpu_test("gpu", GpuTestRequest(matrix_size=256,iterations=2))
    assert response.average_time_ms == pytest.approx(100)
    assert response.result_shape == (256,256)
    assert torch.mm.call_count == 5
    assert torch.cuda.synchronize.call_count == 2
    assert torch.randn.call_args.kwargs["device"] == "cuda:0"


def test_worker_error_and_busy_restoration(monkeypatch):
    worker = Mock(worker_id="gpu", status=WorkerStatus.READY)
    benchmark = Mock(side_effect=GpuTestError("CUDA unavailable"))
    monkeypatch.setattr("worker.diagnostics.run_gpu_test", benchmark)
    with TestClient(create_diagnostic_app(worker)) as c:
        assert c.post("/gpu-test",json={}).status_code == 503
        assert worker.status == WorkerStatus.READY
        worker.status = WorkerStatus.BUSY
        assert c.post("/gpu-test",json={}).status_code == 409
        assert worker.status == WorkerStatus.BUSY


def test_full_http_forward_to_worker_without_cuda(monkeypatch):
    worker = Mock(worker_id="gpu", status=WorkerStatus.READY)
    monkeypatch.setattr("worker.diagnostics.run_gpu_test", lambda worker_id, payload: GpuTestResponse(**result()))
    with TestClient(create_diagnostic_app(worker)) as remote, TestClient(create_app()) as controller:
        controller.post("/workers/register", json={"worker_id":"gpu", "gpu":"Mock GPU",
            "gpu_memory":16, "status":"ready", "metadata":{"diagnostic_url":"https://worker.example"}})
        def forward(url, **kwargs):
            assert url == "https://worker.example/gpu-test"
            return remote.post("/gpu-test", json=kwargs["json"])
        monkeypatch.setattr("controller.gpu_test.requests.post", forward)
        response = controller.post("/workers/gpu/gpu-test", json={"matrix_size":256,"iterations":2})
        assert response.status_code == 200
        assert response.json()["result_shape"] == [256,256]
        assert worker.status == WorkerStatus.READY
