"""Fixed NN execution and dashboard contracts, without requiring PyTorch/CUDA."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.schemas import NnTestRequest, NnTestResponse, WorkerStatus
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app
from worker import nn_test as nn_module


def result(worker_id="nn"):
    return NnTestResponse(worker_id=worker_id, gpu="Mock GPU", cuda_version="test",
                          torch_version="test", device="cuda:0", model="SmallMLP", input_shape=(128,784),
                          batch_size=128, forward_passes=100, status="passed", total_gpu_time_ms=10,
                          average_inference_ms=0.1, peak_memory_mb=4, output_shape=(128,10))


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def register(client, status="ready", metadata=None):
    return client.post("/workers/register", json={
        "worker_id":"nn", "gpu":"Mock GPU", "gpu_memory":16, "status":status,
        "cuda_available":True, "metadata":metadata,
    })


def test_dashboard_page(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for label in ("ColabCluster", "Distributed GPU Dashboard", "Run NN Test", "/workers", "/cluster"):
        assert label in response.text
    assert 'innerHTML' not in response.text


@pytest.mark.parametrize("body", [{"code":"print(1)"}, {"model":"other"},
                                    {"batch_size":1000}, {"forward_passes":1}])
def test_nn_request_rejects_custom_execution(client, body):
    with pytest.raises(ValidationError):
        NnTestRequest.model_validate(body)
    assert client.post("/workers/nn/nn-test",json=body).status_code == 422


def test_empty_request_and_response():
    assert NnTestRequest().model_dump() == {}
    response = result()
    assert response.model == "SmallMLP" and response.forward_passes == 100
    assert response.device == "cuda:0" and response.status == "passed"
    with pytest.raises(ValidationError):
        NnTestResponse.model_validate({**response.model_dump(), "output_shape":[128,99]})


def test_unknown_worker(client):
    assert client.post("/workers/missing/nn-test",json={}).status_code == 404


def test_missing_diagnostic_url_restores_status(client):
    register(client)
    assert client.post("/workers/nn/nn-test",json={}).status_code == 409
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"


@pytest.mark.parametrize("status", ["busy","error"])
def test_controller_requires_ready(client, status):
    register(client,status=status)
    assert client.post("/workers/nn/nn-test",json={}).status_code == 409
    assert client.get("/workers").json()["workers"][0]["status"] == status


def test_controller_forward_and_poll_busy(client, monkeypatch):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    response = Mock(status_code=200)
    response.json.return_value = result().model_dump(mode="json")
    def post(url, **kwargs):
        assert url == "https://worker.example/nn-test" and kwargs["json"] == {}
        assert kwargs["timeout"] == (5,120) and kwargs["allow_redirects"] is False
        assert client.get("/workers").json()["workers"][0]["status"] == "busy"
        assert client.get("/cluster").json()["busy_workers"] == 1
        return response
    monkeypatch.setattr("controller.nn_test.requests.post",post)
    assert client.post("/workers/nn/nn-test",json={}).json()["status"] == "passed"
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"
    response.close.assert_called_once()


@pytest.mark.parametrize("error,code", [(requests.Timeout(),504),(requests.ConnectionError(),502)])
def test_transport_failure_restores_status(client,monkeypatch,error,code):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    monkeypatch.setattr("controller.nn_test.requests.post",Mock(side_effect=error))
    assert client.post("/workers/nn/nn-test",json={}).status_code == code
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"


@pytest.mark.parametrize("body", [{}, result("wrong").model_dump(mode="json"),
                                    {**result().model_dump(mode="json"),"device":"cpu"}])
def test_invalid_remote_result(client,monkeypatch,body):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    response=Mock(status_code=200)
    response.json.return_value=body
    monkeypatch.setattr("controller.nn_test.requests.post",Mock(return_value=response))
    assert client.post("/workers/nn/nn-test",json={}).status_code == 502


def test_cleanup_does_not_resurrect_worker(client,monkeypatch):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    def forward(worker):
        client.post("/workers/unregister",json={"worker_id":"nn"})
        register(client,status="error")
        return result()
    monkeypatch.setattr("controller.api.forward_nn_test",forward)
    assert client.post("/workers/nn/nn-test",json={}).status_code == 200
    assert client.get("/workers").json()["workers"][0]["status"] == "error"


def test_mock_nn_cuda_execution(monkeypatch):
    torch=MagicMock()
    torch.cuda.is_available.return_value=True
    torch.cuda.get_device_name.return_value="Mock GPU"
    torch.cuda.max_memory_allocated.return_value=4*1024**2
    torch.version.cuda="test"
    torch.__version__="test"
    torch.randn.return_value.shape=(128,784)
    model=torch.nn.Sequential.return_value.to.return_value
    model.return_value.shape=(128,10)
    model.return_value.device="cuda:0"
    monkeypatch.setattr(nn_module.time,"perf_counter",Mock(side_effect=[1,1.01]))
    monkeypatch.setattr(nn_module.importlib,"import_module",Mock(return_value=torch))
    response=nn_module.run_nn_test("nn")
    assert response.total_gpu_time_ms == pytest.approx(10)
    assert response.average_inference_ms == pytest.approx(.1)
    assert response.peak_memory_mb == 4
    assert model.call_count == 103
    model.eval.assert_called_once()
    torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)
    assert torch.cuda.synchronize.call_count == 3
    torch.randn.assert_called_once_with(128,784,device="cuda:0",dtype=torch.float32)
    assert [call.args for call in torch.nn.Linear.call_args_list] == [(784,128),(128,64),(64,10)]


def test_missing_cuda_and_torch(monkeypatch):
    importer=Mock(side_effect=ImportError("missing"))
    monkeypatch.setattr(nn_module.importlib,"import_module",importer)
    with pytest.raises(nn_module.NnTestError,match="PyTorch"):
        nn_module.run_nn_test("nn")
    importer.side_effect=None
    torch=MagicMock()
    importer.return_value=torch
    torch.cuda.is_available.return_value=False
    with pytest.raises(nn_module.NnTestError,match="CUDA is unavailable"):
        nn_module.run_nn_test("nn")
    torch.nn.Sequential.assert_not_called()


def test_worker_shared_lock_and_status(monkeypatch):
    worker=SimpleNamespace(worker_id="nn",status=WorkerStatus.READY)
    entered,release=Event(),Event()
    def execute(worker_id):
        assert worker.status == WorkerStatus.BUSY
        entered.set()
        assert release.wait(5)
        return result()
    monkeypatch.setattr("worker.diagnostics.run_nn_test",execute)
    with TestClient(create_diagnostic_app(worker)) as client, ThreadPoolExecutor(1) as pool:
        future=pool.submit(client.post,"/nn-test",json={})
        try:
            assert entered.wait(5)
            assert client.post("/gpu-test",json={}).status_code == 409
            assert client.post("/nn-test",json={}).status_code == 409
        finally:
            release.set()
        assert future.result().status_code == 200
        assert worker.status == WorkerStatus.READY


def test_worker_error_restores_status(monkeypatch):
    worker=SimpleNamespace(worker_id="nn",status=WorkerStatus.READY)
    monkeypatch.setattr("worker.diagnostics.run_nn_test",Mock(side_effect=nn_module.NnTestError("failed")))
    with TestClient(create_diagnostic_app(worker)) as client:
        assert client.post("/nn-test",json={}).status_code == 503
        assert worker.status == WorkerStatus.READY
        assert client.post("/nn-test",json={"code":"bad"}).status_code == 422
