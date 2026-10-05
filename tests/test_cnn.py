"""CNN diagnostic coverage with mocked GPU operations and HTTP calls."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common.schemas import CnnTestRequest, CnnTestResponse, WorkerStatus
from controller.main import create_app
from worker import cnn_test as module
from worker.diagnostics import create_diagnostic_app


def result(worker_id="cnn"):
    return CnnTestResponse(worker_id=worker_id,gpu="Mock GPU",cuda_version="test",torch_version="test",
        device="cuda:0",model="SmallCNN",input_shape=(32,3,32,32),batch_size=32,forward_passes=100,
        total_gpu_time_ms=100,average_inference_ms=1,throughput_images_per_second=32000,
        peak_memory_mb=8,output_shape=(32,10),status="passed")


def test_request_and_response():
    assert CnnTestRequest().model_dump() == {}
    assert result().output_shape == (32,10)
    assert result().model_dump(mode="json")["input_shape"] == [32,3,32,32]


@pytest.mark.parametrize("body", [{"code":"bad"},{"model":"custom"},{"batch_size":1000},{"forward_passes":1}])
def test_reject_custom_requests(body):
    with pytest.raises(ValidationError):
        CnnTestRequest.model_validate(body)
    worker=SimpleNamespace(worker_id="cnn",status=WorkerStatus.READY)
    with TestClient(create_diagnostic_app(worker)) as client:
        assert client.post("/cnn-test",json=body).status_code == 422


@pytest.mark.parametrize("change", [{"device":"cpu"},{"output_shape":[32,11]},
    {"input_shape":[32,1,32,32]},{"throughput_images_per_second":0},{"total_gpu_time_ms":float("nan")}])
def test_reject_invalid_result(change):
    with pytest.raises(ValidationError):
        CnnTestResponse.model_validate({**result().model_dump(),**change})


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def register(client,status="ready",metadata=None):
    client.post("/workers/register",json={"worker_id":"cnn","gpu":"Mock GPU","gpu_memory":16,
        "cuda_available":True,"status":status,"metadata":metadata})


def test_unknown_and_missing_url(client):
    assert client.post("/workers/missing/cnn-test",json={}).status_code == 404
    register(client)
    assert client.post("/workers/cnn/cnn-test",json={}).status_code == 409
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"


@pytest.mark.parametrize("status", ["busy","error"])
def test_not_ready(client,status):
    register(client,status=status)
    assert client.post("/workers/cnn/cnn-test",json={}).status_code == 409
    assert client.get("/workers").json()["workers"][0]["status"] == status


def test_expired(client):
    register(client)
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value=datetime.now(timezone.utc)+timedelta(seconds=31)
        assert client.post("/workers/cnn/cnn-test",json={}).status_code == 404


def test_forward_and_busy_state(client,monkeypatch):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    response=Mock(status_code=200)
    response.json.return_value=result().model_dump(mode="json")
    def post(url,**kwargs):
        assert url == "https://worker.example/cnn-test"
        assert kwargs == {"json":{},"timeout":(5,120),"allow_redirects":False}
        assert client.get("/workers").json()["workers"][0]["status"] == "busy"
        return response
    monkeypatch.setattr("controller.cnn_test.requests.post",post)
    assert client.post("/workers/cnn/cnn-test",json={}).json()["model"] == "SmallCNN"
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"
    response.close.assert_called_once()


@pytest.mark.parametrize("error,code",[(requests.Timeout(),504),(requests.ConnectionError(),502)])
def test_transport_failures_restore_state(client,monkeypatch,error,code):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    monkeypatch.setattr("controller.cnn_test.requests.post",Mock(side_effect=error))
    assert client.post("/workers/cnn/cnn-test",json={}).status_code == code
    assert client.get("/workers").json()["workers"][0]["status"] == "ready"


@pytest.mark.parametrize("body",[{},result("other").model_dump(mode="json")])
def test_invalid_remote_response(client,monkeypatch,body):
    register(client,metadata={"diagnostic_url":"https://worker.example"})
    response=Mock(status_code=200);response.json.return_value=body
    monkeypatch.setattr("controller.cnn_test.requests.post",Mock(return_value=response))
    assert client.post("/workers/cnn/cnn-test",json={}).status_code == 502


def fake_torch():
    torch=MagicMock()
    class Module:
        def to(self,**kwargs):
            self.target=kwargs
            return self
        def eval(self):
            self.evaluating=True
        def __call__(self,inputs):
            return self.forward(inputs)
    torch.nn.Module=Module
    torch.cuda.is_available.return_value=True
    torch.cuda.get_device_name.return_value="Mock GPU"
    torch.cuda.max_memory_allocated.return_value=8*1024**2
    torch.version.cuda="test";torch.__version__="test"
    torch.randn.return_value.shape=(32,3,32,32)
    output=torch.nn.Sequential.return_value.return_value
    output.shape=(32,10);output.device="cuda:0"
    return torch


def test_cnn_architecture_and_cuda_timing(monkeypatch):
    torch=fake_torch()
    built=[]
    builder=module.build_small_cnn
    def build(api):
        model=builder(api);built.append(model);return model
    monkeypatch.setattr(module,"build_small_cnn",build)
    monkeypatch.setattr(module.time,"perf_counter",Mock(side_effect=[1,1.1]))
    monkeypatch.setattr(module.importlib,"import_module",Mock(return_value=torch))
    response=module.run_cnn_test("cnn")
    assert built[0].__class__.__name__ == "SmallCNN" and built[0].evaluating
    assert built[0].target == {"device":"cuda:0","dtype":torch.float32}
    assert response.output_shape == (32,10)
    assert response.total_gpu_time_ms == pytest.approx(100)
    assert response.average_inference_ms == pytest.approx(1)
    assert response.throughput_images_per_second == pytest.approx(32000)
    assert response.peak_memory_mb == 8
    assert torch.nn.Sequential.return_value.call_count == 110
    assert [call.args for call in torch.nn.Conv2d.call_args_list] == [(3,16),(16,32)]
    for call in torch.nn.Conv2d.call_args_list:
        assert call.kwargs == {"kernel_size":3,"padding":1}
    assert [call.args for call in torch.nn.MaxPool2d.call_args_list] == [(2,),(2,)]
    torch.nn.AdaptiveAvgPool2d.assert_called_once_with((1,1))
    torch.nn.Flatten.assert_called_once_with()
    torch.nn.Linear.assert_called_once_with(32,10)
    torch.no_grad.assert_called_once_with()
    torch.no_grad.return_value.__enter__.assert_called_once()
    torch.no_grad.return_value.__exit__.assert_called_once()
    torch.optim.assert_not_called()
    torch.nn.Sequential.return_value.return_value.backward.assert_not_called()
    torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)
    assert torch.cuda.synchronize.call_count == 2
    torch.random.default_generator.manual_seed.assert_called_once_with(0)
    torch.random.fork_rng.assert_called_once_with(devices=[])
    torch.Generator.assert_called_once_with(device="cuda:0")
    torch.Generator.return_value.manual_seed.assert_called_once_with(0)
    assert torch.randn.call_args.args == (32,3,32,32)
    assert torch.randn.call_args.kwargs["generator"] is torch.Generator.return_value.manual_seed.return_value


def test_missing_torch_cuda_and_execution_error(monkeypatch):
    importer=Mock(side_effect=ImportError("missing"))
    monkeypatch.setattr(module.importlib,"import_module",importer)
    with pytest.raises(module.CnnTestError,match="PyTorch"):
        module.run_cnn_test("cnn")
    importer.side_effect=None
    torch=fake_torch();importer.return_value=torch
    torch.cuda.is_available.return_value=False
    with pytest.raises(module.CnnTestError,match="CUDA is unavailable"):
        module.run_cnn_test("cnn")
    torch.nn.Sequential.assert_not_called()
    torch.cuda.is_available.return_value=True
    torch.nn.Sequential.side_effect=RuntimeError("out of memory")
    with pytest.raises(module.CnnTestError,match="out of memory"):
        module.run_cnn_test("cnn")


def test_worker_lock_blocks_all_diagnostics(monkeypatch):
    worker=SimpleNamespace(worker_id="cnn",status=WorkerStatus.READY)
    entered,release=Event(),Event()
    def execute(worker_id):
        assert worker.status == WorkerStatus.BUSY
        entered.set();assert release.wait(5)
        return result()
    monkeypatch.setattr("worker.diagnostics.run_cnn_test",execute)
    with TestClient(create_diagnostic_app(worker)) as client,ThreadPoolExecutor(1) as pool:
        pending=pool.submit(client.post,"/cnn-test",json={})
        try:
            assert entered.wait(5)
            for route in ("/cnn-test","/nn-test","/gpu-test"):
                assert client.post(route,json={}).status_code == 409
        finally:
            release.set()
        assert pending.result().status_code == 200
        assert worker.status == WorkerStatus.READY


def test_worker_failure_restores_ready(monkeypatch):
    worker=SimpleNamespace(worker_id="cnn",status=WorkerStatus.READY)
    monkeypatch.setattr("worker.diagnostics.run_cnn_test",Mock(side_effect=module.CnnTestError("failed")))
    with TestClient(create_diagnostic_app(worker)) as client:
        assert client.post("/cnn-test",json={}).status_code == 503
        assert worker.status == WorkerStatus.READY
        worker.status=WorkerStatus.BUSY
        assert client.post("/cnn-test",json={}).status_code == 409
        assert worker.status == WorkerStatus.BUSY


def test_dashboard_cnn_button(client):
    response=client.get("/dashboard")
    assert response.status_code == 200
    for text in ("Run CNN Test","Run NN Test","CNN TEST RUNNING","throughput_images_per_second"):
        assert text in response.text


@pytest.mark.parametrize("route,error_path,error_type", [
    ("/gpu-test","worker.diagnostics.run_gpu_test","gpu"),
    ("/nn-test","worker.diagnostics.run_nn_test","nn"),
])
def test_existing_diagnostics_also_block_cnn(monkeypatch,route,error_path,error_type):
    from worker.gpu_test import GpuTestError
    from worker.nn_test import NnTestError
    worker=SimpleNamespace(worker_id="cnn",status=WorkerStatus.READY)
    entered,release=Event(),Event()
    def execute(*args):
        entered.set();assert release.wait(5)
        raise (GpuTestError if error_type == "gpu" else NnTestError)("fixture complete")
    monkeypatch.setattr(error_path,execute)
    with TestClient(create_diagnostic_app(worker)) as client,ThreadPoolExecutor(1) as pool:
        pending=pool.submit(client.post,route,json={})
        try:
            assert entered.wait(5)
            assert client.post("/cnn-test",json={}).status_code == 409
        finally:
            release.set()
        assert pending.result().status_code == 503
        assert worker.status == WorkerStatus.READY
