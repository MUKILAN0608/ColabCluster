"""Fixed parallel dispatch, protocol validation and GPU-free worker tests."""
from threading import Barrier, Event
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient
from controller.main import create_app
from common.schemas import WorkerStatus
from worker.diagnostics import create_diagnostic_app
from worker import inference
from test_cnn import fake_torch

IDS = ("COLAB-GPU-TEST", "COLAB-GPU-TEST-2")


def result(index):
    return dict(worker_id=IDS[index], partition=index, gpu="Tesla T4", cuda_available=True,
        cuda_version="13.0", torch_version="test", device="cuda:0", model="SmallCNN",
        input_shape=[32,3,32,32], batch_size=32, forward_passes=1,
        total_gpu_time_ms=10 + index, average_inference_ms=10 + index,
        throughput_images_per_second=32000/(10+index), peak_memory_mb=8,
        output_shape=[32,10], status="passed")


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        for index, identifier in enumerate(IDS):
            assert client.post("/workers/register",json=dict(worker_id=identifier,
                gpu="Tesla T4",gpu_memory=14.56,cuda_available=True,status="ready",
                metadata={"diagnostic_url":f"https://worker-{index}.example"})).status_code == 201
        yield client


def states(client):
    return {w["worker_id"]:w["status"] for w in client.get("/workers").json()["workers"]}


def test_concurrent_dispatch_and_aggregation(client, monkeypatch):
    overlap = Barrier(2, timeout=5)
    def post(url, **kwargs):
        index=kwargs["json"]["partition"]
        assert url == f"https://worker-{index}.example/inference"
        assert kwargs == dict(json={"partition":index},timeout=(5,120),allow_redirects=False)
        assert set(states(client).values()) == {"busy"}
        # A ready heartbeat cannot erase the reservation during dispatch.
        client.post("/workers/heartbeat",json={"worker_id":IDS[index],"status":"ready"})
        assert set(states(client).values()) == {"busy"}
        overlap.wait()  # Sequential dispatch deterministically fails this barrier.
        response=Mock(status_code=200)
        response.json.return_value=result(index)
        return response
    monkeypatch.setattr("controller.inference.requests.post",post)
    response=client.post("/inference/two-worker",json={})
    assert response.status_code == 200, response.text
    data=response.json()
    assert data["status"] == "passed" and data["total_samples"] == 64
    assert data["worker_count"] == 2
    assert [w["worker_id"] for w in data["workers"]] == list(IDS)
    assert [w["partition"] for w in data["workers"]] == [0,1]
    assert data["output_shapes"] == [[32,10],[32,10]]
    assert data["sum_worker_gpu_time_ms"] == 21
    assert data["effective_throughput_images_per_second"] == pytest.approx(64000/data["parallel_wall_time_ms"])
    assert set(states(client).values()) == {"ready"}


@pytest.mark.parametrize("index",[0,1])
@pytest.mark.parametrize("condition,code",[("missing",404),("busy",409)])
def test_unavailable_pair_does_not_dispatch(client,monkeypatch,index,condition,code):
    if condition == "missing":
        client.post("/workers/unregister",json={"worker_id":IDS[index]})
    else:
        client.post("/workers/heartbeat",json={"worker_id":IDS[index],"status":"busy"})
    before=states(client)
    post=Mock();monkeypatch.setattr("controller.inference.requests.post",post)
    assert client.post("/inference/two-worker",json={}).status_code == code
    assert states(client) == before
    post.assert_not_called()


@pytest.mark.parametrize("index",[0,1])
@pytest.mark.parametrize("failure,code",[("timeout",504),("connection",502),("http",503),
    ("identity",502),("shape",502),("gpu",502),("cuda",502),("partition",502)])
def test_failure_never_reports_success_and_restores_pair(client,monkeypatch,index,failure,code):
    completed=[]
    def post(url, **kwargs):
        part=kwargs["json"]["partition"]
        completed.append(part)
        data=result(part)
        if part == index:
            if failure == "timeout": raise requests.Timeout()
            if failure == "connection": raise requests.ConnectionError()
            if failure == "identity": data["worker_id"]="wrong"
            if failure == "shape": data["output_shape"]=[64,10]
            if failure == "gpu": data["gpu"]="CPU"
            if failure == "cuda": data["device"]="cpu"
            if failure == "partition": data["partition"]=1-part
        response=Mock(status_code=503 if part==index and failure=="http" else 200)
        response.json.return_value=data
        return response
    monkeypatch.setattr("controller.inference.requests.post",post)
    response=client.post("/inference/two-worker",json={})
    assert response.status_code == code
    assert sorted(completed) == [0,1]
    assert set(states(client).values()) == {"ready"}


@pytest.mark.parametrize("body",[{"code":"print(1)"},{"worker_ids":[IDS[0],IDS[0]]},{"total_samples":128}])
def test_reject_custom_workload(client,body):
    assert client.post("/inference/two-worker",json=body).status_code == 422


@pytest.mark.parametrize("partition",[0,1])
def test_worker_fixed_partition_cuda_execution(monkeypatch,partition):
    torch=fake_torch();torch.cuda.get_device_name.return_value="Tesla T4"
    monkeypatch.setattr(inference.importlib,"import_module",Mock(return_value=torch))
    monkeypatch.setattr(inference.time,"perf_counter",Mock(side_effect=[1,1.1]))
    data=inference.run_inference(IDS[partition],partition)
    assert data.forward_passes == 1 and data.partition == partition
    assert data.throughput_images_per_second == pytest.approx(320)
    assert data.average_inference_ms == pytest.approx(100)
    assert torch.nn.Sequential.return_value.call_count == 11
    torch.Generator.return_value.manual_seed.assert_called_once_with(1000+partition)
    torch.random.default_generator.manual_seed.assert_called_once_with(0)
    torch.no_grad.return_value.__enter__.assert_called_once()
    assert torch.cuda.synchronize.call_count == 2


def test_worker_inference_shares_diagnostic_lock(monkeypatch):
    worker=SimpleNamespace(worker_id=IDS[0],status=WorkerStatus.READY)
    entered, release=Event(),Event()
    def execute(identifier,partition):
        assert worker.status == WorkerStatus.BUSY
        entered.set();assert release.wait(5)
        return result(partition)
    monkeypatch.setattr("worker.diagnostics.run_inference",execute)
    with TestClient(create_diagnostic_app(worker)) as client, ThreadPoolExecutor(1) as pool:
        pending=pool.submit(client.post,"/inference",json={"partition":0})
        try:
            assert entered.wait(5)
            for route,body in [("/cnn-test",{}),("/nn-test",{}),("/gpu-test",{}),("/inference",{"partition":1})]:
                assert client.post(route,json=body).status_code == 409
        finally:
            release.set()
        assert pending.result().status_code == 200
        assert worker.status == WorkerStatus.READY
        assert client.post("/inference",json={"partition":2}).status_code == 422
        monkeypatch.setattr("worker.diagnostics.run_inference",Mock(side_effect=inference.CnnTestError("CUDA failed")))
        assert client.post("/inference",json={"partition":0}).status_code == 503
        assert worker.status == WorkerStatus.READY


@pytest.mark.parametrize("change",[{"gpu":"None"},{"cuda_available":False},
    {"metadata":{}},{"metadata":{"diagnostic_url":"https://worker-0.example"}}])
def test_preflight_rejects_invalid_worker_without_dispatch(client,monkeypatch,change):
    registry=client.app.state.registry
    record=registry.get_worker(IDS[1]).model_dump()
    record.update(change)
    client.post("/workers/unregister",json={"worker_id":IDS[1]})
    record.pop("registered_at");record.pop("last_seen")
    record.pop("gpu_utilization");record.pop("gpu_memory_used")
    assert client.post("/workers/register",json=record).status_code == 201
    post=Mock();monkeypatch.setattr("controller.inference.requests.post",post)
    assert client.post("/inference/two-worker",json={}).status_code == 409
    post.assert_not_called()
    assert set(states(client).values()) == {"ready"}


def test_failure_waits_for_peer_before_releasing_reservations(client,monkeypatch):
    entered, release=Event(),Event()
    def post(url,**kwargs):
        if kwargs["json"]["partition"] == 0:
            assert entered.wait(5)
            raise requests.ConnectionError()
        entered.set();assert release.wait(5)
        response=Mock(status_code=200);response.json.return_value=result(1)
        return response
    monkeypatch.setattr("controller.inference.requests.post",post)
    with ThreadPoolExecutor(1) as pool:
        pending=pool.submit(client.post,"/inference/two-worker",json={})
        try:
            assert entered.wait(5)
            assert not pending.done()
            assert set(states(client).values()) == {"busy"}
        finally:
            release.set()
        assert pending.result().status_code == 502
    assert set(states(client).values()) == {"ready"}


def test_removed_worker_is_not_restored(client,monkeypatch):
    barrier=Barrier(2,timeout=5)
    def post(url,**kwargs):
        part=kwargs["json"]["partition"]
        if part == 0:
            client.post("/workers/unregister",json={"worker_id":IDS[0]})
        barrier.wait()
        response=Mock(status_code=200);response.json.return_value=result(part)
        return response
    monkeypatch.setattr("controller.inference.requests.post",post)
    assert client.post("/inference/two-worker",json={}).status_code == 200
    assert states(client) == {IDS[1]:"ready"}


def test_worker_requires_cuda(monkeypatch):
    torch=fake_torch();torch.cuda.is_available.return_value=False
    monkeypatch.setattr(inference.importlib,"import_module",Mock(return_value=torch))
    with pytest.raises(inference.CnnTestError,match="CUDA is unavailable"):
        inference.run_inference(IDS[0],0)
    torch.nn.Sequential.assert_not_called()
