"""Controlled single-worker baseline without real network or GPU execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from common.schemas import WorkerStatus
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app
from worker import inference
from test_cnn import fake_torch


def result(identifier="a"):
    return dict(worker_id=identifier,gpu="Tesla T4",cuda_available=True,cuda_version="13.0",
        torch_version="test",device="cuda:0",model="SmallCNN",input_shape=[64,3,32,32],
        batch_size=64,forward_passes=1,total_gpu_time_ms=10,average_inference_ms=10,
        throughput_images_per_second=6400,peak_memory_mb=8,output_shape=[64,10],status="passed")


def register(client,identifier,status="ready",**changes):
    assert client.post("/workers/register",json=dict(worker_id=identifier,gpu="Tesla T4",
        gpu_memory=14.56,cuda_available=True,status=status,
        metadata={"diagnostic_url":f"https://{identifier}.example"},**changes)).status_code == 201


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def states(client):
    return {w["worker_id"]:w["status"] for w in client.get("/workers").json()["workers"]}


def test_openapi(client):
    schema=client.get("/openapi.json").json()
    route=schema["paths"]["/inference/single-worker"]["post"]
    assert route["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/SingleWorkerRequest")
    assert route["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/SingleWorkerResponse")
    assert "/inference/two-worker" in schema["paths"]


@pytest.mark.parametrize("count",[1,2])
def test_one_request_ordering_timing_and_restoration(client,monkeypatch,count):
    if count==2: register(client,"b")  # Reverse registration order deliberately.
    register(client,"a")
    events=[]
    ticks=iter([5,7])
    def clock():
        events.append("clock")
        return next(ticks)
    response=Mock(status_code=200)
    def decode():
        events.append("decode")
        return result()
    response.json.side_effect=decode
    def post(url,**kwargs):
        events.append("post")
        assert url=="https://a.example/inference/single-worker"
        assert kwargs==dict(json={},timeout=(5,120),allow_redirects=False)
        assert states(client)==({"a":"busy","b":"ready"} if count==2 else {"a":"busy"})
        client.post("/workers/heartbeat",json={"worker_id":"a","status":"ready"})
        assert states(client)["a"]=="busy"
        return response
    transport=Mock(side_effect=post)
    monkeypatch.setattr("controller.inference.requests.post",transport)
    monkeypatch.setattr("controller.inference.perf_counter",clock)
    reply=client.post("/inference/single-worker",json={})
    assert reply.status_code==200,reply.text
    data=reply.json()
    assert data["worker_id"]=="a" and data["status"]=="passed"
    assert data["total_samples"]==data["batch_size"]==64
    assert data["input_shape"]==[64,3,32,32] and data["output_shape"]==[64,10]
    assert data.get("worker_count",1)==1
    assert data["total_gpu_time_ms"]==10 and data["wall_time_ms"]==2000
    assert data["effective_throughput_images_per_second"]==32
    assert events==["clock","post","clock","decode"]
    transport.assert_called_once();response.close.assert_called_once()
    assert set(states(client).values())=={"ready"}


@pytest.mark.parametrize("status",[None,"busy","error"])
def test_no_ready(client,monkeypatch,status):
    if status: register(client,"a",status)
    transport=Mock();monkeypatch.setattr("controller.inference.requests.post",transport)
    reply=client.post("/inference/single-worker",json={})
    assert reply.status_code==409 and reply.json()["detail"]=="No READY workers available"
    transport.assert_not_called()


def test_skip_busy_worker(client,monkeypatch):
    register(client,"a","busy");register(client,"b")
    response=Mock(status_code=200);response.json.return_value=result("b")
    transport=Mock(return_value=response);monkeypatch.setattr("controller.inference.requests.post",transport)
    assert client.post("/inference/single-worker",json={}).json()["worker_id"]=="b"
    assert transport.call_args.args[0]=="https://b.example/inference/single-worker"
    assert states(client)=={"a":"busy","b":"ready"}


@pytest.mark.parametrize("failure,code",[("timeout",504),("connection",502),("http",503),
    ("identity",502),("shape",502),("cuda",502),("json",502),("time",502)])
def test_failure_restores_selected_worker(client,monkeypatch,failure,code):
    register(client,"a");register(client,"b")
    response=Mock(status_code=503 if failure=="http" else 200)
    data=result()
    if failure=="identity": data["worker_id"]="b"
    if failure=="shape": data["output_shape"]=[32,10]
    if failure=="cuda": data["device"]="cpu"
    if failure=="time": data["total_gpu_time_ms"]=float("nan")
    response.json.return_value=data
    if failure=="json": response.json.side_effect=ValueError("invalid JSON")
    transport=Mock(return_value=response)
    if failure=="timeout":transport.side_effect=requests.Timeout()
    if failure=="connection":transport.side_effect=requests.ConnectionError()
    monkeypatch.setattr("controller.inference.requests.post",transport)
    assert client.post("/inference/single-worker",json={}).status_code==code
    assert states(client)=={"a":"ready","b":"ready"}
    if failure not in ("timeout","connection"): response.close.assert_called_once()


@pytest.mark.parametrize("body",[{"batch_size":32},{"worker_id":"a"},{"code":"print(1)"}])
def test_fixed_request(client,body):
    assert client.post("/inference/single-worker",json=body).status_code==422


def test_expired_worker_not_selected(client,monkeypatch):
    register(client,"a")
    with patch("controller.registry.datetime") as clock:
        clock.now.return_value=datetime.now(timezone.utc)+timedelta(seconds=100)
        assert client.post("/inference/single-worker",json={}).status_code==409


def test_removal_is_not_undone(client,monkeypatch):
    register(client,"a")
    def post(*args,**kwargs):
        client.post("/workers/unregister",json={"worker_id":"a"})
        response=Mock(status_code=200);response.json.return_value=result()
        return response
    monkeypatch.setattr("controller.inference.requests.post",post)
    assert client.post("/inference/single-worker",json={}).status_code==200
    assert states(client)=={}


def test_worker_reuses_model_timing_and_both_partition_seeds(monkeypatch):
    torch=fake_torch();torch.cuda.get_device_name.return_value="Tesla T4"
    torch.cat.return_value.shape=(64,3,32,32)
    torch.nn.Sequential.return_value.return_value.shape=(64,10)
    # Emulate Generator.manual_seed returning the same generator.
    generator=torch.Generator.return_value
    generator.manual_seed.return_value=generator
    monkeypatch.setattr(inference.importlib,"import_module",Mock(return_value=torch))
    monkeypatch.setattr(inference.time,"perf_counter",Mock(side_effect=[1,1.1]))
    data=inference.run_inference("a",0,single=True)
    assert data.batch_size==64 and data.output_shape==(64,10)
    assert data.throughput_images_per_second==pytest.approx(640)
    assert data.total_gpu_time_ms==pytest.approx(100)
    assert [c.args for c in generator.manual_seed.call_args_list]==[(1000,),(1001,)]
    assert torch.randn.call_count==2
    for call in torch.randn.call_args_list:
        assert call.args==(32,3,32,32)
        assert call.kwargs==dict(device="cuda:0",dtype=torch.float32,generator=generator)
    torch.cat.assert_called_once_with((torch.randn.return_value,torch.randn.return_value),dim=0)
    assert torch.nn.Sequential.return_value.call_count==11
    torch.no_grad.return_value.__enter__.assert_called_once()
    torch.random.default_generator.manual_seed.assert_called_once_with(0)
    torch.random.fork_rng.assert_called_once_with(devices=[])
    assert torch.cuda.synchronize.call_count==2
    torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)


def test_worker_lock_and_failure_cleanup(monkeypatch):
    worker=SimpleNamespace(worker_id="a",status=WorkerStatus.READY)
    entered,release=Event(),Event()
    def execute(identifier,partition,*,single=False):
        assert identifier=="a" and partition==0 and single
        assert worker.status==WorkerStatus.BUSY
        entered.set();assert release.wait(5)
        return result()
    monkeypatch.setattr("worker.diagnostics.run_inference",execute)
    with TestClient(create_diagnostic_app(worker)) as client,ThreadPoolExecutor(1) as pool:
        pending=pool.submit(client.post,"/inference/single-worker",json={})
        try:
            assert entered.wait(5)
            for route,body in [("/inference",{"partition":0}),("/inference/single-worker",{}),
                               ("/cnn-test",{}),("/nn-test",{}),("/gpu-test",{})]:
                assert client.post(route,json=body).status_code==409
        finally:release.set()
        assert pending.result().status_code==200
        assert worker.status==WorkerStatus.READY
        monkeypatch.setattr("worker.diagnostics.run_inference",Mock(side_effect=inference.CnnTestError("failed")))
        assert client.post("/inference/single-worker",json={}).status_code==503
        assert worker.status==WorkerStatus.READY
