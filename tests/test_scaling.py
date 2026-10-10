"""Scaling protocol, orchestration, inputs and benchmark safeguards without GPUs."""
import csv
from threading import Barrier
from unittest.mock import Mock
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError
from common.scaling import ScalingRequest, ScalingBatchRequest, ScalingBatchResponse, ScalingResponse, SIZES, WORKER_IDS, partition_samples
from controller.main import create_app
from worker import inference
from scripts import benchmark_scaling as bench
from test_cnn import fake_torch


def worker_result(size,count,index):
    n=partition_samples(size,count)[index]
    return dict(worker_id=WORKER_IDS[index],partition=index,gpu="Tesla T4",cuda_available=True,
        cuda_version="13.0",torch_version="test",device="cuda:0",model="SmallCNN",input_shape=[n,3,32,32],
        batch_size=n,forward_passes=1,total_gpu_time_ms=10,average_inference_ms=10,
        throughput_images_per_second=n*100,peak_memory_mb=8,output_shape=[n,10],status="passed")


def result(size,count):
    return dict(total_samples=size,worker_count=count,status="passed",workers=[worker_result(size,count,i) for i in range(count)],
        wall_time_ms=100,sum_worker_gpu_time_ms=count*10,effective_throughput_images_per_second=size*10)


@pytest.mark.parametrize("size",SIZES)
@pytest.mark.parametrize("count",[1,2])
def test_workload_and_output_validation(size,count):
    payload=ScalingRequest(total_samples=size,worker_count=count)
    counts=[ScalingBatchRequest(**payload.model_dump(),partition=i).assigned_samples for i in range(count)]
    assert sum(counts)==size and max(counts)-min(counts)<=1
    assert sum(w.output_shape[0] for w in ScalingResponse.model_validate(result(size,count)).workers)==size
    data=result(size,count);data["workers"][0]["output_shape"][0]+=1
    with pytest.raises(ValidationError):ScalingResponse.model_validate(data)


@pytest.mark.parametrize("size",[0,63,65,128,4097,"256"])
def test_invalid_workload(size):
    with pytest.raises(ValidationError):ScalingRequest(total_samples=size,worker_count=2)


def test_odd_partition_and_invalid_assignment():
    assert partition_samples(65,2)==[33,32]
    with pytest.raises(ValidationError):ScalingBatchRequest(total_samples=64,worker_count=1,partition=1)


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        for i,name in enumerate(WORKER_IDS):
            client.post("/workers/register",json=dict(worker_id=name,gpu="Tesla T4",gpu_memory=14.56,
                cuda_available=True,status="ready",metadata={"diagnostic_url":f"https://w{i}.example"}))
        yield client


@pytest.mark.parametrize("count",[1,2])
def test_dispatch_concurrency_and_assignment(client,monkeypatch,count):
    barrier=Barrier(count,timeout=5)
    calls=[]
    def post(url,**kwargs):
        data=kwargs["json"];i=data["partition"];calls.append(i)
        assert url==f"https://w{i}.example/inference/scaling"
        assert data==dict(total_samples=1024,worker_count=count,partition=i)
        barrier.wait()
        response=Mock(status_code=200);response.json.return_value=worker_result(1024,count,i)
        return response
    monkeypatch.setattr("controller.scaling.requests.post",post)
    response=client.post("/inference/scaling",json=dict(total_samples=1024,worker_count=count))
    assert response.status_code==200,response.text
    assert sorted(calls)==list(range(count))
    assert sum(w["batch_size"] for w in response.json()["workers"])==1024
    assert all(w["status"]=="ready" for w in client.get("/workers").json()["workers"])


@pytest.mark.parametrize("failure,code",[("timeout",504),("http",503),("wrong_count",502)])
def test_failure_restores_workers(client,monkeypatch,failure,code):
    def post(url,**kwargs):
        if failure=="timeout":raise requests.Timeout()
        response=Mock(status_code=503 if failure=="http" else 200)
        data=worker_result(64,2,kwargs["json"]["partition"])
        data["batch_size"]=64
        response.json.return_value=data
        return response
    monkeypatch.setattr("controller.scaling.requests.post",post)
    assert client.post("/inference/scaling",json=dict(total_samples=64,worker_count=2)).status_code==code
    assert all(w["status"]=="ready" for w in client.get("/workers").json()["workers"])


@pytest.mark.parametrize("size",SIZES)
@pytest.mark.parametrize("count",[1,2])
def test_shared_cuda_inputs_and_timing(monkeypatch,size,count):
    batch=ScalingBatchRequest(total_samples=size,worker_count=count,partition=count-1)
    torch=fake_torch();torch.cuda.get_device_name.return_value="Tesla T4"
    torch.cat.return_value.shape=(batch.assigned_samples,3,32,32)
    torch.nn.Sequential.return_value.return_value.shape=(batch.assigned_samples,10)
    monkeypatch.setattr(inference.importlib,"import_module",Mock(return_value=torch))
    monkeypatch.setattr(inference.time,"perf_counter",Mock(side_effect=[1,1.1]))
    data=inference.run_inference("test",batch.partition,scaling=batch)
    assert data.batch_size==batch.assigned_samples
    assert data.throughput_images_per_second==pytest.approx(batch.assigned_samples/0.1)
    offset=0 if count==1 else size//2
    assert [c.args[0] for c in torch.Generator.return_value.manual_seed.call_args_list]==list(range(1000+offset//32,1000+(offset+batch.assigned_samples)//32))
    assert torch.nn.Sequential.return_value.call_count==11
    torch.no_grad.return_value.__enter__.assert_called_once()
    assert torch.cuda.synchronize.call_count==2


def test_preflight_blocks_old_controller(monkeypatch):
    monkeypatch.setattr(bench,"get_json",Mock(return_value={"paths":{}}))
    with pytest.raises(ValueError,match="Controller lacks"):bench.preflight("http://controller")


def test_benchmark_failure_persistence_report_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,"preflight",Mock(return_value={}))
    calls=[]
    def post(url,**kwargs):
        data=kwargs["json"];calls.append(data)
        if len(calls)==1:raise requests.ConnectionError("temporary failure")
        response=Mock(status_code=200);response.json.return_value=result(data["total_samples"],data["worker_count"])
        return response
    monkeypatch.setattr(bench.shared.requests,"post",post)
    ticks=iter(range(1000));monkeypatch.setattr(bench.shared.time,"perf_counter",lambda:next(ticks))
    complete,rows,directory=bench.run_benchmark(output_root=tmp_path)
    assert complete and len(rows)==41
    assert sum(r["status"]=="failed" for r in rows)==1
    assert all(s["successful"]==5 for s in bench.grouped_statistics(rows).values())
    with (directory/"measurements.csv").open(newline="") as stream:saved=list(csv.DictReader(stream))
    assert bench.grouped_statistics(saved)==bench.grouped_statistics(rows)
    text=(tmp_path/"scaling_report.md").read_text()
    assert "1.000" in text and "statistical significance" in text
    before=(directory/"measurements.csv").read_bytes()
    monkeypatch.setattr(bench,"preflight",Mock(side_effect=ValueError("worker offline")))
    complete,rows,blocked=bench.run_benchmark(output_root=tmp_path)
    assert not complete and rows==[] and blocked!=directory
    assert "Preflight blocked" in (tmp_path/"scaling_report.md").read_text()
    assert (directory/"measurements.csv").read_bytes()==before


def test_benchmark_caps_failures_and_no_speedup_claim(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,"preflight",Mock(return_value={}))
    monkeypatch.setattr(bench.shared.requests,"post",Mock(side_effect=requests.Timeout("timeout")))
    complete,rows,_=bench.run_benchmark(output_root=tmp_path)
    assert not complete and len(rows)==80
    assert all(r["status"]=="failed" for r in rows)
    assert "Incomplete; target not met" in (tmp_path/"scaling_report.md").read_text()


@pytest.mark.parametrize("failure",["busy","old_worker","same_url","wrong_gpu","missing"])
def test_worker_preflight_safeguards(monkeypatch,failure):
    from datetime import datetime,timezone
    stamp=datetime.now(timezone.utc).isoformat()
    records=[dict(worker_id=name,gpu="Tesla T4",gpu_memory=14.56,cuda_available=True,status="ready",
        registered_at=stamp,last_seen=stamp,metadata={"diagnostic_url":f"https://w{i}.example"}) for i,name in enumerate(WORKER_IDS)]
    if failure=="busy":records[1]["status"]="busy"
    if failure=="wrong_gpu":records[1]["gpu"]="None"
    if failure=="same_url":records[1]["metadata"]=records[0]["metadata"]
    if failure=="missing":records.pop()
    def get(url):
        if url.endswith("/workers"):return {"workers":records}
        if failure=="old_worker" and "w1.example" in url:return {"paths":{}}
        return {"paths":{"/inference/scaling":{}}}
    monkeypatch.setattr(bench,"get_json",get)
    from fastapi import HTTPException
    with pytest.raises((ValueError,HTTPException)):bench.preflight("http://controller")


def test_worker_scaling_route_and_cleanup(monkeypatch):
    from types import SimpleNamespace
    from common.schemas import WorkerStatus
    from worker.diagnostics import create_diagnostic_app
    worker=SimpleNamespace(worker_id=WORKER_IDS[0],status=WorkerStatus.READY)
    def run(identifier,partition,*,scaling):
        assert worker.status==WorkerStatus.BUSY
        assert scaling.assigned_samples==256
        return worker_result(256,1,0)
    monkeypatch.setattr("worker.diagnostics.run_inference",run)
    with TestClient(create_diagnostic_app(worker)) as client:
        assert client.post("/inference/scaling",json=dict(total_samples=256,worker_count=1,partition=0)).status_code==200
        assert worker.status==WorkerStatus.READY
        monkeypatch.setattr("worker.diagnostics.run_inference",Mock(side_effect=inference.CnnTestError("CUDA OOM")))
        assert client.post("/inference/scaling",json=dict(total_samples=4096,worker_count=1,partition=0)).status_code==503
        assert worker.status==WorkerStatus.READY
