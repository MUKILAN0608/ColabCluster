"""Benchmark parsing, failure persistence and statistics without live GPUs."""
import csv
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_single_worker.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def payload():
    return dict(status="passed",worker_id="a",gpu="Tesla T4",total_samples=64,
                wall_time_ms=100,total_gpu_time_ms=10,
                effective_throughput_images_per_second=640,output_shape=[64,10])


def response(data=None,code=200):
    reply=Mock(status_code=code,text="upstream failure")
    reply.json.return_value=payload() if data is None else data
    return reply


def test_parse_aliases_and_numeric_strings():
    data=payload();data["worker_name"]=data.pop("worker_id")
    data["gpu_model"]=data.pop("gpu");data["wall_time_ms"]="100"
    values,errors=benchmark.parse_response(data)
    assert not errors
    assert values["worker_id"]=="a" and values["server_wall_time_ms"]==100
    assert values["output_shape"]=="[64, 10]"


@pytest.mark.parametrize("change",[{"wall_time_ms":None},{"wall_time_ms":True},
    {"wall_time_ms":"nan"},{"total_gpu_time_ms":-1},{"total_samples":32},
    {"output_shape":[32,10]},{"status":"failed"},{"worker_id":{}},
    {"effective_throughput_images_per_second":float("inf")}])
def test_invalid_fields_do_not_pass(change):
    data=payload();data.update(change)
    _,errors=benchmark.parse_response(data)
    assert errors


@pytest.mark.parametrize("data",[None,[],"passed",42])
def test_non_object(data):
    assert benchmark.parse_response(data)[1]


def test_timing_covers_request_not_decoding(monkeypatch):
    events=[]
    times=iter([1,1.25])
    def tick():events.append("clock");return next(times)
    reply=response()
    def decode():events.append("decode");return payload()
    reply.json.side_effect=decode
    def post(url,**kwargs):
        events.append("post")
        assert url=="http://controller/inference/single-worker"
        assert kwargs==dict(json={},timeout=135,allow_redirects=False)
        return reply
    monkeypatch.setattr(benchmark.time,"perf_counter",tick)
    monkeypatch.setattr(benchmark.requests,"post",post)
    row=benchmark.attempt("http://controller/",1,135)
    assert row["client_wall_time_ms"]==250
    assert row["server_wall_time_ms"]==100
    assert events==["clock","post","clock","decode"]
    reply.close.assert_called_once()


def test_failures_continue_and_csv_has_every_attempt(tmp_path,monkeypatch):
    invalid=response();invalid.json.side_effect=ValueError("bad json")
    replies=[response(),requests.Timeout("timed out"),response({"detail":"No READY workers available"},409),invalid,
             response(),response(),response(),response(),response(),response()]
    post=Mock(side_effect=replies)
    monkeypatch.setattr(benchmark.requests,"post",post)
    monkeypatch.setattr(benchmark.time,"perf_counter",Mock(side_effect=range(20)))
    destination=tmp_path/"results"/"single.csv"
    rows,summary=benchmark.run_benchmark("http://controller",output=destination)
    with destination.open(newline="",encoding="utf-8") as stream:
        saved=list(csv.DictReader(stream))
    assert len(saved)==10 and [r["run"] for r in saved]==[str(n) for n in range(1,11)]
    assert post.call_count==10
    assert "Timeout: timed out" in saved[1]["error"]
    assert saved[2]["http_status"]=="409" and "No READY" in saved[2]["error"]
    assert "invalid JSON" in saved[3]["error"]
    assert summary==benchmark.summarize(saved)
    assert summary["successful"]==7 and summary["failed"]==3
    assert summary["mean_client_wall_time_ms"]==1000
    assert summary["mean_gpu_inference_time_ms"]==10
    assert summary["mean_effective_throughput_images_per_second"]==640
    for item in replies:
        if not isinstance(item,Exception):item.close.assert_called_once()


def test_statistics_are_success_only():
    rows=[dict(status="passed",client_wall_time_ms=n,gpu_inference_time_ms=10,
               effective_throughput_images_per_second=640) for n in [100,200,300]]
    rows.append(dict(status="failed",client_wall_time_ms=9999))
    stats=benchmark.summarize(rows)
    assert stats["mean_client_wall_time_ms"]==stats["median_client_wall_time_ms"]==200
    assert stats["sample_stdev_client_wall_time_ms"]==100
    assert stats["min_client_wall_time_ms"]==100 and stats["max_client_wall_time_ms"]==300
    assert stats["failed"]==1


def test_all_failed_has_no_invented_measurements():
    stats=benchmark.summarize([dict(status="failed")]*10)
    assert stats["successful"]==0 and stats["failed"]==10
    assert stats["mean_client_wall_time_ms"] is None
    assert stats["mean_gpu_inference_time_ms"] is None


def test_connection_failure(monkeypatch):
    monkeypatch.setattr(benchmark.requests,"post",Mock(side_effect=requests.ConnectionError("refused")))
    row=benchmark.attempt("http://controller",1,1)
    assert row["status"]=="failed" and row["http_status"]==""
    assert row["error"]=="ConnectionError: refused"
    assert row["client_wall_time_ms"]>=0
