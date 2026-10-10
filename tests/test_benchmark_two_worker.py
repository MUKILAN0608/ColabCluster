"""Two-worker benchmark validation without remote GPU execution."""
import csv
import json
from unittest.mock import Mock
import pytest
import requests
from scripts import benchmark_two_worker as benchmark


def payload():
    return dict(status="passed",worker_count=2,total_samples=64,parallel_wall_time_ms=100,
        effective_throughput_images_per_second=640,sum_worker_gpu_time_ms=21,
        output_shapes=[[32,10],[32,10]],workers=[dict(worker_id=name,gpu="Tesla T4",
            status="passed",device="cuda:0",batch_size=32,output_shape=[32,10],
            total_gpu_time_ms=10+i) for i,name in enumerate(["a","b"])])


def reply(data=None,code=200):
    response=Mock(status_code=code,text="bad reply")
    response.json.return_value=payload() if data is None else data
    return response


def test_parser_and_optional_gpu_time():
    data=payload();values,errors=benchmark.parse_response(data)
    assert not errors and values["combined_gpu_time_ms"]==21
    assert json.loads(values["worker_ids"])==["a","b"]
    del data["sum_worker_gpu_time_ms"]
    assert benchmark.parse_response(data)[0]["combined_gpu_time_ms"]==21
    for w in data["workers"]:del w["total_gpu_time_ms"]
    values,errors=benchmark.parse_response(data)
    assert not errors and "combined_gpu_time_ms" not in values


@pytest.mark.parametrize("field,value",[("worker_count",1),("total_samples",32),
    ("workers",[]),("workers",[None,None]),("output_shapes",[[64,10]]),
    ("parallel_wall_time_ms",float("nan")),("status","failed"),
    ("effective_throughput_images_per_second",True)])
def test_invalid_response(field,value):
    data=payload();data[field]=value
    assert benchmark.parse_response(data)[1]


@pytest.mark.parametrize("change",[{"worker_id":"a"},{"batch_size":64},
    {"status":"failed"},{"device":"cpu"},{"output_shape":[64,10]},{"gpu":None}])
def test_invalid_worker(change):
    data=payload();data["workers"][1].update(change)
    assert benchmark.parse_response(data)[1]


def test_non_object():
    assert benchmark.parse_response([])[1]


def test_csv_failures_timing_and_stats(tmp_path,monkeypatch):
    bad=reply();bad.json.side_effect=ValueError()
    responses=[reply(),requests.Timeout("expired"),reply({"detail":"No READY workers"},409),bad]+[reply() for _ in range(6)]
    transport=Mock(side_effect=responses)
    monkeypatch.setattr(benchmark.shared.requests,"post",transport)
    monkeypatch.setattr(benchmark.shared.time,"perf_counter",Mock(side_effect=range(20)))
    path=tmp_path/"results"/"two.csv"
    rows,summary=benchmark.run_benchmark("http://controller",output=path)
    with path.open(newline="",encoding="utf-8") as stream:saved=list(csv.DictReader(stream))
    assert len(saved)==10 and [r["run"] for r in saved]==[str(n) for n in range(1,11)]
    assert summary==benchmark.summarize(saved)
    assert summary["successful"]==7 and summary["failed"]==3
    assert summary["mean_client_wall_time_ms"]==1000
    assert summary["mean_combined_gpu_time_ms"]==21
    assert summary["mean_effective_throughput_images_per_second"]==640
    assert saved[0]["server_wall_time_ms"]=="100.0"
    assert saved[2]["http_status"]=="409" and "No READY" in saved[2]["error"]
    assert "Timeout" in saved[1]["error"] and "invalid JSON" in saved[3]["error"]
    for call in transport.call_args_list:
        assert call.args==("http://controller/inference/two-worker",)
        assert call.kwargs==dict(json={},timeout=135,allow_redirects=False)
    for response in responses:
        if not isinstance(response,Exception):response.close.assert_called_once()


def test_statistics_exclude_failures_and_missing_gpu():
    rows=[dict(status="passed",client_wall_time_ms=n,combined_gpu_time_ms=gpu,
               effective_throughput_images_per_second=640) for n,gpu in [(100,20),(200,""),(300,40)]]
    rows.append(dict(status="failed",client_wall_time_ms=9000))
    summary=benchmark.summarize(rows)
    assert summary["mean_client_wall_time_ms"]==summary["median_client_wall_time_ms"]==200
    assert summary["sample_stdev_client_wall_time_ms"]==100
    assert summary["min_client_wall_time_ms"]==100 and summary["max_client_wall_time_ms"]==300
    assert summary["mean_combined_gpu_time_ms"]==30 and summary["gpu_time_available_runs"]==2
    assert benchmark.summarize([dict(status="failed")])["mean_combined_gpu_time_ms"] is None
