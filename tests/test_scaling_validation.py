"""Repeatability design, uncertainty and plots from temporary/mock observations."""
import csv
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import pytest
import requests
from pydantic import ValidationError
from common.scaling import ScalingRequest, ScalingBatchRequest, VALIDATION_SIZES, WORKER_IDS, partition_samples
from scripts import validate_scaling as run
from scripts import analyze_scaling_validation as analysis
from test_scaling import result
from worker import inference
from test_cnn import fake_torch


def metadata():
    return dict(experiment_id="test",method=run.METHOD,seed=42,
        targets={str(n):20 if n in (1024,4096) else 5 for n in VALIDATION_SIZES},
        workers={},software={"source_sha256":{"test":"hash"}},stop_reason="")


def rows_for(size=4096,n=20):
    rows=[]
    for i in range(1,n+1):
        for count in (1,2):
            row=dict.fromkeys(run.FIELDS,"")
            row.update(run=len(rows)+1,total_samples=size,worker_count=count,status="passed",api_status="passed",http_status="200",
                worker_ids=json.dumps(list(WORKER_IDS[:count])),assigned_samples=json.dumps(partition_samples(size,count)),
                gpu_models=json.dumps(["Tesla T4"]*count),client_wall_time_ms=100 if count==1 else 80,
                server_wall_time_ms=90 if count==1 else 70,gpu_inference_time_ms=10,
                effective_throughput_images_per_second=size/(.09 if count==1 else .07),
                output_shapes=json.dumps([[v,10] for v in partition_samples(size,count)]),experiment_id="test",
                timestamp_utc="2026-10-10T00:00:00Z",round=i,order=count if i%2 else 3-count,phase="confirmation",seed=42,
                worker_versions=json.dumps({name:{"cuda_version":"13.0","torch_version":"test"} for name in WORKER_IDS[:count]}))
            rows.append(row)
    return rows


def save(directory,rows,meta=None):
    directory.mkdir(exist_ok=True)
    (directory/"metadata.json").write_text(json.dumps(meta or metadata()),encoding="utf-8")
    with (directory/"measurements.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=run.FIELDS);writer.writeheader();writer.writerows(rows)


@pytest.mark.parametrize("size",VALIDATION_SIZES)
def test_extended_sizes_and_identical_seed_ranges(monkeypatch,size):
    ScalingRequest(total_samples=size,worker_count=2)
    ranges=[]
    for count in (1,2):
        combined=[]
        for index in range(count):
            batch=ScalingBatchRequest(total_samples=size,worker_count=count,partition=index)
            torch=fake_torch();torch.cuda.get_device_name.return_value="Tesla T4"
            torch.cat.return_value.shape=(batch.assigned_samples,3,32,32)
            torch.nn.Sequential.return_value.return_value.shape=(batch.assigned_samples,10)
            monkeypatch.setattr(inference.importlib,"import_module",Mock(return_value=torch))
            monkeypatch.setattr(inference.time,"perf_counter",Mock(side_effect=[1,1.1]))
            response=inference.run_inference(WORKER_IDS[index],index,scaling=batch)
            assert response.output_shape[0]==batch.assigned_samples
            combined.extend(c.args[0] for c in torch.Generator.return_value.manual_seed.call_args_list)
        ranges.append(combined)
    assert ranges[0]==ranges[1]==list(range(1000,1000+size//32))
    assert partition_samples(65,2)==[33,32]


def test_invalid_size_still_rejected():
    with pytest.raises(ValidationError):ScalingRequest(total_samples=1000,worker_count=2)


def test_balanced_order():
    for first in (1,2):
        orders=[run.order_for_round(n,first) for n in range(1,21)]
        assert sum(o[0]==1 for o in orders)==10
        assert all(set(o)=={1,2} for o in orders)


def test_statistics_ci_and_reproducibility():
    rows=rows_for()
    a=analysis.group_summary(rows,4096,1,20,42,500)
    b=analysis.group_summary(rows,4096,2,20,42,500)
    assert a["mean_client_ms"]==a["median_client_ms"]==100
    assert a["sample_sd_ms"]==a["iqr_ms"]==0
    assert a["mean_ci_low"]==a["mean_ci_high"]==100
    c=analysis.comparison(rows,a,b,42,500)
    assert c["speedup"]==1.25 and c["difference_ms"]==-20
    assert c["speedup_ci"]==[1.25,1.25] and c["difference_ci"]==[-20,-20]
    rows[0]["client_wall_time_ms"]=1000
    assert analysis.mean_ci(rows[::2],42,500)==analysis.mean_ci(rows[::2],42,500)
    assert analysis.group_summary(rows,4096,1,20,42,500)["max_client_ms"]==1000


def test_incomplete_no_comparison():
    rows=rows_for(n=5)
    a=analysis.group_summary(rows,4096,1,20,42,100)
    b=analysis.group_summary(rows,4096,2,20,42,100)
    c=analysis.comparison(rows,a,b,42,100)
    assert not c["complete"] and c["speedup"] is None and c["speedup_ci"] is None


def test_metadata_incompatibility(tmp_path):
    m=metadata();other=deepcopy(m);other["seed"]=99
    assert not analysis.compatible_metadata(m,other)
    assert analysis.compatible_metadata(m,deepcopy(m))
    other=deepcopy(m);other["software"]["source_sha256"]["test"]="changed"
    assert not analysis.compatible_metadata(m,other)
    m["method"]={"legacy":True};save(tmp_path,[],m)
    with pytest.raises(ValueError,match="Incompatible"):analysis.load(tmp_path)


def test_report_plots_and_source_preservation(tmp_path):
    save(tmp_path,rows_for())
    before={name:(tmp_path/name).read_bytes() for name in ("measurements.csv","metadata.json")}
    data=analysis.analyze(tmp_path,resamples=100)
    assert data["advantage_4096_supported"] and data["smallest_observed_advantage"]==4096
    output=tmp_path/"analysis"
    report=(output/"report.md").read_text(encoding="utf-8")
    assert "**yes**" in report and "1.250" in report and "Incomplete: 1024" in report
    assert "not an exact crossover" in report and "server" in report and "ms" in report
    from PIL import Image
    for name in ("wall_time","speedup","throughput","trials"):
        with Image.open(output/(name+".png")) as image:
            assert image.width>=1000 and image.height>=600
    for name,content in before.items():assert (tmp_path/name).read_bytes()==content
    assert analysis.analyze(tmp_path,resamples=100)==data


def test_empty_report_has_no_fabricated_ci(tmp_path):
    save(tmp_path,[])
    data=analysis.analyze(tmp_path,resamples=100)
    assert not data["advantage_4096_supported"] and data["smallest_observed_advantage"] is None
    assert all(c["speedup"] is None for c in data["comparisons"])
    assert "not established" in (tmp_path/"analysis/report.md").read_text()


def mock_preflight():
    return {name:dict(worker_id=name,gpu="Tesla T4",gpu_memory=14.56,cuda_available=True,
        registered_at="same",metadata={"torch_version":"test"}) for name in WORKER_IDS}


def test_runner_records_failures_and_meets_targets(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"preflight",Mock(return_value=mock_preflight()))
    monkeypatch.setattr(analysis,"analyze",Mock())
    calls=[]
    def post(url,**kwargs):
        body=kwargs["json"];calls.append(body)
        if len(calls)==1:raise requests.Timeout("bounded failure")
        response=Mock(status_code=200);response.json.return_value=result(body["total_samples"],body["worker_count"])
        return response
    monkeypatch.setattr(run.shared.requests,"post",post)
    meta,rows,directory=run.run_experiment(root=tmp_path)
    assert meta["complete"] and len(rows)==141
    assert rows[0]["status"]=="failed" and "Timeout" in rows[0]["error"]
    assert all(r["round"]<=2*meta["targets"][str(r["total_samples"])] for r in rows)
    assert set(r["total_samples"] for r in rows[:81])=={1024,4096}
    assert "diagnostic_url" not in (directory/"metadata.json").read_text()


def test_worker_disappears_stops_dispatch(tmp_path,monkeypatch):
    check=Mock(side_effect=[mock_preflight(),mock_preflight(),ValueError("worker unavailable")])
    monkeypatch.setattr(run,"preflight",check);monkeypatch.setattr(analysis,"analyze",Mock())
    response=Mock(status_code=200);response.json.return_value=result(1024,1)
    monkeypatch.setattr(run.shared.requests,"post",Mock(return_value=response))
    meta,rows,_=run.run_experiment(root=tmp_path,seed=1)
    assert not meta["complete"] and len(rows)==1
    assert "unavailable" in meta["stop_reason"]


def test_attempt_limit(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"preflight",Mock(return_value=mock_preflight()))
    monkeypatch.setattr(analysis,"analyze",Mock())
    monkeypatch.setattr(run.shared.requests,"post",Mock(side_effect=requests.Timeout("failed")))
    meta,rows,_=run.run_experiment(root=tmp_path)
    assert not meta["complete"] and len(rows)==280
    assert all(r["status"]=="failed" for r in rows)


def test_interrupt_is_recorded(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"preflight",Mock(return_value=mock_preflight()))
    monkeypatch.setattr(analysis,"analyze",Mock())
    monkeypatch.setattr(run.shared.requests,"post",Mock(side_effect=KeyboardInterrupt()))
    meta,rows,_=run.run_experiment(root=tmp_path)
    assert len(rows)==1 and rows[0]["status"]=="failed" and "Interrupted" in meta["stop_reason"]
