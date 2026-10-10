"""Comparison calculations and input validation using temporary CSVs."""
import csv
from pathlib import Path
import pytest
from scripts import compare_inference_benchmarks as tool


def write_csv(path, walls=(100,200,300), **changes):
    rows=[dict(run=str(i+1),status="passed",client_wall_time_ms=w,server_wall_time_ms=w-10 if isinstance(w,(int,float)) else 90,
               total_samples=64,effective_throughput_images_per_second=500,
               gpu_inference_time_ms=2,**changes) for i,w in enumerate(walls)]
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader();writer.writerows(rows)
    return rows


def test_statistics_and_input_preserved(tmp_path):
    path=tmp_path/"one.csv";write_csv(path)
    before=path.read_bytes()
    data=tool.load_measurements(path)
    assert data["successful"]==3 and data["failed"]==0
    assert data["mean_client_wall_time_ms"]==data["median_client_wall_time_ms"]==200
    assert data["sample_stdev_client_wall_time_ms"]==100
    assert data["min_client_wall_time_ms"]==100 and data["max_client_wall_time_ms"]==300
    assert data["mean_server_wall_time_ms"]==190
    assert data["mean_gpu_time_ms"]==2 and data["mean_reported_throughput"]==500
    assert path.read_bytes()==before


@pytest.mark.parametrize("ratio,interpretation",[(0.8,"faster"),(1.27,"slower"),(1.03,"approximately equivalent"),(1,"approximately equivalent")])
def test_derived_metrics_and_interpretation(ratio,interpretation):
    single=dict(mean_client_wall_time_ms=100,mean_reported_throughput=1000)
    two=dict(mean_client_wall_time_ms=100*ratio,mean_reported_throughput=800)
    data=tool.compare(single,two)
    assert data["speedup"]==pytest.approx(1/ratio)
    assert data["wall_time_change_percent"]==pytest.approx((ratio-1)*100)
    assert data["throughput_change_percent"]==pytest.approx(-20)
    assert data["efficiency_percent"]==pytest.approx(50/ratio)
    assert data["interpretation"]==interpretation


def test_missing_columns(tmp_path):
    path=tmp_path/"bad.csv";path.write_text("run,status\n1,passed\n")
    with pytest.raises(ValueError,match="missing required columns"):
        tool.load_measurements(path)


@pytest.mark.parametrize("bad",["", "nan", "inf", "-1", "bad"])
def test_incomplete_runs_are_flagged(tmp_path,bad):
    path=tmp_path/"incomplete.csv";write_csv(path,walls=(100,bad))
    data=tool.load_measurements(path)
    assert data["successful"]==1 and data["failed"]==1
    assert data["sample_stdev_client_wall_time_ms"] is None
    assert "run 2" in data["excluded"][0]
    assert "client_wall_time_ms" in data["excluded"][0]


def test_failed_row_and_optional_gpu(tmp_path):
    path=tmp_path/"rows.csv"
    path.write_text("run,status,client_wall_time_ms,server_wall_time_ms,total_samples,effective_throughput_images_per_second,error\n1,passed,100,90,64,500,\n2,failed,999,90,64,500,timeout\n")
    data=tool.load_measurements(path)
    assert data["attempted"]==2 and data["failed"]==1 and data["mean_client_wall_time_ms"]==100
    assert data["mean_gpu_time_ms"] is None and data["gpu_time_available_runs"]==0
    assert "timeout" in data["excluded"][0]
    assert data["notes"]


def test_combined_gpu_and_no_outlier_removal(tmp_path):
    path=tmp_path/"two.csv";write_csv(path,walls=(100,10000),combined_gpu_time_ms=5,worker_count=2)
    data=tool.load_measurements(path,two_worker=True)
    assert data["mean_gpu_time_ms"]==5
    assert data["mean_client_wall_time_ms"]==5050 and data["successful"]==2


def test_empty_csv_and_insufficient_results(tmp_path):
    path=tmp_path/"empty.csv"
    path.write_text(",".join(sorted(tool.REQUIRED))+"\n")
    data=tool.load_measurements(path)
    findings=tool.compare(data,data)
    assert findings["speedup"] is None
    assert findings["interpretation"]=="insufficient successful measurements"
    report=tool.render_report(data,data,findings,path,path)
    assert "N/A" in report and "statistical significance" in report


def test_main_writes_report_without_touching_inputs(tmp_path,monkeypatch):
    one,two,out=[tmp_path/name for name in ("one.csv","two.csv","report.md")]
    write_csv(one);write_csv(two,walls=(200,400,600),combined_gpu_time_ms=4,worker_count=2)
    before=(one.read_bytes(),two.read_bytes())
    monkeypatch.setattr("sys.argv",["compare","--single",str(one),"--two",str(two),"--output",str(out)])
    assert tool.main()==0
    report=out.read_text(encoding="utf-8")
    assert "**0.500x**" in report and "**slower**" in report
    assert "**25.00%**" in report and "not evidence of useful scaling" in report
    assert (one.read_bytes(),two.read_bytes())==before
    monkeypatch.setattr("sys.argv",["compare","--single",str(one),"--two",str(two),"--output",str(one)])
    with pytest.raises(SystemExit):tool.main()
    assert one.read_bytes()==before[0]
