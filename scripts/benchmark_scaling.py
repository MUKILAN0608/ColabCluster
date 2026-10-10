"""Bounded scaling trials with readiness preflight and append-only run directories."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_single_worker as shared
from common.scaling import SIZES, WORKER_IDS, ScalingResponse
from controller.inference import diagnostic_url
from common.schemas import WorkerInfo
import requests
from common.config import http_timeout
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1] / "results" / "scaling"
FIELDS = ("run", "total_samples", "worker_count", "status", "api_status", "http_status",
          "worker_ids", "assigned_samples", "gpu_models", "client_wall_time_ms", "server_wall_time_ms",
          "gpu_inference_time_ms", "worker_gpu_times_ms", "effective_throughput_images_per_second",
          "output_shapes", "error")


def get_json(url):
    timeout = http_timeout("preflight")
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        raise type(exc)(f"Preflight GET {url}; connect={timeout[0]:g}s read={timeout[1]:g}s; "
                        f"{type(exc).__name__}: {exc}") from exc
    try:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f"Expected HTTP 200 from {url}, got {response.status_code}")
        return response.json()
    finally:
        response.close()


def preflight(controller_url):
    """GET-only checks: no inference is issued until both updated workers are ready."""
    schema = get_json(controller_url + "/openapi.json")
    if "/inference/scaling" not in schema.get("paths", {}):
        raise ValueError("Controller lacks /inference/scaling; restart it with the updated source")
    workers = {w["worker_id"]: w for w in get_json(controller_url + "/workers")["workers"]}
    urls = []
    for identifier in WORKER_IDS:
        record = workers.get(identifier)
        if record is None or record.get("status") != "ready":
            raise ValueError(f"Required worker {identifier} is missing or not READY")
        worker = WorkerInfo.model_validate(record)
        base = diagnostic_url(worker).removesuffix("/inference")
        urls.append(base)
        schema = get_json(base + "/openapi.json")
        if "/inference/scaling" not in schema.get("paths", {}):
            raise ValueError(f"{identifier} lacks scaling support; upload the rebuilt bundle and restart it")
    if len(set(urls)) != 2:
        raise ValueError("Worker diagnostic URLs must differ")
    ready_ids = sorted(k for k,w in workers.items() if w.get("status") == "ready")
    if ready_ids[0] != WORKER_IDS[0]:
        raise ValueError("First READY worker is not COLAB-GPU-TEST; baseline worker would differ")
    return {identifier: workers[identifier] for identifier in WORKER_IDS}


def parse_response(data, size, count):
    try:
        result = ScalingResponse.model_validate(data)
        if result.total_samples != size or result.worker_count != count:
            raise ValueError("Response does not match requested workload")
        if [w.worker_id for w in result.workers] != list(WORKER_IDS[:count]):
            raise ValueError("Unexpected worker selection during benchmark")
    except ValueError as exc:
        return {}, [str(exc)]
    return dict(api_status=result.status, worker_ids=json.dumps([w.worker_id for w in result.workers]),
        assigned_samples=json.dumps([w.batch_size for w in result.workers]),
        gpu_models=json.dumps([w.gpu for w in result.workers]), server_wall_time_ms=result.wall_time_ms,
        gpu_inference_time_ms=result.sum_worker_gpu_time_ms,
        worker_gpu_times_ms=json.dumps([w.total_gpu_time_ms for w in result.workers]),
        effective_throughput_images_per_second=result.effective_throughput_images_per_second,
        output_shapes=json.dumps([w.output_shape for w in result.workers])), []


def grouped_statistics(rows):
    return {(size,count): shared.summarize([r for r in rows if int(r["total_samples"])==size
            and int(r["worker_count"])==count]) for size in SIZES for count in (1,2)}


def report(rows, trials, problem, csv_path):
    stats = grouped_statistics(rows)
    lines = ["# Workload scaling benchmark", "", f"Measurements: `{csv_path}`", "",
             f"Target: {trials} successful trials per configuration and workload.", "",
             "## Readiness / completion", "", problem or "Preflight passed. See counts below for completion.", "",
             "| Samples | Workers | Passed | Failed | Mean client ms | Median ms | Sample SD ms | Mean server ms | Mean summed GPU ms | Mean reported images/s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):return "N/A" if value is None else f"{value:.3f}"
    for (size,count),s in stats.items():
        values=[size,count,s["successful"],s["failed"]]+[fmt(s[k]) for k in
            ("mean_client_wall_time_ms","median_client_wall_time_ms","sample_stdev_client_wall_time_ms")]
        group=[r for r in rows if int(r["total_samples"])==size and int(r["worker_count"])==count and r["status"]=="passed"]
        values += [fmt(statistics.mean(float(r["server_wall_time_ms"]) for r in group)) if group else "N/A",
                   fmt(s["mean_gpu_inference_time_ms"]),fmt(s["mean_effective_throughput_images_per_second"])]
        lines.append("| " + " | ".join(map(str,values)) + " |")
    lines += ["", "## Observed comparisons", "", "| Samples | Speedup (one/two client means) | Wall-time change % | Reported throughput change % | Observation |", "|---|---:|---:|---:|---|"]
    for size in SIZES:
        a,b=stats[size,1],stats[size,2]
        if min(a["successful"],b["successful"]) < trials:
            lines.append(f"| {size} | N/A | N/A | N/A | Incomplete; target not met |")
            continue
        speed=a["mean_client_wall_time_ms"]/b["mean_client_wall_time_ms"]
        wall=(1/speed-1)*100
        throughput=(b["mean_effective_throughput_images_per_second"]/a["mean_effective_throughput_images_per_second"]-1)*100
        lines.append(f"| {size} | {speed:.3f} | {wall:.2f} | {throughput:.2f} | Two workers {'faster' if speed>1 else 'slower' if speed<1 else 'equal'} in observed means |")
    lines += ["", "## Methodology and limitations", "",
        "Each workload uses float32 SmallCNN, seed-0 model initialization, ten warmups and one timed CUDA forward. Global 32-sample blocks use private seeds 1000+block, so one/two-worker configurations process identical synthetic inputs. Workers report assigned batch sizes and validated output shapes; totals must match the request.",
        "Only 64, 256, 1024 and 4096 are accepted. These divide evenly; the partition helper assigns any remainder to the first worker and is tested with odd totals. This does not enable other workload sizes.",
        "Requests are sequential; one/two-worker configurations alternate each round and which goes first alternates by round. Workload sizes run in ascending order. Each configuration stops at the target successes, with at most twice that many attempts. Failed attempts remain in the CSV. This is bounded failure handling, not hidden retries.",
        "Client perf_counter timing covers complete HTTP requests before JSON parsing, identically for both configurations. Server wall time includes executor, HTTP, setup, warmup and response validation. GPU time sums worker synchronized compute durations, not elapsed parallel time. Reported throughput uses server wall time, not client wall time.",
        "No successful outliers are removed. Failed attempts are excluded from means but counted. Small trial counts and variable network/runtime conditions do not establish statistical significance or general scaling behavior. No speedup conclusion is produced until each configuration meets the requested target."]
    return "\n".join(lines)+"\n"


def run_benchmark(controller_url="http://127.0.0.1:8000", trials=5, timeout=135, output_root=ROOT):
    if trials < 5:
        raise ValueError("At least five successful trials are required")
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    run_dir=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8])
    run_dir.mkdir()
    csv_path=run_dir/"measurements.csv"
    rows=[];problem=""
    metadata=dict(controller_url=controller_url,trials=trials,timeout=timeout,max_attempts_per_configuration=trials*2)
    try:
        metadata["workers"]=preflight(controller_url.rstrip("/"))
    except (requests.RequestException,ValueError,KeyError,TypeError,AttributeError,HTTPException) as exc:
        problem=f"Preflight blocked: {type(exc).__name__}: {exc}"
    # Every invocation preserves its own preflight and measurements, even when blocked.
    metadata["preflight_error"]=problem
    (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    with csv_path.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();stream.flush()
        if not problem:
            for size in SIZES:
                successes={1:0,2:0}
                for round_index in range(trials*2):
                    for count in ((1,2) if round_index%2==0 else (2,1)):
                        if successes[count]>=trials:continue
                        row=shared.attempt(controller_url,len(rows)+1,timeout,endpoint="/inference/scaling",
                            fields=FIELDS,parse=lambda data:parse_response(data,size,count),
                            body=dict(total_samples=size,worker_count=count))
                        row.update(total_samples=size,worker_count=count)
                        rows.append(row);writer.writerow(row);stream.flush()
                        successes[count]+=row["status"]=="passed"
                        print(f"Samples={size} workers={count} attempt={round_index+1}: {row['status']} {row['error']}",flush=True)
                    if all(n>=trials for n in successes.values()):break
    text=report(rows,trials,problem,csv_path)
    (run_dir/"scaling_report.md").write_text(text,encoding="utf-8")
    (root/"scaling_report.md").write_text(text,encoding="utf-8")
    print(problem or "Scaling trials finished; see recorded completion counts.")
    print(f"Report: {root/'scaling_report.md'}")
    complete=not problem and all(s["successful"]>=trials for s in grouped_statistics(rows).values())
    return complete,rows,run_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url",default="http://127.0.0.1:8000")
    parser.add_argument("--trials",type=shared.positive_int,default=5)
    parser.add_argument("--timeout",type=shared.timeout_value,default=135)
    args=parser.parse_args()
    if args.trials<5:parser.error("--trials must be at least 5")
    complete,_,_=run_benchmark(args.controller_url,args.trials,args.timeout)
    return 0 if complete else 1


if __name__=="__main__":
    raise SystemExit(main())
