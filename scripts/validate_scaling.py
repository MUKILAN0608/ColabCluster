"""Combined confirmation/crossover experiment; bounded sequential requests only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scaling import VALIDATION_SIZES, WORKER_IDS
from controller.inference import diagnostic_url
from common.schemas import WorkerInfo
from scripts import benchmark_scaling as scaling
from scripts import benchmark_single_worker as shared

METHOD = dict(model="SmallCNN", model_seed=0, input_seed_base=1000, input_block_samples=32,
              dtype="float32", device="cuda:0", warmups=10, timed_forwards=1,
              timing="client perf_counter complete HTTP body before JSON parsing",
              order="seeded balanced alternating rounds", protocol="scaling-validation-v1")
FIELDS = scaling.FIELDS + ("experiment_id", "timestamp_utc", "round", "order", "phase", "seed", "worker_versions")


def safe_error(exc):
    # Avoid persisting remote URLs, credentials or arbitrary upstream error bodies.
    text = str(exc)
    text = re.sub(r"https?://[^\s'\"]+", "[endpoint]", text)
    text = re.sub(r"(?i)(token|password|secret|authorization)\s*[:=]\s*[^,;\s]+", r"\1=[redacted]", text)
    return f"{type(exc).__name__}: {text[:600]}"


def sanitized_workers(workers):
    return {key: {"worker_id":w["worker_id"], "gpu":w["gpu"], "gpu_memory":w["gpu_memory"],
        "registered_at":w.get("registered_at"), "cuda_available":w.get("cuda_available"),
        "software":{k:w.get("metadata",{}).get(k) for k in ("platform","python_version","torch_version")}}
        for key,w in workers.items()}


def preflight(url, sizes):
    workers = scaling.preflight(url)
    endpoints = [url] + [diagnostic_url(WorkerInfo.model_validate(w)).removesuffix("/inference") for w in workers.values()]
    for index, endpoint in enumerate(endpoints):
        schema = scaling.get_json(endpoint + "/openapi.json")
        definitions = schema.get("components",{}).get("schemas",{})
        name = "ScalingRequest" if index == 0 else "ScalingBatchRequest"
        allowed = definitions.get(name,{}).get("properties",{}).get("total_samples",{}).get("enum",[])
        if not set(sizes).issubset(allowed):
            raise ValueError(f"{'Controller' if index==0 else WORKER_IDS[index-1]} has outdated workload validation; restart with the current bundle")
    return workers


def order_for_round(round_number, first):
    return (first, 3-first) if round_number % 2 else (3-first, first)


def software_metadata():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        commit = "unavailable"
    sources = ["common/scaling.py","worker/inference.py","worker/cnn_test.py","controller/scaling.py",
               "scripts/validate_scaling.py","scripts/analyze_scaling_validation.py"]
    return dict(python=sys.version.split()[0], git_commit=commit,
        packages={name:importlib.metadata.version(name) for name in ("requests","pydantic","matplotlib")},
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})


def run_experiment(url="http://127.0.0.1:8000", trials=20, exploratory=5, seed=20261010,
                   timeout=135, root=scaling.ROOT):
    if trials < 20 or exploratory < 5:
        raise ValueError("Confirmation requires >=20 and exploration >=5 successful trials")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    identifier="validation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
    directory=root/identifier;directory.mkdir()
    targets={str(n):trials if n in (1024,4096) else exploratory for n in VALIDATION_SIZES}
    metadata=dict(experiment_id=identifier,created_utc=datetime.now(timezone.utc).isoformat(),method=METHOD,
        seed=seed,targets=targets,timeout_seconds=timeout,attempt_multiplier=2,software=software_metadata(),
        stages=[1024,4096]+[n for n in VALIDATION_SIZES if n not in (1024,4096)],
        complete=False,stop_reason="",workers={})
    path=directory/"metadata.json"
    def persist():path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    persist()
    rng=random.Random(seed);rows=[];baseline=None
    def check():
        nonlocal baseline
        current=sanitized_workers(preflight(url.rstrip("/"),VALIDATION_SIZES))
        if baseline is None:baseline=current;metadata["workers"]=current;persist()
        elif current!=baseline:raise ValueError("Worker identity, registration or software changed; refusing to pool runtimes")
    try:
        with (directory/"measurements.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();stream.flush()
            check()
            for size in metadata["stages"]:
                target=targets[str(size)];success={1:0,2:0};first=rng.choice((1,2))
                for round_number in range(1,2*target+1):
                    for position,count in enumerate(order_for_round(round_number,first),1):
                        if success[count]>=target:continue
                        check()  # stop before another dispatch if a worker disappeared
                        versions={}
                        def parse(data):
                            values,errors=scaling.parse_response(data,size,count)
                            if not errors:
                                versions.update({w["worker_id"]:{"cuda_version":w.get("cuda_version"),"torch_version":w.get("torch_version")} for w in data["workers"]})
                            return values,errors
                        interrupted = None
                        start = time.perf_counter()
                        try:
                            row=shared.attempt(url,len(rows)+1,timeout,endpoint="/inference/scaling",
                                fields=FIELDS,parse=parse,body=dict(total_samples=size,worker_count=count))
                        except (Exception, KeyboardInterrupt) as exc:
                            interrupted = exc
                            row=dict.fromkeys(FIELDS, "")
                            row.update(run=len(rows)+1,status="failed",client_wall_time_ms=(time.perf_counter()-start)*1000,
                                       error=f"Interrupted/unexpected failure: {type(exc).__name__}")
                        row.update(total_samples=size,worker_count=count,experiment_id=identifier,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),round=round_number,order=position,
                            phase="confirmation" if size in (1024,4096) else "exploration",seed=seed,
                            worker_versions=json.dumps(versions))
                        if row["error"]:row["error"]=safe_error(RuntimeError(row["error"]))
                        rows.append(row);writer.writerow(row);stream.flush()
                        success[count]+=row["status"]=="passed"
                        print(f"samples={size} round={round_number} workers={count}: {row['status']}",flush=True)
                        if interrupted is not None:
                            raise interrupted
                        if row["status"] != "passed":
                            # Failure retained. GET preflight must pass before any new attempt.
                            check()
                    if all(v>=target for v in success.values()):break
    except KeyboardInterrupt:
        metadata["stop_reason"]="Interrupted by operator; completed attempts preserved"
    except Exception as exc:
        metadata["stop_reason"]=safe_error(exc)
    finally:
        metadata["complete"]=all(sum(r["status"]=="passed" and r["total_samples"]==int(size) and r["worker_count"]==count for r in rows)>=target
            for size,target in targets.items() for count in (1,2))
        persist()
    from scripts.analyze_scaling_validation import analyze
    analyze(directory)
    print(f"Output: {directory}",flush=True)
    print(metadata["stop_reason"] or ("Completed targets" if metadata["complete"] else "Attempt limits reached"),flush=True)
    return metadata,rows,directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url",default="http://127.0.0.1:8000")
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--exploratory-trials",type=int,default=5)
    parser.add_argument("--seed",type=int,default=20261010)
    parser.add_argument("--timeout",type=shared.timeout_value,default=135)
    args=parser.parse_args()
    if args.trials<20 or args.exploratory_trials<5:parser.error("Minimum targets: 20 confirmation, 5 exploration")
    from urllib.parse import urlsplit
    parsed=urlsplit(args.controller_url)
    if parsed.scheme not in ("http","https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("Use an HTTP(S) controller URL without credentials, query or fragment")
    metadata,_,_=run_experiment(args.controller_url,args.trials,args.exploratory_trials,args.seed,args.timeout)
    return 0 if metadata["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
