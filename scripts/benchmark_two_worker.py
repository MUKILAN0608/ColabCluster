"""Sequential two-worker benchmark using the single-worker measurement conventions."""
import json
from pathlib import Path
import statistics

try:
    from . import benchmark_single_worker as shared
except ImportError:
    import benchmark_single_worker as shared

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "two_worker_benchmark.csv"
FIELDS = ("run", "status", "api_status", "http_status", "client_wall_time_ms",
          "server_wall_time_ms", "total_samples", "worker_count", "worker_ids", "gpu_models",
          "combined_gpu_time_ms", "effective_throughput_images_per_second", "output_shapes", "error")


def parse_response(data):
    values, errors = {}, []
    if not isinstance(data, dict):
        return values, ["Response must be a JSON object"]
    if isinstance(data.get("status"), str):
        values["api_status"] = data["status"]
    if data.get("status") != "passed":
        errors.append("API status is not passed")
    for target, source in {
        "server_wall_time_ms": "parallel_wall_time_ms", "total_samples": "total_samples",
        "worker_count": "worker_count",
        "effective_throughput_images_per_second": "effective_throughput_images_per_second",
    }.items():
        number = shared.positive_number(data.get(source))
        if number is None:
            errors.append(f"Missing or invalid {source}")
        else:
            values[target] = number
    if values.get("total_samples") != 64:
        errors.append("Expected total_samples=64")
    if values.get("worker_count") != 2:
        errors.append("Expected worker_count=2")
    workers = data.get("workers")
    if not isinstance(workers, list) or len(workers) != 2 or not all(isinstance(w, dict) for w in workers):
        errors.append("Expected two worker results")
        workers = []
    ids, gpus, times = [], [], []
    for worker in workers:
        identifier = worker.get("worker_id")
        gpu = worker.get("gpu")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append("Missing worker ID")
        else:
            ids.append(identifier)
        if not isinstance(gpu, str) or not gpu.strip():
            errors.append("Missing GPU model")
        else:
            gpus.append(gpu)
        if (worker.get("status") != "passed" or worker.get("device") != "cuda:0"
                or shared.positive_number(worker.get("batch_size")) != 32
                or worker.get("output_shape") != [32, 10]):
            errors.append("Worker must report passed CUDA inference for 32 samples with output [32,10]")
        number = shared.positive_number(worker.get("total_gpu_time_ms"))
        if number is not None:
            times.append(number)
    values["worker_ids"] = json.dumps(ids)
    values["gpu_models"] = json.dumps(gpus)
    if len(set(ids)) != 2:
        errors.append("Expected two distinct worker IDs")
    shapes = data.get("output_shapes")
    if isinstance(shapes, list):
        values["output_shapes"] = json.dumps(shapes)
    if shapes != [[32, 10], [32, 10]]:
        errors.append("Expected output_shapes=[[32,10],[32,10]]")
    combined = data.get("sum_worker_gpu_time_ms")
    if combined is not None:
        number = shared.positive_number(combined)
        if number is None:
            errors.append("Invalid sum_worker_gpu_time_ms")
        else:
            values["combined_gpu_time_ms"] = number
    elif len(times) == 2:
        values["combined_gpu_time_ms"] = sum(times)
    return values, errors


def attempt(controller_url, run, timeout):
    return shared.attempt(controller_url, run, timeout, endpoint="/inference/two-worker",
                          fields=FIELDS, parse=parse_response)


def summarize(rows):
    # Reuse identical wall-time statistics; GPU timing is optional for this CSV.
    adapted = [dict(row, gpu_inference_time_ms=0) for row in rows]
    summary = shared.summarize(adapted)
    del summary["mean_gpu_inference_time_ms"]
    gpu_times = [number for row in rows if row["status"] == "passed"
                 if (number := shared.positive_number(row.get("combined_gpu_time_ms"))) is not None]
    summary["mean_combined_gpu_time_ms"] = statistics.mean(gpu_times) if gpu_times else None
    summary["gpu_time_available_runs"] = len(gpu_times)
    return summary


def run_benchmark(controller_url, runs=10, timeout=135, output=DEFAULT_OUTPUT):
    return shared.run_benchmark(controller_url, runs, timeout, output, fields=FIELDS,
                                attempt_fn=attempt, summarize_fn=summarize)


if __name__ == "__main__":
    raise SystemExit(shared.main(runner=run_benchmark))
