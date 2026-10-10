"""Sequential fixed-workload benchmark; no retries or endpoint changes."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit

import requests

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "single_worker_benchmark.csv"
FIELDS = ("run", "status", "api_status", "http_status", "worker_id", "gpu", "total_samples",
          "client_wall_time_ms", "server_wall_time_ms", "gpu_inference_time_ms",
          "effective_throughput_images_per_second", "output_shape", "error")


def positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def parse_response(data):
    """Keep partial measurements, but never count an unverified reply as passed."""
    values = {}
    errors = []
    if not isinstance(data, dict):
        return values, ["Response must be a JSON object"]
    for target, aliases in {
        "api_status": ("status",), "worker_id": ("worker_id", "worker_name"),
        "gpu": ("gpu", "gpu_model"),
    }.items():
        value = next((data[k] for k in aliases if k in data), None)
        if isinstance(value, str) and value.strip():
            values[target] = value
        else:
            errors.append(f"Missing or invalid {target}")
    if values.get("api_status") != "passed":
        errors.append("API status is not passed")
    for target, source in {
        "total_samples": "total_samples", "server_wall_time_ms": "wall_time_ms",
        "gpu_inference_time_ms": "total_gpu_time_ms",
        "effective_throughput_images_per_second": "effective_throughput_images_per_second",
    }.items():
        value = positive_number(data.get(source))
        if value is None:
            errors.append(f"Missing or invalid {source}")
        else:
            values[target] = value
    if values.get("total_samples") != 64:
        errors.append("Expected total_samples=64")
    shape = data.get("output_shape")
    if isinstance(shape, (list, tuple)):
        values["output_shape"] = json.dumps(shape)
    if shape != [64, 10] and shape != (64, 10):
        errors.append("Expected output_shape=[64,10]")
    return values, errors


def attempt(controller_url, run, timeout, *, endpoint="/inference/single-worker",
            fields=FIELDS, parse=None, body=None):
    row = dict.fromkeys(fields, "")
    row.update(run=run, status="failed")
    response = None
    start = time.perf_counter()
    try:
        try:
            response = requests.post(controller_url.rstrip("/") + endpoint,
                                     json={} if body is None else body, timeout=timeout, allow_redirects=False)
        finally:
            row["client_wall_time_ms"] = (time.perf_counter() - start) * 1000
        row["http_status"] = response.status_code
        try:
            data = response.json()
        except ValueError:
            row["error"] = f"HTTP {response.status_code}: invalid JSON; {response.text}"
            return row
        values, errors = (parse or parse_response)(data)
        row.update(values)
        if not 200 <= response.status_code < 300:
            detail = data.get("detail", data.get("error", data)) if isinstance(data, dict) else data
            row["error"] = f"HTTP {response.status_code}: {json.dumps(detail, ensure_ascii=False)}"
        elif errors:
            row["error"] = "; ".join(errors)
        else:
            row["status"] = "passed"
    except requests.RequestException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
    return row


def summarize(rows):
    """Compute benchmark statistics only from successful, validated attempts."""
    successful = [row for row in rows if row["status"] == "passed"]
    wall = [float(row["client_wall_time_ms"]) for row in successful]
    summary = {"attempted": len(rows), "successful": len(successful),
               "failed": len(rows) - len(successful)}
    summary.update({
        "mean_client_wall_time_ms": statistics.mean(wall) if wall else None,
        "median_client_wall_time_ms": statistics.median(wall) if wall else None,
        "sample_stdev_client_wall_time_ms": statistics.stdev(wall) if len(wall) > 1 else None,
        "min_client_wall_time_ms": min(wall) if wall else None,
        "max_client_wall_time_ms": max(wall) if wall else None,
        "mean_gpu_inference_time_ms": statistics.mean(float(r["gpu_inference_time_ms"]) for r in successful) if successful else None,
        "mean_effective_throughput_images_per_second": statistics.mean(float(r["effective_throughput_images_per_second"]) for r in successful) if successful else None,
    })
    return summary


def run_benchmark(controller_url, runs=10, timeout=135, output=DEFAULT_OUTPUT, *,
                  fields=FIELDS, attempt_fn=None, summarize_fn=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Persist each attempt immediately; an interruption retains completed rows.
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number in range(1, runs + 1):
            row = (attempt_fn or attempt)(controller_url, number, timeout)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"Run {number}/{runs}: {row['status']} | client {row['client_wall_time_ms']:.3f} ms | {row['error']}", flush=True)
    summary = (summarize_fn or summarize)(rows)
    print("Summary (validated successful runs only; sample standard deviation):")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"CSV: {output.resolve()}")
    return rows, summary


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("runs must be positive")
    return number


def timeout_value(value):
    number = positive_number(value)
    if number is None:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def main(runner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--timeout", type=timeout_value, default=135)
    args = parser.parse_args()
    try:
        url = urlsplit(args.controller_url)
        valid = url.scheme in ("http", "https") and url.hostname and not (
            url.username or url.password or url.query or url.fragment)
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        parser.error("controller URL must be HTTP(S), without credentials/query/fragment")
    _, summary = (runner or run_benchmark)(args.controller_url, args.runs, args.timeout)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
