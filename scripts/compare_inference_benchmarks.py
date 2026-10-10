"""Compare saved inference measurements without running requests or changing inputs."""
import argparse
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"run", "status", "client_wall_time_ms", "server_wall_time_ms",
            "total_samples", "effective_throughput_images_per_second"}


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def load_measurements(path, *, two_worker=False):
    """Flag invalid rows explicitly; optional missing GPU timing does not discard a run."""
    gpu_column = "combined_gpu_time_ms" if two_worker else "gpu_inference_time_ms"
    rows, excluded, notes, seen = [], [], [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        attempted = 0
        for line, row in enumerate(reader, 2):
            attempted += 1
            reasons = []
            run = row.get("run", "")
            if not run or not run.isdigit() or int(run) < 1:
                reasons.append("invalid run number")
            elif run in seen:
                reasons.append("duplicate run number")
            seen.add(run)
            if row.get("status") != "passed":
                reasons.append(f"status={row.get('status')!r}; {row.get('error', '')}")
            if "http_status" in row:
                try:
                    if not 200 <= int(row["http_status"]) < 300:
                        reasons.append("non-success HTTP status")
                except (ValueError, TypeError):
                    reasons.append("invalid HTTP status")
            if "api_status" in row and row["api_status"] != "passed":
                reasons.append("API status not passed")
            parsed = {key: number(row.get(key)) for key in
                      ("client_wall_time_ms", "server_wall_time_ms", "effective_throughput_images_per_second")}
            for key, value in parsed.items():
                if value is None:
                    reasons.append(f"missing/invalid {key}")
            if number(row.get("total_samples")) != 64:
                reasons.append("expected 64 total samples")
            if two_worker and "worker_count" in row and number(row["worker_count"]) != 2:
                reasons.append("expected two workers")
            if None in row:
                reasons.append("extra CSV fields")
            gpu = number(row.get(gpu_column))
            if gpu is None:
                notes.append(f"CSV line {line}, run {run}: GPU time unavailable/invalid")
            parsed["gpu_time_ms"] = gpu
            if reasons:
                excluded.append(f"CSV line {line}, run {run}: " + "; ".join(reasons))
            else:
                rows.append(parsed)
    wall = [r["client_wall_time_ms"] for r in rows]
    gpu = [r["gpu_time_ms"] for r in rows if r["gpu_time_ms"] is not None]
    def mean(values):
        return statistics.mean(values) if values else None
    return dict(attempted=attempted, successful=len(rows), failed=attempted-len(rows),
        mean_client_wall_time_ms=mean(wall), median_client_wall_time_ms=statistics.median(wall) if wall else None,
        sample_stdev_client_wall_time_ms=statistics.stdev(wall) if len(wall)>1 else None,
        min_client_wall_time_ms=min(wall) if wall else None, max_client_wall_time_ms=max(wall) if wall else None,
        mean_server_wall_time_ms=mean([r["server_wall_time_ms"] for r in rows]),
        mean_gpu_time_ms=mean(gpu), gpu_time_available_runs=len(gpu),
        mean_reported_throughput=mean([r["effective_throughput_images_per_second"] for r in rows]),
        excluded=excluded, notes=notes)


def compare(single, two):
    a, b = single["mean_client_wall_time_ms"], two["mean_client_wall_time_ms"]
    if a is None or b is None:
        return dict(interpretation="insufficient successful measurements", speedup=None,
                    wall_time_change_percent=None, throughput_change_percent=None, efficiency_percent=None)
    speedup = a / b
    change = (b-a)/a*100
    interpretation = "approximately equivalent" if abs(change) <= 5 else "slower" if change > 0 else "faster"
    return dict(interpretation=interpretation, speedup=speedup, wall_time_change_percent=change,
        throughput_change_percent=(two["mean_reported_throughput"] / single["mean_reported_throughput"]-1)*100,
        efficiency_percent=speedup/2*100)


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(single, two, findings, single_path, two_path):
    lines = ["# Single-worker vs. two-worker inference benchmark", "", "## Measured results", "",
        f"Inputs (read-only): `{single_path}` and `{two_path}`.", "",
        "| Metric | Single worker | Two workers |", "|---|---:|---:|"]
    for label, key in [("Attempted runs","attempted"),("Successful complete runs","successful"),
        ("Failed/incomplete runs","failed"),("Mean client wall time (ms)","mean_client_wall_time_ms"),
        ("Median client wall time (ms)","median_client_wall_time_ms"),
        ("Sample standard deviation (ms)","sample_stdev_client_wall_time_ms"),
        ("Minimum client wall time (ms)","min_client_wall_time_ms"),
        ("Maximum client wall time (ms)","max_client_wall_time_ms"),
        ("Mean server wall time (ms)","mean_server_wall_time_ms"),
        ("Mean GPU time (ms; two-worker value is summed GPU time)","mean_gpu_time_ms"),
        ("Runs with usable GPU time","gpu_time_available_runs"),
        ("Mean reported effective throughput (images/s)","mean_reported_throughput")]:
        digits = 3 if key == "mean_gpu_time_ms" else 2
        lines.append(f"| {label} | {fmt(single[key],digits)} | {fmt(two[key],digits)} |")
    lines += ["", "## Derived comparison", "",
        f"- Speedup = single mean / two-worker mean: **{fmt(findings['speedup'],3)}x**.",
        f"- Wall-time change = (two mean - single mean) / single mean: **{fmt(findings['wall_time_change_percent'])}%**.",
        f"- Mean reported throughput change = (two throughput / single throughput - 1): **{fmt(findings['throughput_change_percent'])}%**.",
        f"- Conventional parallel efficiency = speedup / 2 x 100: **{fmt(findings['efficiency_percent'])}%**.",
        "", "## Interpretation", "",
        f"Observed two-worker configuration: **{findings['interpretation']}** for this measured workload.",
        "Approximately equivalent means a mean wall-time difference within +/-5%; this is an explicit descriptive threshold, not a statistical test.",
        "When speedup is below 1, efficiency is not evidence of useful scaling: the two-worker mean is slower despite using two GPUs.",
        "These observations do not establish a general property of distributed inference or statistical significance.",
        "", "## Methodology and limitations", "",
        "Both configurations use SmallCNN and 64 synthetic samples: one worker processes a 64-sample batch; two workers process 32 samples each concurrently. Each benchmark issues sequential requests.",
        "Client wall time is the primary end-to-end measurement, timed with perf_counter around complete HTTP requests before JSON parsing. It includes remote HTTP/tunnel latency, worker setup, warmup and response overhead; it is not GPU-only execution time.",
        "Server wall time is reported separately. Its exact timing boundary differs: the two-worker measurement includes thread-pool and result validation overhead; the single-worker timer stops at HTTP response receipt. GPU time excludes setup/warmup; summed two-worker GPU time is not elapsed parallel wall time.",
        "Reported effective throughput is the arithmetic mean of API-reported rates based on server wall time, not 64 divided by mean client wall time. These quantities are intentionally distinguished.",
        "The existing inputs contain only 10 runs per configuration, collected separately rather than as randomized/interleaved paired trials. Network variation, ephemeral runtimes and first-request effects limit inference. No outliers or initial successful runs are removed, no significance test is claimed, and no causal overhead breakdown is inferred.",
        "Failed or incomplete rows are counted and listed below but excluded from timing statistics. Missing optional GPU times only reduce GPU-time availability. N/A indicates unavailable statistics; sample deviation requires at least two successful runs.",
        "", "## Data validation", ""]
    for label, data in [("Single worker",single),("Two workers",two)]:
        lines.append(f"### {label}")
        lines.append("")
        issues = data["excluded"] + data["notes"]
        lines += ["- " + issue.replace("\n"," ").replace("\r"," ") for issue in issues] if issues else ["No failed/incomplete rows or unavailable GPU timings detected."]
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, default=ROOT/"results/single_worker_benchmark.csv")
    parser.add_argument("--two", type=Path, default=ROOT/"results/two_worker_benchmark.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"results/benchmark_comparison.md")
    args = parser.parse_args()
    if args.output.resolve() in (args.single.resolve(),args.two.resolve()):
        parser.error("report output must not overwrite either input")
    try:
        single = load_measurements(args.single)
        two = load_measurements(args.two,two_worker=True)
    except (OSError,ValueError,csv.Error) as exc:
        parser.error(str(exc))
    findings = compare(single,two)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render_report(single,two,findings,args.single,args.two),encoding="utf-8")
    print(f"Single mean client wall time: {fmt(single['mean_client_wall_time_ms'])} ms")
    print(f"Two-worker mean client wall time: {fmt(two['mean_client_wall_time_ms'])} ms")
    print(f"Observed: {findings['interpretation']}; speedup {fmt(findings['speedup'],3)}x; wall-time change {fmt(findings['wall_time_change_percent'])}%")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
