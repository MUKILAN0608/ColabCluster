# Single-worker vs. two-worker inference benchmark

## Measured results

Inputs (read-only): `C:\Users\Mukil\New folder (12)\colabcluster\results\single_worker_benchmark.csv` and `C:\Users\Mukil\New folder (12)\colabcluster\results\two_worker_benchmark.csv`.

| Metric | Single worker | Two workers |
|---|---:|---:|
| Attempted runs | 10.00 | 10.00 |
| Successful complete runs | 10.00 | 10.00 |
| Failed/incomplete runs | 0.00 | 0.00 |
| Mean client wall time (ms) | 739.06 | 940.18 |
| Median client wall time (ms) | 647.80 | 858.16 |
| Sample standard deviation (ms) | 247.32 | 299.19 |
| Minimum client wall time (ms) | 572.22 | 625.94 |
| Maximum client wall time (ms) | 1360.60 | 1391.54 |
| Mean server wall time (ms) | 720.46 | 925.82 |
| Mean GPU time (ms; two-worker value is summed GPU time) | 0.574 | 0.719 |
| Runs with usable GPU time | 10.00 | 10.00 |
| Mean reported effective throughput (images/s) | 95.58 | 75.45 |

## Derived comparison

- Speedup = single mean / two-worker mean: **0.786x**.
- Wall-time change = (two mean - single mean) / single mean: **27.21%**.
- Mean reported throughput change = (two throughput / single throughput - 1): **-21.06%**.
- Conventional parallel efficiency = speedup / 2 x 100: **39.30%**.

## Interpretation

Observed two-worker configuration: **slower** for this measured workload.
Approximately equivalent means a mean wall-time difference within +/-5%; this is an explicit descriptive threshold, not a statistical test.
When speedup is below 1, efficiency is not evidence of useful scaling: the two-worker mean is slower despite using two GPUs.
These observations do not establish a general property of distributed inference or statistical significance.

## Methodology and limitations

Both configurations use SmallCNN and 64 synthetic samples: one worker processes a 64-sample batch; two workers process 32 samples each concurrently. Each benchmark issues sequential requests.
Client wall time is the primary end-to-end measurement, timed with perf_counter around complete HTTP requests before JSON parsing. It includes remote HTTP/tunnel latency, worker setup, warmup and response overhead; it is not GPU-only execution time.
Server wall time is reported separately. Its exact timing boundary differs: the two-worker measurement includes thread-pool and result validation overhead; the single-worker timer stops at HTTP response receipt. GPU time excludes setup/warmup; summed two-worker GPU time is not elapsed parallel wall time.
Reported effective throughput is the arithmetic mean of API-reported rates based on server wall time, not 64 divided by mean client wall time. These quantities are intentionally distinguished.
The existing inputs contain only 10 runs per configuration, collected separately rather than as randomized/interleaved paired trials. Network variation, ephemeral runtimes and first-request effects limit inference. No outliers or initial successful runs are removed, no significance test is claimed, and no causal overhead breakdown is inferred.
Failed or incomplete rows are counted and listed below but excluded from timing statistics. Missing optional GPU times only reduce GPU-time availability. N/A indicates unavailable statistics; sample deviation requires at least two successful runs.

## Data validation

### Single worker

No failed/incomplete rows or unavailable GPU timings detected.

### Two workers

No failed/incomplete rows or unavailable GPU timings detected.
