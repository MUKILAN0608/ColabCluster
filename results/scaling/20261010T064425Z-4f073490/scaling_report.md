# Workload scaling benchmark

Measurements: `C:\Users\Mukil\New folder (12)\colabcluster\results\scaling\20261010T064425Z-4f073490\measurements.csv`

Target: 5 successful trials per configuration and workload.

## Readiness / completion

Preflight blocked: ValueError: Controller lacks /inference/scaling; restart it with the updated source

| Samples | Workers | Passed | Failed | Mean client ms | Median ms | Sample SD ms | Mean server ms | Mean summed GPU ms | Mean reported images/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 64 | 2 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 256 | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 256 | 2 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 2 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 4096 | 1 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| 4096 | 2 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

## Observed comparisons

| Samples | Speedup (one/two client means) | Wall-time change % | Reported throughput change % | Observation |
|---|---:|---:|---:|---|
| 64 | N/A | N/A | N/A | Incomplete; target not met |
| 256 | N/A | N/A | N/A | Incomplete; target not met |
| 1024 | N/A | N/A | N/A | Incomplete; target not met |
| 4096 | N/A | N/A | N/A | Incomplete; target not met |

## Methodology and limitations

Each workload uses float32 SmallCNN, seed-0 model initialization, ten warmups and one timed CUDA forward. Global 32-sample blocks use private seeds 1000+block, so one/two-worker configurations process identical synthetic inputs. Workers report assigned batch sizes and validated output shapes; totals must match the request.
Only 64, 256, 1024 and 4096 are accepted. These divide evenly; the partition helper assigns any remainder to the first worker and is tested with odd totals. This does not enable other workload sizes.
Requests are sequential; one/two-worker configurations alternate each round and which goes first alternates by round. Workload sizes run in ascending order. Each configuration stops at the target successes, with at most twice that many attempts. Failed attempts remain in the CSV. This is bounded failure handling, not hidden retries.
Client perf_counter timing covers complete HTTP requests before JSON parsing, identically for both configurations. Server wall time includes executor, HTTP, setup, warmup and response validation. GPU time sums worker synchronized compute durations, not elapsed parallel time. Reported throughput uses server wall time, not client wall time.
No successful outliers are removed. Failed attempts are excluded from means but counted. Small trial counts and variable network/runtime conditions do not establish statistical significance or general scaling behavior. No speedup conclusion is produced until each configuration meets the requested target.
