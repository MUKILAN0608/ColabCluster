# Workload scaling benchmark

Measurements: `C:\Users\Mukil\New folder (12)\colabcluster\results\scaling\20261010T093440Z-b49f5e6d\measurements.csv`

Target: 5 successful trials per configuration and workload.

## Readiness / completion

Preflight passed. See counts below for completion.

| Samples | Workers | Passed | Failed | Mean client ms | Median ms | Sample SD ms | Mean server ms | Mean summed GPU ms | Mean reported images/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 5 | 0 | 1024.557 | 562.346 | 1052.529 | 1003.842 | 0.539 | 100.560 |
| 64 | 2 | 5 | 0 | 1180.958 | 643.101 | 959.283 | 1154.404 | 0.692 | 79.563 |
| 256 | 1 | 5 | 0 | 573.865 | 565.431 | 38.599 | 555.016 | 1.844 | 463.306 |
| 256 | 2 | 5 | 0 | 635.736 | 608.867 | 76.901 | 618.392 | 2.042 | 419.047 |
| 1024 | 1 | 5 | 0 | 653.830 | 642.581 | 37.225 | 642.164 | 6.827 | 1598.346 |
| 1024 | 2 | 5 | 0 | 657.513 | 664.577 | 22.149 | 642.600 | 7.021 | 1594.931 |
| 4096 | 1 | 5 | 0 | 827.478 | 789.199 | 66.485 | 813.246 | 23.719 | 5064.616 |
| 4096 | 2 | 5 | 0 | 733.726 | 735.750 | 25.253 | 719.372 | 26.685 | 5699.230 |

## Observed comparisons

| Samples | Speedup (one/two client means) | Wall-time change % | Reported throughput change % | Observation |
|---|---:|---:|---:|---|
| 64 | 0.868 | 15.27 | -20.88 | Two workers slower in observed means |
| 256 | 0.903 | 10.78 | -9.55 | Two workers slower in observed means |
| 1024 | 0.994 | 0.56 | -0.21 | Two workers slower in observed means |
| 4096 | 1.128 | -11.33 | 12.53 | Two workers faster in observed means |

## Methodology and limitations

Each workload uses float32 SmallCNN, seed-0 model initialization, ten warmups and one timed CUDA forward. Global 32-sample blocks use private seeds 1000+block, so one/two-worker configurations process identical synthetic inputs. Workers report assigned batch sizes and validated output shapes; totals must match the request.
Only 64, 256, 1024 and 4096 are accepted. These divide evenly; the partition helper assigns any remainder to the first worker and is tested with odd totals. This does not enable other workload sizes.
Requests are sequential; one/two-worker configurations alternate each round and which goes first alternates by round. Workload sizes run in ascending order. Each configuration stops at the target successes, with at most twice that many attempts. Failed attempts remain in the CSV. This is bounded failure handling, not hidden retries.
Client perf_counter timing covers complete HTTP requests before JSON parsing, identically for both configurations. Server wall time includes executor, HTTP, setup, warmup and response validation. GPU time sums worker synchronized compute durations, not elapsed parallel time. Reported throughput uses server wall time, not client wall time.
No successful outliers are removed. Failed attempts are excluded from means but counted. Small trial counts and variable network/runtime conditions do not establish statistical significance or general scaling behavior. No speedup conclusion is produced until each configuration meets the requested target.
