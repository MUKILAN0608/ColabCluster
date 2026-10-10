# ColabCluster repeatability and crossover validation

## Setup and provenance

Experiment: `validation_20261010T115546Z_5d593f1c`. Raw attempts: 46. Inputs are read-only; SHA256 hashes are in analysis.json.
Stop reason: ReadTimeout: HTTPSConnectionPool(host='generators-holiday-midi-survey.trycloudflare.com', port=443): Read timed out. (read timeout=5).
Software: {"git_commit": "21e2a4033fd0e792b2726760e006880c45b6f52f", "packages": {"matplotlib": "3.11.2", "pydantic": "2.13.5", "requests": "2.34.2"}, "python": "3.14.3", "source_sha256": {"common/scaling.py": "bd0717011d689d64cae8eed07df61fd1d89659b2ab31e65528dfd345ae0d8b14", "controller/scaling.py": "72cc00162913cc7204cacf5efe0876d3e72d7382982011e989cce5350a1e6297", "scripts/analyze_scaling_validation.py": "459d144b9e6e7622fbb5c047334cb58a7941afacf64966dacda81b398e9a9ed3", "scripts/validate_scaling.py": "9cda480d72b001920ef484b02ef2656c1a4e63412f2128d3f34cacddbae7c67f", "worker/cnn_test.py": "3fbfebc251a939c06a26d9382c323224183960ff375b78cbd58266d7dc0feb12", "worker/inference.py": "7d5abb426aa9852b3d180d718190d5bbb1ad17849432f3f5b657056876c9fcaf"}}

Recorded workers (no tunnel addresses or credentials):
```json
{
  "COLAB-GPU-TEST": {
    "worker_id": "COLAB-GPU-TEST",
    "gpu": "Tesla T4",
    "gpu_memory": 14.56317138671875,
    "registered_at": "2026-10-10T11:51:23.869370Z",
    "cuda_available": true,
    "software": {
      "platform": "Linux",
      "python_version": "3.13.15",
      "torch_version": "2.11.0+cu130"
    }
  },
  "COLAB-GPU-TEST-2": {
    "worker_id": "COLAB-GPU-TEST-2",
    "gpu": "Tesla T4",
    "gpu_memory": 14.56317138671875,
    "registered_at": "2026-10-10T11:55:16.343587Z",
    "cuda_available": true,
    "software": {
      "platform": "Linux",
      "python_version": "3.13.15",
      "torch_version": "2.11.0+cu130"
    }
  }
}
```
Observed response CUDA/PyTorch versions:
```json
{
  "COLAB-GPU-TEST": {
    "cuda_version": "13.0",
    "torch_version": "2.11.0+cu130"
  },
  "COLAB-GPU-TEST-2": {
    "cuda_version": "13.0",
    "torch_version": "2.11.0+cu130"
  }
}
```
Model/input configuration:
```json
{
  "model": "SmallCNN",
  "model_seed": 0,
  "input_seed_base": 1000,
  "input_block_samples": 32,
  "dtype": "float32",
  "device": "cuda:0",
  "warmups": 10,
  "timed_forwards": 1,
  "timing": "client perf_counter complete HTTP body before JSON parsing",
  "order": "seeded balanced alternating rounds",
  "protocol": "scaling-validation-v1"
}
```

## Descriptive measurements

| Samples | Workers | Passed/target | Failed | Mean ms | Median ms | SD ms | IQR ms | Min/max ms | Mean CI ms | Mean/median server ms | Summed GPU ms | Reported images/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 256 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 512 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 512 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 768 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 768 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 1024 | 1 | 20/20 | 0 | 663.798 | 661.574 | 42.079 | 40.044 | 603.429/787.894 | 647.109, 682.995 | 651.742/641.885 | 6.828 | 1576.215 |
| 1024 | 2 | 20/20 | 0 | 862.021 | 714.802 | 515.606 | 41.921 | 684.434/2995.957 | 718.862, 1094.348 | 852.994/707.775 | 7.017 | 1352.476 |
| 1536 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 1536 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 2048 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 2048 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 3072 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 3072 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 4096 | 1 | 3/20 | 0 | 984.096 | 1028.330 | 115.648 | 109.119 | 852.860/1071.098 | N/A, N/A | 973.600/1007.740 | 25.964 | 4246.538 |
| 4096 | 2 | 2/20 | 1 | 859.020 | 859.020 | 10.813 | 7.646 | 851.374/866.666 | N/A, N/A | 851.961/851.961 | 32.334 | 4808.171 |

## Calculated effects (only when both targets are met)

| Samples | Complete | Speedup | Wall change % | Difference two-minus-one ms | Throughput change % | Speedup CI | Difference CI ms | Paired/unpaired rounds |
|---|---|---|---|---|---|---|---|---|
| 256 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 512 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 768 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 1024 | True | 0.770 | 29.862 | 198.223 | -14.195 | [0.6046647756469555, 0.917113537696247] | [59.691968997503864, 433.8167046217493] | 20/0 |
| 1536 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 2048 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 3072 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 4096 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |

## Interpretation

4096-sample repeatable advantage supported under the stated bootstrap assumptions: **not established**.
Smallest tested workload with a completed measured mean advantage: **not established**. Neighboring tested workloads and uncertainty are shown above; this is not an exact crossover threshold.
A lower mean alone is not called statistically significant. Confirmation requires the full target and a two-minus-one 95% interval wholly below zero. This is conditional evidence within this session, not proof of universal scaling or persistence across Colab sessions.
If confirmation is absent or uncertain, the next experiment is a fresh fully updated session with the same balanced design and more independent rounds/sessions, not a scheduler or architectural change.

## Statistical method and limitations

Analysis seed 20261010; 5000 resamples; 95% percentile intervals. Means use order-stratified resampling of observed rounds. Comparisons resample whole rounds jointly within first-configuration strata, retaining one/two-worker dependence and unpaired successful observations. Fewer than five paired rounds suppress comparison intervals. Missing/failing trials are counted; successful outliers are never removed.
Assumptions: rounds are exchangeable within order strata; both configurations in a round share comparable conditions. Time drift/autocorrelation across rounds, failures not missing at random and changes across workloads can invalidate nominal coverage. With five exploratory rounds intervals are particularly unstable. Pointwise intervals across eight workloads are not multiplicity-adjusted; crossover selection is exploratory, not a confirmatory significance claim.
Client perf_counter covers complete HTTP response receipt before parsing. Server wall time includes remote model/input setup, warmup, dispatch and validation. Summed GPU time is compute across devices, never parallel elapsed time. Reported throughput is the mean of server-based sample rates, not samples divided by mean client wall time.
The two confirmation sizes run first. Exploratory sizes use the same session/method and confirmation observations are reused, not rerun or merged with older data. Older five-trial experiments lack sufficient round/seed/source metadata and are deliberately not pooled. Workload order and preflight traffic remain possible confounders. Worker/model inputs match but a random untrained model and synthetic data do not measure application accuracy.
Bootstrap reference: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). The local implementation uses the standard library and round-level sampling rather than adding SciPy.

## Failed attempts / incomplete configurations

- Run 46, samples 4096, workers 2, HTTP 504: RuntimeError: HTTP 504: "COLAB-GPU-TEST timed out; remote work may continue"
- Incomplete: 256 samples, 1 worker(s): 0/5 successful.
- Incomplete: 256 samples, 2 worker(s): 0/5 successful.
- Incomplete: 512 samples, 1 worker(s): 0/5 successful.
- Incomplete: 512 samples, 2 worker(s): 0/5 successful.
- Incomplete: 768 samples, 1 worker(s): 0/5 successful.
- Incomplete: 768 samples, 2 worker(s): 0/5 successful.
- Incomplete: 1536 samples, 1 worker(s): 0/5 successful.
- Incomplete: 1536 samples, 2 worker(s): 0/5 successful.
- Incomplete: 2048 samples, 1 worker(s): 0/5 successful.
- Incomplete: 2048 samples, 2 worker(s): 0/5 successful.
- Incomplete: 3072 samples, 1 worker(s): 0/5 successful.
- Incomplete: 3072 samples, 2 worker(s): 0/5 successful.
- Incomplete: 4096 samples, 1 worker(s): 3/20 successful.
- Incomplete: 4096 samples, 2 worker(s): 2/20 successful.

## Figures

![Client wall time](wall_time.png)
![Speedup](speedup.png)
![Reported throughput](throughput.png)
![Trial variability](trials.png)
