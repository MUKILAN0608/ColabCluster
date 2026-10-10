# ColabCluster repeatability and crossover validation

## Setup and provenance

Experiment: `validation_20261010T113208Z_19ec2613`. Raw attempts: 0. Inputs are read-only; SHA256 hashes are in analysis.json.
Stop reason: ValueError: Required worker COLAB-GPU-TEST is missing or not READY.
Software: {"git_commit": "579bdfef5e4dd38e4b409b663acb7701ca26ba48", "packages": {"matplotlib": "3.11.2", "pydantic": "2.13.5", "requests": "2.34.2"}, "python": "3.14.3", "source_sha256": {"common/scaling.py": "bd0717011d689d64cae8eed07df61fd1d89659b2ab31e65528dfd345ae0d8b14", "controller/scaling.py": "72cc00162913cc7204cacf5efe0876d3e72d7382982011e989cce5350a1e6297", "scripts/analyze_scaling_validation.py": "df9743289c04e6ca11da0122f3b80ee8a700f328f5e9a919d5fe07de2faf1818", "scripts/validate_scaling.py": "2d72485ef478b73db5aef01daf72f965ee463a1dd07cbd074b99df023825a36e", "worker/cnn_test.py": "3fbfebc251a939c06a26d9382c323224183960ff375b78cbd58266d7dc0feb12", "worker/inference.py": "7d5abb426aa9852b3d180d718190d5bbb1ad17849432f3f5b657056876c9fcaf"}}

Recorded workers (no tunnel addresses or credentials):
```json
{}
```
Observed response CUDA/PyTorch versions:
```json
{}
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
| 1024 | 1 | 0/20 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 1024 | 2 | 0/20 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 1536 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 1536 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 2048 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 2048 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 3072 | 1 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 3072 | 2 | 0/5 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 4096 | 1 | 0/20 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |
| 4096 | 2 | 0/20 | 0 | N/A | N/A | N/A | N/A | N/A/N/A | N/A, N/A | N/A/N/A | N/A | N/A |

## Calculated effects (only when both targets are met)

| Samples | Complete | Speedup | Wall change % | Difference two-minus-one ms | Throughput change % | Speedup CI | Difference CI ms | Paired/unpaired rounds |
|---|---|---|---|---|---|---|---|---|
| 256 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 512 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 768 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
| 1024 | False | N/A | N/A | N/A | N/A | N/A | N/A | 0/0 |
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

No failed inference attempts recorded. Zero attempts after a blocked preflight is not a successful experiment.
- Incomplete: 256 samples, 1 worker(s): 0/5 successful.
- Incomplete: 256 samples, 2 worker(s): 0/5 successful.
- Incomplete: 512 samples, 1 worker(s): 0/5 successful.
- Incomplete: 512 samples, 2 worker(s): 0/5 successful.
- Incomplete: 768 samples, 1 worker(s): 0/5 successful.
- Incomplete: 768 samples, 2 worker(s): 0/5 successful.
- Incomplete: 1024 samples, 1 worker(s): 0/20 successful.
- Incomplete: 1024 samples, 2 worker(s): 0/20 successful.
- Incomplete: 1536 samples, 1 worker(s): 0/5 successful.
- Incomplete: 1536 samples, 2 worker(s): 0/5 successful.
- Incomplete: 2048 samples, 1 worker(s): 0/5 successful.
- Incomplete: 2048 samples, 2 worker(s): 0/5 successful.
- Incomplete: 3072 samples, 1 worker(s): 0/5 successful.
- Incomplete: 3072 samples, 2 worker(s): 0/5 successful.
- Incomplete: 4096 samples, 1 worker(s): 0/20 successful.
- Incomplete: 4096 samples, 2 worker(s): 0/20 successful.

## Figures

![Client wall time](wall_time.png)
![Speedup](speedup.png)
![Reported throughput](throughput.png)
![Trial variability](trials.png)
