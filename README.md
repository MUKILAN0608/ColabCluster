# ColabCluster

ColabCluster is a lightweight open-source distributed AI orchestration platform for ephemeral and heterogeneous GPU workers.

## Project Vision

ColabCluster aims to make distributed AI workloads accessible by coordinating temporary, diverse GPU workers through a lightweight orchestration platform.

## Initial Goals

- GPU worker registration
- Worker health monitoring
- Heartbeats
- Task scheduling
- Distributed inference
- Resource-aware scheduling
- Worker failure recovery
- Heterogeneous GPU support

## Future Goals

- Ray integration
- PyTorch Distributed
- GPU-aware scheduling
- Adaptive scheduling
- Monitoring dashboard
- Local, cloud, and Colab worker backends

## Status

Step 7 adds multi-worker cluster summaries while preserving active-only cleanup.
Registration, hardware detection, and the shared protocol remain compatible.

## Controller

The ColabCluster controller is the control plane. It currently maintains an
in-memory worker registry and exposes health, registration, heartbeat, and
unregister APIs. Persistent storage and distributed scheduling are intentionally
not implemented yet. The worker registers once and sends periodic heartbeats.
There is no automatic tunneling or task execution.

From the `colabcluster` directory, use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
uvicorn controller.main:app --reload
```

The controller listens at http://127.0.0.1:8000. Interactive Swagger documentation
is available at http://127.0.0.1:8000/docs.

For subsequent starts from the parent workspace directory, run:

```powershell
.\start-controller.cmd
```

This launcher selects the project's virtual environment and changes to the
project directory automatically. It also accepts `--port 8001` if needed and
does not require PowerShell script execution to be enabled.

If you see `ModuleNotFoundError: No module named 'controller'`, stop Uvicorn
with Ctrl+C. The `controller` package is inside `colabcluster`, so running a
global Uvicorn from the parent directory will not find it. Use the launcher
above, or run these commands from the parent directory without activation:

```powershell
cd .\colabcluster
.\.venv\Scripts\python.exe -m uvicorn controller.main:app --reload
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Controller health and version |
| GET | `/workers` | Registered workers (`{"workers": []}` initially) |
| POST | `/workers/register` | Register worker metadata (201; duplicate ID: 409) |
| POST | `/workers/heartbeat` | Refresh `last_seen` (unknown ID: 404) |
| POST | `/workers/unregister` | Remove a worker (unknown ID: 404) |

Registration JSON example (GPU memory in GiB):

```json
{"worker_id": "manual-test", "gpu": "T4", "gpu_memory": 16, "status": "ready"}
```

Heartbeat and unregister still accept `{"worker_id": "manual-test"}`. Required strings
must be nonempty, GPU memory must be finite and nonnegative, and invalid requests
return 422. Registration and heartbeat return a `message` and the `worker` record,
including timezone-aware UTC `registered_at` and `last_seen` timestamps. Status is
one of `ready`, `busy`, `offline`, or `error`; heartbeat updates it only when
explicitly supplied.

Verify health in PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/workers
```

Expected health response:

```json
{"status": "healthy", "service": "colabcluster-controller", "version": "0.1.0"}
```

Run the complete test suite:

```powershell
python -m pytest -q
```

Registry data is lost on restart or development reload. Run a single server
process: multiple processes would each have an independent registry. No worker
process is required to try the API manually in Swagger UI.

## Worker Protocol

Workers communicate with the Controller through structured Pydantic schemas in
`common.schemas`. The protocol is backend-independent: future Colab, local GPU,
Docker, cloud GPU, and Kubernetes workers can use the same contracts.
Registration, heartbeat, and unregistration requests and responses are defined,
along with `WorkerInfo` and the `WorkerStatus` enum (`READY`, `BUSY`, `OFFLINE`,
`ERROR`, serialized as lowercase values). Future task/result messages will be
introduced in later stages. The Colab entry point registers and then sends heartbeat messages.

Registration requires `worker_id`, `gpu`, `gpu_memory`, and `status`. Optional
fields are `cpu_count`, `ram_total`, `cuda_available`, and `metadata`. All memory
quantities are in GiB; utilization is a percentage from 0 to 100.

Example heartbeat:

```json
{
  "worker_id": "manual-test",
  "status": "busy",
  "gpu_utilization": 75.0,
  "gpu_memory_used": 8.0,
  "timestamp": "2026-09-27T10:00:00Z"
}
```

Timestamps must include a timezone. For Step 2 compatibility, heartbeat status
and timestamp may be omitted: the model defaults to `ready` and current UTC,
but the controller preserves stored status when the request omits it. The
controller always sets `last_seen` and response `server_timestamp` using its own
UTC clock, independently of the worker's timestamp. The latest GPU telemetry is stored in the worker record; unavailable metrics
are represented by null. Telemetry is not used for scheduling.

Unregistration also accepts an optional `reason`. All three mutation responses
include `success`, `worker_id`, and `message`; heartbeat also includes
`server_timestamp`. Registration and heartbeat retain the Step 2 `worker`
response field. Unregistration reasons are accepted but not persisted.
Swagger `/docs` shows the shared request and response schemas.

## Worker Runtime

A worker represents one compute machine/runtime. It automatically detects its
logical CPU count, total RAM, Python/platform versions, and, when available,
the first CUDA GPU and its VRAM. Workers may be CPU-only or GPU-enabled.
`HardwareInfo` and `LocalWorkerInfo` are immutable Pydantic snapshots. Memory
values and CLI output use **GiB** (1 GiB = 1,073,741,824 bytes), matching the
existing protocol's units.

From the workspace directory, enter the project and run using its
virtual environment (no activation required):

```powershell
cd .\colabcluster
.\.venv\Scripts\python.exe -m worker.worker
```

With the project environment activated, the equivalent command is
`python -m worker.worker`. The command prints hardware information and exits.

Assign a worker ID in PowerShell:

```powershell
$env:COLABCLUSTER_WORKER_ID = "TEST-WORKER-01"
.\.venv\Scripts\python.exe -m worker.worker
Remove-Item Env:COLABCLUSTER_WORKER_ID
```

An explicit `Worker(worker_id="...")` takes precedence over the environment.
Missing or blank environment IDs use a UUID-based ID that remains stable for
the process lifetime; a new process generates a new fallback ID. Each Worker
retains its identity and hardware snapshot after initialization. Blank explicit
IDs are rejected. If the OS cannot report CPU count, the detector uses 1.

PyTorch is **optional** and is not installed by this project. Without it, or
when CUDA is unavailable, GPU detection returns `gpu="None"`, `gpu_memory=0`,
and `cuda_available=False`. A broken PyTorch installation or CUDA driver also
falls back gracefully, with a diagnostic warning. This fallback reports the
absence of a usable PyTorch CUDA device, not proof that the machine has no GPU.
The existing `psutil` dependency supplies CPU and RAM information.

The hardware-only `worker.worker` command still makes no HTTP requests. Use
`worker.colab_worker` below for the Step 5 registration workflow.


## Connecting a Google Colab Worker

1. Start the updated controller on your PC from `colabcluster`:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn controller.main:app --host 0.0.0.0 --port 8000
   ```

   Run `Invoke-RestMethod http://127.0.0.1:8000/health` in a second terminal.
   The existing `start-controller.cmd` launcher also works, binding to loopback
   by default for a tunnel forwarding to localhost. Restart any older controller
   process to load Step 6; in-memory registrations are cleared on restart.

2. Manually provide a reachable network/tunnel endpoint forwarding to port 8000.
   **127.0.0.1 inside Colab refers to Colab, not your PC.** This project does not
   create tunnels. The development API has no authentication; control temporary
   exposure and close it after testing. Do not commit tunnel credentials.

3. Open `examples/colab_worker.ipynb` in Colab. Publish/sync the updated project
   before using the repository clone. An existing checkout must also be updated;
   the notebook checks that the heartbeat module is present. Choose a GPU runtime
   if available, install dependencies, and verify detected hardware.

4. Configure the worker in Colab:

   ```python
   import os
   os.environ["COLABCLUSTER_CONTROLLER_URL"] = "https://YOUR-REACHABLE-CONTROLLER-ENDPOINT"
   os.environ["COLABCLUSTER_WORKER_ID"] = "COLAB-01"
   os.environ["COLABCLUSTER_HEARTBEAT_INTERVAL"] = "10"
   ```

5. Run `python -u -m worker.colab_worker` from the project directory. The notebook
   starts this command for you. It registers once, sends heartbeats, and keeps
   running. GPU memory values remain in GiB and enum values serialize lowercase.

6. On the PC, run the following twice, at least one heartbeat interval apart:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/workers | ConvertTo-Json -Depth 6
   ```

   Confirm the Colab worker's `last_seen` advances and its state stays `ready`
   (or `busy` if explicitly set). A successful registration alone is not proof
   of successful heartbeats.

The same workflow can be rehearsed locally in another terminal:

```powershell
$env:COLABCLUSTER_CONTROLLER_URL = "http://127.0.0.1:8000"
$env:COLABCLUSTER_WORKER_ID = "LOCAL-TEST-01"
.\.venv\Scripts\python.exe -u -m worker.colab_worker
```

The client validates acknowledgement schemas and worker IDs, reports HTTP and
transport failures, and uses connection/read timeouts of 5/15 seconds. The URL
defaults to localhost when unset. Registration is attempted once; HTTP 409 means
that ID is already registered. Explicitly unregister an existing entry through `/docs`, or wait for expiry,
before registering the same ID again. HTTP 404 during heartbeats
is logged and retried at the next interval; it never auto-registers the worker.

## Worker Heartbeat & Liveness

Workers send an immediate heartbeat after registration, then wait the configured
interval after each attempt. Temporary connection failures are logged and the
next interval tries again. No complex retries, task execution, or scheduling are
implemented. Slow requests can extend the time between successful heartbeats.

| Variable | Where to set it | Default | Meaning |
| --- | --- | --- | --- |
| `COLABCLUSTER_HEARTBEAT_INTERVAL` | Worker | `10` seconds | Delay between heartbeat attempts |
| `COLABCLUSTER_WORKER_TIMEOUT` | Controller | `30` seconds | Maximum age of `last_seen` before removal |

Both accept finite positive numbers, including fractional seconds. Invalid
configuration fails early. Keep the timeout comfortably longer than the heartbeat
interval plus expected request latency. Set controller configuration before
starting/restarting the server, and worker configuration before starting the worker.

```powershell
$env:COLABCLUSTER_WORKER_TIMEOUT = "30"
```

The controller uses its own timezone-aware UTC receipt time for `last_seen`,
ignoring worker clock skew. Its application lifespan starts one background monitor
and cancels/awaits it on shutdown. The monitor checks every `min(5, timeout / 2)`
seconds. A record is removed when its age is **strictly greater** than the timeout, so
cleanup may take up to one additional monitor interval. `GET /workers` represents
the **current active worker set**; expired workers are deleted from memory, not
retained as offline records. With no active workers it returns `{"workers": []}`.
A future version may add persistent worker history; it is outside this scope.

Heartbeats preserve the current status when omitted by older clients. The runtime
sends its operational `Worker.status` (initially `READY`), so `BUSY` remains `BUSY`.
An explicit valid status updates an existing registry entry. Once a worker has
expired, heartbeats return 404; restart the worker to register a fresh entry,
including when reusing the same worker ID. There is no automatic re-registration. The worker does not infer its state from
controller responses or automatically reset `BUSY` to `READY`.

GPU metrics are best-effort: utilization percent and global used VRAM in GiB for
CUDA device 0. Missing PyTorch/CUDA/NVML support does not prevent heartbeats;
unavailable metrics are null and replace older metric values. Collection uses
[PyTorch CUDA APIs](https://docs.pytorch.org/docs/stable/cuda.html); no GPU or new
dependency is required. This is health reporting, not task scheduling.

Ctrl+C or SIGTERM attempts a bounded unregister request. If cleanup fails, or the
worker crashes/is forcibly terminated, timeout monitoring still removes it from the active registry.
On Windows, forced process termination does not execute Python cleanup. Normal
successful unregister removes the record, so use an abrupt stop for the offline test.
The notebook includes an explicit crash-test option that skips unregister.

For the offline test, stop/kill the worker without unregistering and query `/workers`
after the timeout plus one monitor interval (over 35 seconds with defaults).
Expect the stopped worker to disappear (and `{"workers": []}` if it was the only
worker). Restart the worker to register again; it should reappear as `ready` with
fresh timestamps. For manual Colab cells, run registration again before resuming
heartbeats. Restarting the controller during the timeout test clears the registry
and invalidates that test.

### Active registry cleanup and controller restart

`GET /workers` removes expired or explicitly offline entries under the registry
lock before returning its active snapshot. The background monitor also physically
deletes these entries; there is no historical list or persistent storage.

After updating controller code, stop the old controller with Ctrl+C and restart
it from this project's virtual environment. A running process without `--reload`
continues using its previously imported code. A fresh process starts with an empty
registry, so Colab must register again before sending heartbeats. Timeout defaults
remain 30 seconds and heartbeat defaults remain 10 seconds.

## Multi-Worker Cluster

Multiple Colab workers can register simultaneously with unique IDs. Each worker
has independent heartbeat timestamps, status, metadata, and GPU metrics. An
existing active ID returns HTTP 409 and is never overwritten. Keep running one
controller process: multiple server processes would have separate in-memory registries.

`GET /workers` returns active workers sorted lexicographically by `worker_id`.
`GET /cluster` uses one consistent active snapshot and returns `total_workers`,
`ready_workers`, `busy_workers`, `total_gpu_memory`, `gpus` (counts by GPU name),
and the same sorted full `WorkerInfo` records. Expired and offline workers are
physically removed before either response. A timeout removes only that worker;
healthy workers keep their own timestamps and remain listed.

Memory is measured in **GiB**. Two Tesla T4 workers reporting 14.56 GiB each
produce 29.12 GiB aggregate capacity across **two separate GPUs**, not one unified
29.12 GiB GPU. CPU-only workers (`gpu="None"`) count toward worker/status totals
but not GPU model counts or GPU capacity. ERROR workers with recent heartbeats
remain live and count toward total_workers, but not ready_workers or busy_workers.

Restart the controller in your own terminal after updating the source. Existing
in-memory registrations are cleared, so start workers afterward. In two separate
Colab runtimes, set distinct IDs (`COLAB-01` and `COLAB-02`) and the same current
controller URL, then run the worker entry point or your registration/heartbeat
cells. Do not use the same ID in both runtimes.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/workers | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:8000/cluster | ConvertTo-Json -Depth 6
```

With both workers READY and reporting 14.56 GiB, expect total_workers=2,
ready_workers=2, busy_workers=0, total_gpu_memory=29.12, and gpus={"Tesla T4": 2}.
Exact capacity comes from hardware detection. Stop heartbeats on only COLAB-02,
keeping COLAB-01 heartbeats running. After the 30-second timeout (up to five more
seconds for the monitor), both endpoints should contain only COLAB-01, and
/cluster should report total_workers=1. Restart COLAB-02 with a new registration
to restore two workers. An empty cluster returns zero totals, gpus={}, workers=[].

This step adds no scheduling, task execution, distributed inference, or persistence.

## Remote GPU Execution Test

Step 6.6 adds one fixed CUDA matrix multiplication diagnostic. The Windows
controller and `testing/gpu_test.py` remain CPU-only and never import torch.
Computation takes place on the selected Colab GPU. This is not a scheduler,
general task system, queue, or distributed workload.

There are two directions of HTTP traffic:

- Colab -> PC controller URL: registration and heartbeats (existing tunnel).
- PC controller -> Colab worker URL: the fixed `/gpu-test` endpoint (new manual
  reachable endpoint forwarding to Colab's `127.0.0.1:8001`). The PC tunnel cannot
  forward requests to Colab by itself.

Diagnostics are opt-in. Existing workers continue to register and heartbeat
without a diagnostic server. Set `COLABCLUSTER_WORKER_URL` to enable one. The
runtime starts its small FastAPI diagnostic endpoint in a background thread,
registers the URL in `metadata.diagnostic_url`, and continues heartbeats on the
main thread. Concurrent diagnostic requests receive 409; they are never queued.
Worker state is BUSY during execution and restored afterward.

### Run on a real Colab T4

1. Restart your PC controller from `colabcluster` after updating the code:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn controller.main:app --host 0.0.0.0 --port 8000
   ```

2. Keep the PC-to-controller tunnel running as before. Upload/sync the updated
   code to Colab and choose a GPU runtime. Stop the previous worker before
   starting the updated worker. Use a new worker ID or wait for expiry.

3. In Colab, manually provide a tunnel/reachable endpoint forwarding to local
   port 8001. If `cloudflared` is already installed in that runtime, this Python
   cell starts it without blocking later notebook cells:

   ```python
   import subprocess
   tunnel_log = open("/content/worker-tunnel.log", "w")
   worker_tunnel = subprocess.Popen(
       ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8001"],
       stdout=tunnel_log, stderr=subprocess.STDOUT,
   )
   ```

   Read `/content/worker-tunnel.log` after it prints the public URL. Supply your
   own installed tunnel tool if cloudflared is unavailable; the application does
   not install a provider, open a tunnel, or invent a public worker URL.

4. In the Colab notebook, with the project directory as the working directory:

   ```python
   import os, sys, subprocess
   os.environ["COLABCLUSTER_CONTROLLER_URL"] = "https://YOUR-PC-TUNNEL"
   os.environ["COLABCLUSTER_WORKER_URL"] = "https://YOUR-COLAB-WORKER-TUNNEL"
   os.environ["COLABCLUSTER_WORKER_ID"] = "COLAB-GPU-TEST"
   subprocess.run([sys.executable, "-u", "-m", "worker.colab_worker"], check=True)
   ```

   Keep this cell running. `COLABCLUSTER_WORKER_PORT` optionally changes the
   worker's local port (default 8001); forward the same port in your tunnel.
   Existing manual registration/heartbeat-only cells do not serve this endpoint.

5. From the Windows terminal, check `/workers` for a live T4 with
   `metadata.diagnostic_url`, then run:

   ```powershell
   .\.venv\Scripts\python.exe testing/gpu_test.py --worker-id COLAB-GPU-TEST
   ```

   Without `--worker-id`, the script requires exactly one active CUDA worker
   advertising a diagnostic URL; otherwise it asks you to specify the ID.
   To start small, append `--matrix-size 256 --iterations 2`.

`POST /workers/{worker_id}/gpu-test` accepts defaults of matrix size 4096 and 20
iterations, bounded to 256..8192 and 1..100. It returns the actual result shape,
worker/GPU/CUDA/PyTorch identity, total and mean timing, and allocated memory in
GiB (the response field is named `gpu_memory_allocated_gb`). Float32 matrix
multiplication uses CUDA device 0, three warmups, and synchronization before and
after timing. Tensor references are released on success and failure. PyTorch may
retain freed blocks in its allocator cache; reported allocated memory is sampled
while the matrices are live and may include other allocations in the process.

Missing/expired workers return 404; missing diagnostic URLs or busy workers return
409. CUDA/PyTorch/worker execution failures return 503, unreachable or invalid
worker replies return 502, and request timeouts return 504. A timed-out request
may still be computing on the worker; it is not automatically retried. Registration
and heartbeat timeouts remain unchanged.

Both development endpoints are unauthenticated. Only enable this diagnostic with
trusted workers and controlled temporary endpoints. Close the worker tunnel when
finished. No GPU benchmark is executed on Windows during local tests.

## Repository layout and upload bundle

This directory is the single source of truth. `controller/`, `worker/`, and
`common/` hold runtime code; `tests/` holds automated tests; `testing/` contains
PC-side diagnostics. `examples/` holds the Colab notebook and the preserved
`manual_cuda_exercise.py` standalone CUDA experiment (Colab/GPU only).
The project license is `LICENSE`; it is not duplicated under tests.

Generate the one current upload archive from this directory:

```powershell
.\.venv\Scripts\python.exe scripts/build_colab_bundle.py
```

Upload `artifacts/colabcluster-worker.zip` to Colab. This generated distribution
contains copies of runtime source for transfer; edit the source directories,
never the archive. Old version-named bundles are no longer maintained.

## GPU Dashboard

Open **http://127.0.0.1:8000/dashboard** after starting the current controller:

```powershell
cd "C:\Users\Mukil\New folder (12)\colabcluster"
.\.venv\Scripts\python.exe -m uvicorn controller.main:app --host 0.0.0.0 --port 8000
```

Stop any old controller before restarting; do not launch a duplicate on port 8000.
The plain HTML/CSS/JavaScript dashboard is served by the same FastAPI application.
It polls `/workers` and `/cluster` every 2.5 seconds and shows active workers,
ready/busy counts, GPU capacity/types, heartbeat times, and reported telemetry.
Missing metrics/versions display as unavailable, not zero or guessed values.
On connection loss, cached cards are labeled stale and execution is disabled.
Expired workers disappear on the next successful poll; this is not a history view.
CUDA and PyTorch card versions come from existing metadata when available.

**Run NN Test** triggers one predefined SmallMLP diagnostic on the selected remote
worker. It does not accept code, model definitions, files, or training parameters.
The model is `784 -> 128 -> ReLU -> 64 -> ReLU -> 10`, batch size 128, float32,
three warmups and 100 timed inference passes with synthetic inputs. It requires
CUDA and never silently falls back to CPU. No dataset downloads, training,
persistent models, scheduling, or repeated background tests are performed.

The controller forwards `POST /workers/{worker_id}/nn-test` to `/nn-test` on the
existing worker diagnostic server. Both NN and matrix diagnostics share the same
worker lock, so overlapping requests return 409 rather than being queued.
`/gpu-test` remains unchanged. The controller exposes BUSY during the request,
and the worker sets BUSY while executing; status is restored on completion or
failure. Existing heartbeats keep running without protocol changes. A short test
may finish between polls, so the clicked card also shows a local RUNNING state.
Results are only kept in the current page; refresh clears them.

CUDA synchronization brackets compute timing; reported timing excludes Cloudflare
and HTTP latency. `peak_memory_mb` is in MiB (1024 squared bytes), measured using
PyTorch's per-process/device peak allocated memory after resetting peak counters,
and can include other live allocations in that worker process. Model/input/output
references are released afterward; PyTorch may retain reusable allocator cache.
Failures appear clearly in the dashboard. A network timeout does not cancel CUDA
computation already in progress, and the diagnostic is not automatically retried.

### Manual Colab NN verification

1. Build and upload the current `artifacts/colabcluster-worker.zip` to Colab:
   `.\.venv\Scripts\python.exe scripts/build_colab_bundle.py`.
2. Start/restart the controller and its existing Cloudflare tunnel.
3. Configure/start the Colab worker's existing tunnel forwarding to port 8001,
   then run the updated Colab worker with both existing URL variables set:
   `COLABCLUSTER_CONTROLLER_URL` (PC tunnel) and `COLABCLUSTER_WORKER_URL`
   (Colab diagnostic tunnel). This uses the same setup as the matrix GPU test.
   Restarting an old worker process is necessary to load the new `/nn-test` route.
4. Confirm `/workers` shows your ready T4 with `metadata.diagnostic_url`.
5. Open **http://127.0.0.1:8000/dashboard** and click **Run NN Test** on that worker.
6. Verify **NN TEST PASSED**, Tesla T4, `device: cuda:0`, `SmallMLP`, input
   `[128, 784]`, output `[128, 10]`, 100 passes, and nonzero timing measurements.
7. Confirm the worker returns to READY. To run again, click again explicitly.

The exact equivalent manual request from Windows PowerShell is:

```powershell
$workerId = "COLAB-GPU-TEST"  # Replace with the ID shown by /workers.
Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/workers/$workerId/nn-test" `
    -ContentType "application/json" -Body '{}' -TimeoutSec 135
```

No dashboard tooling or PyTorch is required on the controller for inference.
Automated tests mock CUDA execution; the real T4 test is a manual integration
check and its results must not be inferred from mocked tests. The existing
development API/tunnel access model is unchanged; there is no new authentication.

## Small CNN Remote Inference Test

The fixed **SmallCNN** diagnostic runs inference on the selected remote CUDA
worker. It uses synthetic float32 input `[32, 3, 32, 32]`, with no dataset download,
training, optimizer, arbitrary code, or persistent model files. The architecture is:

```text
Conv2d(3,16,3,padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2)
AdaptiveAvgPool2d((1,1)) -> Flatten -> Linear(32,10)
```

The result shape is `[32, 10]`. Model initialization uses a fixed seed with CPU
random state restored afterward; synthetic input uses a private seeded CUDA
generator. This makes inputs reproducible without reseeding unrelated worker
random streams. GPU timing is not expected to be identical between runs.

Inference uses `model.eval()` and `torch.no_grad()`, ten warmup passes, then 100
timed passes. CUDA synchronization brackets the timed section, which excludes
model construction, warmup, HTTP, and Cloudflare latency. Throughput is
`32 * 100 / elapsed_seconds` images/sec. Peak allocated VRAM is sampled after
resetting peak statistics just before timed inference. `peak_memory_mb` is in
MiB (1024 squared bytes), and includes other live allocations in this process.
Tensor/model references are released afterward; PyTorch may retain allocator cache.

The existing diagnostic server on port 8001 now serves `/gpu-test`, `/nn-test`,
and `/cnn-test`. All three use the same worker instance and lock. The CNN endpoint
sets BUSY during execution and restores the previous state in `finally`, even on
failure. There is no new port, server, tunnel, scheduler, or queue. The matrix and
SmallMLP operations retain their existing behavior. CUDA/PyTorch absence returns
an explicit error instead of silently running CPU inference.

### Manual real T4 check

1. Build `artifacts/colabcluster-worker.zip` with
   `.\.venv\Scripts\python.exe scripts/build_colab_bundle.py`.
2. Upload/extract that bundle in Colab. Restart the updated controller and worker,
   keeping the existing controller/worker Cloudflare configuration. Re-register
   the worker after a controller restart; verify `/workers` shows it as ready.
3. Run this on Windows, replacing the ID if necessary:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/workers/COLAB-GPU-TEST/cnn-test" `
  -ContentType "application/json" `
  -Body '{}' `
  -TimeoutSec 135
```

Alternatively open **http://127.0.0.1:8000/dashboard**, find `COLAB-GPU-TEST`, and
click **Run CNN Test**. The card shows BUSY and CNN TEST RUNNING; completion shows
CNN TEST PASSED with GPU, CUDA/PyTorch versions, device, model, shapes, batch,
100 passes, GPU time, mean inference latency, throughput, and peak VRAM.
Verify Tesla T4, `cuda:0`, `SmallCNN`, output `[32, 10]`, and `status: passed`.
The worker should return to READY. Each click runs only one diagnostic.

The request body must be `{}`; custom architectures/parameters are rejected.
Errors use the existing diagnostic conventions: 404 inactive worker, 409 not
ready/missing URL/concurrent execution, 503 worker CUDA failure, 502 forwarding
or invalid reply, and 504 timeout. Timeouts do not cancel CUDA work already
running and requests are never automatically retried.

Automated tests mock GPU operations and verify routing, shape, inference mode,
status cleanup, and concurrency. They do **not** prove real T4 execution; perform
the manual integration check above for that confirmation. This is a diagnostic,
not a training benchmark.

## Two independent Colab workers

The existing registry and dashboard support this setup without a scheduler:

```text
Controller (Windows, port 8000)
   |-- Colab Worker A (its own runtime and diagnostic tunnel)
   `-- Colab Worker B (a second runtime and diagnostic tunnel)
```

Each runtime needs a unique worker ID, its own diagnostic server, its own
reachable `COLABCLUSTER_WORKER_URL`, and a continuously running heartbeat loop.
Both use the same reachable controller URL. Keep the existing controller and
its tunnel running; do not start a second controller on port 8000.

After uploading/extracting the worker bundle and completing the existing Colab
setup in each runtime, configure the environment in a Python cell. Replace URL
placeholders with the actual tunnel URLs before running:

```python
import os
os.environ["COLABCLUSTER_CONTROLLER_URL"] = "https://<controller-tunnel>.trycloudflare.com"
os.environ["COLABCLUSTER_WORKER_ID"] = "COLAB-GPU-TEST"  # Worker A
os.environ["COLABCLUSTER_WORKER_URL"] = "https://<worker-a-tunnel>.trycloudflare.com"
```

In the **second Colab runtime**, use the same controller URL but these values:

```python
os.environ["COLABCLUSTER_WORKER_ID"] = "COLAB-GPU-TEST-B"
os.environ["COLABCLUSTER_WORKER_URL"] = "https://<worker-b-tunnel>.trycloudflare.com"
```

Each worker tunnel must forward to `http://127.0.0.1:8001` in its own runtime.
Separate runtimes can both use local port 8001. From the extracted project
directory in **each Colab runtime**, run and leave this cell running:

```python
!python -u -m worker.colab_worker
```

That entry point creates `create_diagnostic_app(worker)` automatically when
`COLABCLUSTER_WORKER_URL` is set, registers the worker, and sends heartbeats.
Do not launch `uvicorn worker.diagnostics:app`; there is no module-level app.
Running the worker in Windows PowerShell registers the Windows machine, not
the Colab GPU. Restart a worker to apply changed environment variables; after
a controller restart, restart both workers to register them again.

Check both records from Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/workers | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:8000/cluster | ConvertTo-Json -Depth 6
```

Verify two distinct IDs, Tesla T4/CUDA metadata, different diagnostic URLs,
and advancing `last_seen` values for both. Open
`http://127.0.0.1:8000/dashboard` to see a card for each worker. Manually run a
diagnostic on A, then B; confirm each result identifies the selected worker
and each returns to READY. The dashboard allows one diagnostic request at a
time. Workers remain separate GPUs; reported aggregate memory is not pooled.
Stopping one worker removes only its record, immediately on clean shutdown or
after the heartbeat timeout on loss of connection.

Automated two-worker integration tests use independent in-process diagnostic
apps with simulated GPU results. They verify routing, metadata, heartbeat,
status and removal isolation. They do not establish real two-T4 execution;
that requires two actual Colab runtimes and the manual checks above.


## Two-Worker Parallel Inference

`POST /inference/two-worker` with `{}` runs one fixed 64-sample synthetic
SmallCNN workload on `COLAB-GPU-TEST` and `COLAB-GPU-TEST-2`. Both must be
active, READY Tesla T4 CUDA workers with separate diagnostic URLs. Optional
`worker_ids` must contain exactly those IDs in that order; other workloads,
IDs, tensors and executable code are rejected.

The controller atomically reserves both workers and concurrently sends
`{"partition": 0}` and `{"partition": 1}` to their new `POST /inference`
endpoints. Each generates 32 float32 samples `[32,3,32,32]` locally, using
private CUDA seeds 1000 and 1001 respectively. The existing SmallCNN builder
uses seed 0 for identical initial weights, restoring the CPU RNG afterward.
These are untrained synthetic diagnostics, not meaningful classifications.
No tensors cross the network. Ten warmups precede one timed forward per worker;
warmups are excluded from GPU time and sample counts. Both outputs are `[32,10]`.
The response collects validated per-worker metrics and shapes, not prediction tensors.

`parallel_wall_time_ms` spans concurrent dispatch through both completed
responses, including HTTP/tunnels, model construction, warmup, and controller
overhead. `sum_worker_gpu_time_ms` sums the synchronized timed forwards.
`effective_throughput_images_per_second` is **64 / wall seconds**. Per-worker
throughput is **32 / GPU seconds**. These are different measurements; no speedup
is claimed without a comparable sequential baseline.

Each worker has its own GPU and VRAM; they are not merged into one physical
GPU. This experiment is a precursor to broader execution/scheduling, but adds
no scheduler, queue, training, DDP, parameter synchronization or worker-to-worker
networking. Existing matrix, MLP and CNN diagnostic APIs are unchanged.

### Run the real experiment manually

1. Upload/extract the refreshed `artifacts/colabcluster-worker.zip` in **both**
   Colab runtimes. Stop old worker processes normally before replacing files.
2. Restart the controller with the updated code (stop its existing process first,
   so port 8000 is free). Keep the existing tunnel configuration.
3. Start each updated worker with its existing distinct ID and diagnostic URL:
   `python -u -m worker.colab_worker`. Both must re-register after controller restart.
4. Verify both workers are READY in `/workers`. Open `/dashboard` and click
   **Run 2-Worker Inference**, or run this in Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/inference/two-worker" `
  -ContentType "application/json" -Body '{}' -TimeoutSec 145 |
  ConvertTo-Json -Depth 8
```

Expect status `passed`, two correctly identified Tesla T4 results on `cuda:0`,
partitions 0 and 1, total_samples 64, and output_shapes `[[32,10],[32,10]]`.
Then verify both workers return to READY. Do not interpret mocked tests as real
GPU execution; this new experiment still requires this manual two-T4 check.

Missing/expired workers return 404; busy workers or invalid hardware/URLs return
409 without queuing. Failed/malformed replies return 502 (worker CUDA failures
503), and transport timeouts return 504. Failure of either worker fails the
whole experiment. Both bounded requests finish before controller reservations
are released; deleted/replaced registrations are never resurrected. Heartbeats
continue during execution. A timeout does not cancel remote CUDA work; the
worker's shared diagnostic lock prevents overlapping local diagnostics until
that work ends. There are no automatic execution retries.


## Single-worker inference baseline

`POST /inference/single-worker` with `{}` processes **64 samples on one READY
worker**, providing a controlled baseline for the existing two-worker experiment
(**32 + 32 samples**). There is no worker-selection or sample-count parameter.
The registry selects and reserves the first active READY worker in worker-ID
order, regardless of registration order. Selection and reservation are atomic.
The same Tesla T4/CUDA/diagnostic-URL checks apply; an unsuitable first READY
worker causes an error, rather than silently selecting another GPU.

The controller sends one empty request to the selected worker's
`POST /inference/single-worker`. The worker reuses the existing SmallCNN builder
and inference loop, float32 on `cuda:0`, seed-0 model initialization, no gradients,
ten warmups and one timed forward. It locally generates the same two 32-sample
partitions (CUDA seeds 1000 and 1001), concatenating them into `[64,3,32,32]`
before warmup. Output is `[64,10]`. No tensors are transferred over HTTP.
The two-worker API, seeds, per-worker batch size and responses remain unchanged.

`wall_time_ms` is measured immediately around the controller's blocking HTTP
request, stopping when the response body arrives, before JSON/schema validation.
It includes network/tunnel latency, remote model/input construction, warmup,
inference and response transfer. GPU-only `total_gpu_time_ms` brackets one
CUDA-synchronized forward, excluding setup and warmup. Effective end-to-end
throughput is **64 / (wall_time_ms / 1000)**; worker GPU throughput is reported
separately. The existing parallel wall metric additionally includes thread-pool
and result-validation overhead, so small timing differences must not be treated
as proof of compute speedup. No speedup or two-GPU advantage is claimed yet.

Re-upload the rebuilt worker bundle and restart the updated controller and
workers before manually using this endpoint. Controller restart requires worker
re-registration. Keep the current tunnel/ID configuration. The dashboard now
has **Run 1-Worker Inference** beside **Run 2-Worker Inference**, with selected
worker, GPU, sample count, wall time, GPU time and effective throughput.

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/inference/single-worker" `
  -ContentType "application/json" -Body '{}' -TimeoutSec 145 |
  ConvertTo-Json -Depth 8
```

No READY worker returns 409. Remote errors follow the existing 409/502/503/504
conventions. Selected workers transition READY -> BUSY -> READY, including
failure cleanup; removed/replaced records are not recreated. Incoming READY
heartbeats cannot erase an active reservation. The existing worker diagnostic
lock prevents overlapping local work. As before, HTTP timeout does not cancel
remote CUDA work. No retries, queues, scheduler or training were added.
Automated tests use mocked GPU/network execution; the real baseline and its
10-run benchmark have not been executed as part of this implementation.


## Step 8.2: 10-run single-worker benchmark

From the project directory, with the controller and updated Colab worker running:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_single_worker.py
```

Options: `--controller-url http://127.0.0.1:8000 --runs 10 --timeout 135`.
The runner issues sequential POSTs with `{}`, no automatic retries, to
`/inference/single-worker`. Each request uses the existing fixed 64-sample
SmallCNN workload. The controller selects the worker; the CSV records its actual
identity for every response. A requests timeout bounds connection/read waits,
not an absolute deadline for a response that keeps delivering bytes.

Results go to `results/single_worker_benchmark.csv` relative to the project root,
regardless of the launching directory. Each attempt is flushed immediately,
including failures, HTTP status and error details. Rerunning replaces this
single-worker CSV; preserve it elsewhere first if needed. Two-worker results
are untouched. A nonzero exit code means at least one attempt failed.

Client wall time uses `perf_counter()` around the complete HTTP request/body
receipt, before JSON parsing, and is separate from server `wall_time_ms` and
GPU inference time. The summary reports success/failure counts and mean, median,
sample standard deviation, minimum and maximum client time, plus mean GPU time
and effective throughput. Statistics include only validated successful runs;
missing statistics print `null` (including sample deviation with fewer than two
successes). Invalid/missing required measurements fail validation while retaining
available fields. Failures remain in the CSV and are never silently discarded.
This step collects a baseline only; it does not calculate two-worker speedup.


## Step 8.3: 10-run two-worker benchmark

With the controller and both READY workers running, execute from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_two_worker.py
```

Options match the single-worker runner: `--controller-url http://127.0.0.1:8000
--runs 10 --timeout 135`. Ten sequential requests send `{}` to
`/inference/two-worker`; parallel GPU dispatch remains internal to that endpoint.
The benchmark shares the single-worker HTTP timing, failure handling and CSV
writer. Client wall time ends when the complete HTTP response arrives, before
JSON parsing. Server `parallel_wall_time_ms` is saved separately as
`server_wall_time_ms`.

`results/two_worker_benchmark.csv` records every attempt, HTTP status/error,
worker count, actual IDs, GPU models, 64 total samples, two output shapes,
combined GPU time and effective throughput. Success requires two distinct
workers each reporting passed CUDA execution of 32 samples with output `[32,10]`.
Combined GPU time uses `sum_worker_gpu_time_ms`, falling back to the sum of both
worker times; unavailable GPU times are blank. Summary GPU-time availability
is explicitly counted. All statistics exclude failed attempts, which remain
in the CSV; standard deviation is the sample statistic, or null with fewer than
two successes. The existing single-worker CSV is never touched. Rerunning this
command replaces only the two-worker CSV; archive it first to retain prior runs.
This step does not calculate or claim speedup.


## Step 8.4: Compare saved inference benchmarks

```powershell
.\.venv\Scripts\python.exe scripts\compare_inference_benchmarks.py
```

Reads `results/single_worker_benchmark.csv` and `results/two_worker_benchmark.csv`
without changing them or issuing live requests. Writes
`results/benchmark_comparison.md`. Optional `--single`, `--two`, and `--output`
paths support other saved files; output cannot be either input file.

The primary comparison uses mean **client** wall time: speedup is single/two,
wall-time change is (two-single)/single, and conventional efficiency is speedup/2.
The report also includes sample deviation, server timings, optional GPU timings,
and mean API-reported throughput. Two-worker GPU timing is the sum across GPUs,
not elapsed wall time. Failed/incomplete rows are explicitly listed and excluded
from statistics; successful outliers are retained. Missing required columns
produce an error; unavailable statistics display N/A.

Within +/-5% mean wall-time change is labelled approximately equivalent as a
descriptive convention, not a significance test. The report separates measured
values from interpretation, covers the fixed 64-sample versus 32/32 workload and
HTTP overhead, and explains the limits of these small, separately collected runs.
It does not establish a general distributed-inference speedup.


## Step 8.5: Workload scaling

New controller endpoint `POST /inference/scaling` accepts only:

```json
{"total_samples": 1024, "worker_count": 2}
```

Supported totals: **64, 256, 1024, 4096**. Counts: **1 or 2**. Existing fixed
single/two-worker endpoints and their default 64-sample behavior are unchanged.
The worker endpoint at the same path additionally receives a partition index.
Single-worker selection remains first READY by ID; two-worker selection remains
`COLAB-GPU-TEST` and `COLAB-GPU-TEST-2`. Both must be Tesla T4 CUDA workers.

The shared SmallCNN loop retains seed-0 model initialization, float32 CUDA,
no gradients, ten warmups and one timed forward. Synthetic input uses global
32-sample blocks seeded 1000 + block index: one worker processes all blocks;
two workers process disjoint contiguous halves of exactly the same workload.
Output shapes and per-worker sample assignments are validated, including that
summed output samples equal the requested total. All supported totals divide
evenly; the partition helper assigns any remainder to the first worker and is
unit-tested on odd totals. Unsupported totals still return 422.

Before live measurement, rebuild/upload `artifacts/colabcluster-worker.zip` in
both Colab runtimes and restart their worker processes, preserving IDs and tunnel
URLs. Restart the controller with updated source; workers must re-register after
controller restart. Do not start a second process on occupied port 8000.

```powershell
.\.venv\Scripts\python.exe scripts/build_colab_bundle.py
.\.venv\Scripts\python.exe scripts/benchmark_scaling.py --trials 5
```

Options: `--controller-url http://127.0.0.1:8000`, `--timeout 135`, and
`--trials N` (minimum 5). GET-only preflight checks the running controller's
scaling endpoint, both READY workers, their hardware, diagnostic connectivity
and worker scaling endpoints before inference starts. Stale source/offline
workers produce a recorded preflight error, with no inference requests.

For every size, one/two-worker requests alternate, with alternating order per
round, targeting at least five successes each. Attempts stop after twice the
requested target per configuration if errors persist; every failed attempt
is recorded, and unmet targets produce an incomplete report and nonzero exit.
There is no retry of a request inside the HTTP client. A timeout does not cancel
remote work; subsequent attempts may receive busy errors. Successes do not hide
failed attempts. Each run includes a unique timestamp/ID directory under
`results/scaling/`, with `measurements.csv`, `metadata.json`, and its own report.
The latest report is also written to `results/scaling/scaling_report.md`.
Previous run directories and all existing baseline CSVs remain untouched.

Both configurations use identical client `perf_counter` HTTP timing. The shared
scaling server path includes executor/HTTP/setup/warmup/validation overhead for
both counts. CSVs record requested total, per-worker assignments, IDs, GPU models,
output shapes, client/server wall time, summed/per-worker GPU time, reported
throughput, status, and errors. Reports compare successful means, medians and
sample deviations only after both configurations meet the target; no outliers
are removed. GPU time is summed compute, not parallel wall time; throughput is
API-reported samples/server-wall-seconds. Small samples, ascending workload order,
network latency and ephemeral Colab conditions limit generalization. No statistical
significance or performance improvement is claimed without measured evidence.
