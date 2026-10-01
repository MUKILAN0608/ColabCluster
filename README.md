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
