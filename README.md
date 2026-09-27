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

Step 4 adds local worker initialization and hardware detection to the controller
and shared worker protocol.

## Controller

The ColabCluster controller is the control plane. It currently maintains an
in-memory worker registry and exposes health, registration, heartbeat, and
unregister APIs. Persistent storage and distributed scheduling are intentionally
not implemented yet. The local worker is not connected to the controller.
There is no Colab integration, tunneling, or task execution.

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
introduced in later stages. Google Colab is **not connected yet**.

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
UTC clock, independently of the worker's timestamp. Telemetry is validated but
not retained or used for scheduling at this stage.

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

Controller registration and communication from the worker will be implemented
in a later step. This runtime makes no HTTP requests and performs no heartbeat,
scheduling, task execution, or Google Colab integration.

