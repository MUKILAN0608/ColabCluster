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

Step 2 implements the local controller only.

## Controller

The ColabCluster controller is the control plane. It currently maintains an
in-memory worker registry and exposes health, registration, heartbeat, and
unregister APIs. Persistent storage and distributed scheduling are intentionally
not implemented yet. There is no Colab worker or connection, tunneling, or task
execution.

From the `colabcluster` directory, use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
uvicorn controller.main:app --reload
```

The controller listens at http://127.0.0.1:8000. Interactive Swagger documentation
is available at http://127.0.0.1:8000/docs.

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

Heartbeat and unregister accept `{"worker_id": "manual-test"}`. Required strings
must be nonempty, GPU memory must be finite and nonnegative, and invalid requests
return 422. Registration and heartbeat return a `message` and the `worker` record,
including timezone-aware UTC `registered_at` and `last_seen` timestamps. Status is
caller-supplied metadata; heartbeat does not automatically change it.

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

