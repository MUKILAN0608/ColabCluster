"""Register one worker and send heartbeats: python -m worker.colab_worker."""

import sys
import os
import time
from threading import Thread
import logging
import signal
from threading import current_thread, main_thread

from worker.client import WorkerClient, WorkerClientError
from worker.heartbeat import get_heartbeat_interval, run_heartbeat_loop
from worker.worker import Worker, print_worker_info


def _stop_worker(signum: int, frame: object) -> None:
    """Let SIGTERM follow the same cleanup path as Ctrl+C."""
    raise KeyboardInterrupt


def main() -> int:
    """Register once, send heartbeats, and best-effort unregister on shutdown."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    diagnostic_server = None
    diagnostic_thread = None
    registered = False
    client = None
    worker = None
    previous_handler = None
    if current_thread() is main_thread():
        previous_handler = signal.signal(signal.SIGTERM, _stop_worker)
    try:
        interval = get_heartbeat_interval()
        worker = Worker()
        print_worker_info(worker, title="COLABCLUSTER COLAB WORKER")
        client = WorkerClient()
        if os.getenv("COLABCLUSTER_WORKER_URL", "").strip():
            import uvicorn
            from worker.diagnostics import create_diagnostic_app
            port = int(os.getenv("COLABCLUSTER_WORKER_PORT", "8001"))
            if not 1 <= port <= 65535:
                raise ValueError("COLABCLUSTER_WORKER_PORT must be between 1 and 65535")
            diagnostic_server = uvicorn.Server(uvicorn.Config(
                create_diagnostic_app(worker), host="127.0.0.1", port=port,
            ))
            diagnostic_thread = Thread(target=diagnostic_server.run, daemon=True)
            diagnostic_thread.start()
            deadline = time.monotonic() + 10
            while not diagnostic_server.started:
                if not diagnostic_thread.is_alive() or time.monotonic() > deadline:
                    raise ValueError("Worker diagnostic server failed to start; check its port")
                time.sleep(0.05)
        print(f"Connecting to: {client.controller_url}", flush=True)
        print("Registering worker...", flush=True)
        response = client.register(worker)
        registered = True
        print(response.model_dump_json(indent=2), flush=True)
        print(f"Registration successful! Starting heartbeat every {interval:g} seconds.", flush=True)
        run_heartbeat_loop(worker, client, interval)
    except KeyboardInterrupt:
        print("\nWorker stopped.", flush=True)
        return 0
    except (WorkerClientError, ValueError) as exc:
        print(f"Worker startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            if registered and client is not None and worker is not None:
                try:
                    client.unregister(worker.worker_id, reason="worker shutdown")
                    logging.getLogger(__name__).info("[Worker] Unregistered successfully")
                except (WorkerClientError, KeyboardInterrupt) as exc:
                    logging.getLogger(__name__).warning("[Worker] Unregister failed; liveness timeout will apply: %s", exc)
        finally:
            if diagnostic_server is not None:
                diagnostic_server.should_exit = True
            if diagnostic_thread is not None:
                diagnostic_thread.join(timeout=5)
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
