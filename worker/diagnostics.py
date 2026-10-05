"""Opt-in HTTP endpoint for one fixed GPU diagnostic, not arbitrary tasks."""

from threading import Lock

from fastapi import FastAPI, HTTPException

from common.schemas import GpuTestRequest, GpuTestResponse, WorkerStatus, NnTestRequest, NnTestResponse
from worker.gpu_test import GpuTestError, run_gpu_test
from worker.worker import Worker
from worker.nn_test import NnTestError, run_nn_test
from worker.cnn_test import CnnTestError, run_cnn_test
from common.schemas import CnnTestRequest, CnnTestResponse


def create_diagnostic_app(worker: Worker) -> FastAPI:
    """Serve the fixed benchmark while the main thread sends heartbeats."""
    app = FastAPI(title="ColabCluster GPU diagnostic")
    lock = Lock()

    @app.post("/gpu-test", response_model=GpuTestResponse)
    def gpu_test(payload: GpuTestRequest) -> GpuTestResponse:
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A GPU diagnostic is already running")
        previous = worker.status
        try:
            if previous != WorkerStatus.READY:
                raise HTTPException(409, "Worker is not ready for a GPU diagnostic")
            worker.status = WorkerStatus.BUSY
            return run_gpu_test(worker.worker_id, payload)
        except GpuTestError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            worker.status = previous
            lock.release()

    @app.post("/nn-test", response_model=NnTestResponse)
    def nn_test(payload: NnTestRequest) -> NnTestResponse:
        """Run only the predefined SmallMLP under the existing diagnostic lock."""
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A GPU diagnostic is already running")
        previous = worker.status
        try:
            if previous != WorkerStatus.READY:
                raise HTTPException(409, "Worker is not ready for a GPU diagnostic")
            worker.status = WorkerStatus.BUSY
            return run_nn_test(worker.worker_id)
        except NnTestError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            worker.status = previous
            lock.release()

    @app.post("/cnn-test", response_model=CnnTestResponse)
    def cnn_test(payload: CnnTestRequest) -> CnnTestResponse:
        """Run only SmallCNN, sharing the matrix/MLP concurrency lock."""
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A GPU diagnostic is already running")
        previous = worker.status
        try:
            if previous != WorkerStatus.READY:
                raise HTTPException(409, "Worker is not ready for a GPU diagnostic")
            worker.status = WorkerStatus.BUSY
            return run_cnn_test(worker.worker_id)
        except CnnTestError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            worker.status = previous
            lock.release()

    return app
