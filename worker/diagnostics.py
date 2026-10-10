"""Opt-in HTTP endpoint for one fixed GPU diagnostic, not arbitrary tasks."""

from threading import Lock
from contextlib import contextmanager
from common.scaling import ScalingBatchRequest, ScalingBatchResponse
from common.schemas import SingleWorkerRequest, SingleInferenceResponse
from common.schemas import InferenceBatchRequest, InferenceBatchResponse
from worker.inference import run_inference

from fastapi import FastAPI, HTTPException, Header

from common.schemas import GpuTestRequest, GpuTestResponse, WorkerStatus, NnTestRequest, NnTestResponse
from worker.gpu_test import GpuTestError, run_gpu_test
from worker.worker import Worker
from worker.nn_test import NnTestError, run_nn_test
from worker.cnn_test import CnnTestError, run_cnn_test
from common.schemas import CnnTestRequest, CnnTestResponse

EXECUTION_HISTORY_LIMIT = 4096


def create_diagnostic_app(worker: Worker) -> FastAPI:
    """Serve the fixed benchmark while the main thread sends heartbeats."""
    app = FastAPI(title="ColabCluster GPU diagnostic")
    lock = Lock()
    history_lock = Lock()
    executions = {}

    @app.get("/executions/{request_id}")
    def execution_status(request_id: str):
        with history_lock:
            state = executions.get(request_id, "unknown")
            available = not lock.locked() and worker.status == WorkerStatus.READY
        return {"worker_id": worker.worker_id, "request_id": request_id, "state": state,
                "slot_available": available}

    @contextmanager
    def execution(request_id):
        # Never evict IDs: an old duplicate must not become executable again.
        # The bounded ledger fails closed when full; restart an idle worker then.
        with history_lock:
            if request_id and request_id in executions:
                raise HTTPException(409, "Request ID already received; inspect /executions/{request_id}")
            if request_id and len(executions) >= EXECUTION_HISTORY_LIMIT:
                raise HTTPException(503, "Execution ledger full; restart the idle worker")
            if not lock.acquire(blocking=False):
                raise HTTPException(409, "A GPU diagnostic is already running")
            if request_id:
                executions[request_id] = "running"
        previous = worker.status
        state = "failed"
        try:
            if previous != WorkerStatus.READY:
                raise HTTPException(409, "Worker is not ready for a GPU diagnostic")
            worker.status = WorkerStatus.BUSY
            yield
            state = "completed"
        finally:
            worker.status = previous
            with history_lock:
                if request_id:
                    executions[request_id] = state
                lock.release()


    @app.post("/gpu-test", response_model=GpuTestResponse)
    def gpu_test(payload: GpuTestRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")) -> GpuTestResponse:
        try:
            with execution(request_id):
                return run_gpu_test(worker.worker_id, payload)
        except GpuTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    @app.post("/nn-test", response_model=NnTestResponse)
    def nn_test(payload: NnTestRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")) -> NnTestResponse:
        """Run only the predefined SmallMLP under the existing diagnostic lock."""
        try:
            with execution(request_id):
                return run_nn_test(worker.worker_id)
        except NnTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    @app.post("/cnn-test", response_model=CnnTestResponse)
    def cnn_test(payload: CnnTestRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")) -> CnnTestResponse:
        """Run only SmallCNN, sharing the matrix/MLP concurrency lock."""
        try:
            with execution(request_id):
                return run_cnn_test(worker.worker_id)
        except CnnTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    @app.post("/inference", response_model=InferenceBatchResponse)
    def inference(payload: InferenceBatchRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")) -> InferenceBatchResponse:
        try:
            with execution(request_id):
                return run_inference(worker.worker_id, payload.partition)
        except CnnTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    @app.post("/inference/single-worker", response_model=SingleInferenceResponse)
    def single_inference(payload: SingleWorkerRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")) -> SingleInferenceResponse:
        try:
            with execution(request_id):
                return run_inference(worker.worker_id, 0, single=True)
        except CnnTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    @app.post("/inference/scaling", response_model=ScalingBatchResponse)
    def scaling_inference(payload: ScalingBatchRequest, request_id: str | None = Header(default=None, alias="X-ColabCluster-Request-ID", pattern="^[a-f0-9]{32}$")):
        try:
            with execution(request_id):
                return run_inference(worker.worker_id, payload.partition, scaling=payload)
        except CnnTestError as exc:
            raise HTTPException(503, str(exc), headers={"X-ColabCluster-Request-ID": request_id or "",
                "X-ColabCluster-Execution-State": "failed"}) from exc


    return app
