"""Validated HTTP routes for the local controller."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from common.schemas import (
    ClusterSummary, HealthResponse, WorkersResponse, WorkerRegistration, WorkerRegistrationResponse,
    WorkerHeartbeat, WorkerHeartbeatResponse, WorkerUnregister, WorkerUnregisterResponse,
)
from common.schemas import GpuTestRequest, GpuTestResponse
from controller.gpu_test import forward_gpu_test
from controller.registry import DuplicateWorkerError, UnknownWorkerError, WorkerRegistry

router = APIRouter()


def get_registry(request: Request) -> WorkerRegistry:
    """Resolve the registry belonging to this application."""
    return request.app.state.registry


Registry = Annotated[WorkerRegistry, Depends(get_registry)]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report controller health."""
    return HealthResponse()


@router.get("/workers", response_model=WorkersResponse)
def list_workers(request: Request, registry: Registry) -> WorkersResponse:
    """List all workers currently in memory."""
    return WorkersResponse(workers=registry.get_workers(timeout=request.app.state.worker_timeout))


@router.post("/workers/register", response_model=WorkerRegistrationResponse, status_code=201)
def register_worker(payload: WorkerRegistration, registry: Registry) -> WorkerRegistrationResponse:
    """Register a worker with a unique ID."""
    try:
        worker = registry.register_worker(payload)
    except DuplicateWorkerError:
        raise HTTPException(409, f"Worker '{payload.worker_id}' is already registered") from None
    return WorkerRegistrationResponse(
        success=True, worker_id=worker.worker_id, message="Worker registered", worker=worker
    )


@router.post("/workers/heartbeat", response_model=WorkerHeartbeatResponse)
def heartbeat(payload: WorkerHeartbeat, registry: Registry) -> WorkerHeartbeatResponse:
    """Update the last-seen time of an existing worker."""
    try:
        status = payload.status if "status" in payload.model_fields_set else None
        worker = registry.update_heartbeat(
            payload.worker_id, status=status, gpu_utilization=payload.gpu_utilization,
            gpu_memory_used=payload.gpu_memory_used,
        )
    except UnknownWorkerError:
        raise HTTPException(404, f"Worker '{payload.worker_id}' not found") from None
    return WorkerHeartbeatResponse(
        success=True, worker_id=worker.worker_id, message="Heartbeat received",
        server_timestamp=worker.last_seen, worker=worker,
    )


@router.post("/workers/unregister", response_model=WorkerUnregisterResponse)
def unregister_worker(payload: WorkerUnregister, registry: Registry) -> WorkerUnregisterResponse:
    """Remove an existing worker from memory."""
    try:
        registry.unregister_worker(payload.worker_id)
    except UnknownWorkerError:
        raise HTTPException(404, f"Worker '{payload.worker_id}' not found") from None
    return WorkerUnregisterResponse(success=True, message="Worker unregistered", worker_id=payload.worker_id)


@router.get("/cluster", response_model=ClusterSummary)
def cluster_summary(request: Request, registry: Registry) -> ClusterSummary:
    """Summarize currently active workers and their separate GPU capacities."""
    return registry.get_cluster_summary(timeout=request.app.state.worker_timeout)


@router.post("/workers/{worker_id}/gpu-test", response_model=GpuTestResponse)
def gpu_test(worker_id: str, payload: GpuTestRequest, request: Request, registry: Registry) -> GpuTestResponse:
    """Forward a bounded GPU diagnostic to a selected active worker."""
    workers = registry.get_workers(timeout=request.app.state.worker_timeout)
    worker = next((item for item in workers if item.worker_id == worker_id), None)
    if worker is None:
        raise HTTPException(404, "Worker is not active")
    return forward_gpu_test(worker, payload)
