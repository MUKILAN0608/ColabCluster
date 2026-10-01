"""Validated HTTP routes for the local controller."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from common.schemas import (
    HealthResponse, WorkersResponse, WorkerRegistration, WorkerRegistrationResponse,
    WorkerHeartbeat, WorkerHeartbeatResponse, WorkerUnregister, WorkerUnregisterResponse,
)
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
def list_workers(registry: Registry) -> WorkersResponse:
    """List all workers currently in memory."""
    return WorkersResponse(workers=registry.get_workers())


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
        worker = registry.update_heartbeat(payload.worker_id, status=status)
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
