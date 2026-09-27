"""Validated HTTP routes for the local controller."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from controller.registry import (
    DuplicateWorkerError,
    UnknownWorkerError,
    Worker,
    WorkerRegistration,
    WorkerRegistry,
)

router = APIRouter()


class HealthResponse(BaseModel):
    """Controller health information."""

    status: Literal["healthy"] = "healthy"
    service: Literal["colabcluster-controller"] = "colabcluster-controller"
    version: Literal["0.1.0"] = "0.1.0"


class WorkersResponse(BaseModel):
    """Snapshot of registered workers."""

    workers: list[Worker]


class WorkerIdRequest(BaseModel):
    """Identify a registered worker."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    worker_id: str = Field(min_length=1)


class WorkerResponse(BaseModel):
    """Successful registration or heartbeat."""

    message: str
    worker: Worker


class UnregisterResponse(BaseModel):
    """Confirmation that a worker was removed."""

    message: str
    worker_id: str


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


@router.post("/workers/register", response_model=WorkerResponse, status_code=201)
def register_worker(payload: WorkerRegistration, registry: Registry) -> WorkerResponse:
    """Register a worker with a unique ID."""
    try:
        worker = registry.register_worker(payload)
    except DuplicateWorkerError:
        raise HTTPException(409, f"Worker '{payload.worker_id}' is already registered") from None
    return WorkerResponse(message="Worker registered", worker=worker)


@router.post("/workers/heartbeat", response_model=WorkerResponse)
def heartbeat(payload: WorkerIdRequest, registry: Registry) -> WorkerResponse:
    """Update the last-seen time of an existing worker."""
    try:
        worker = registry.update_heartbeat(payload.worker_id)
    except UnknownWorkerError:
        raise HTTPException(404, f"Worker '{payload.worker_id}' not found") from None
    return WorkerResponse(message="Heartbeat received", worker=worker)


@router.post("/workers/unregister", response_model=UnregisterResponse)
def unregister_worker(payload: WorkerIdRequest, registry: Registry) -> UnregisterResponse:
    """Remove an existing worker from memory."""
    try:
        registry.unregister_worker(payload.worker_id)
    except UnknownWorkerError:
        raise HTTPException(404, f"Worker '{payload.worker_id}' not found") from None
    return UnregisterResponse(message="Worker unregistered", worker_id=payload.worker_id)
