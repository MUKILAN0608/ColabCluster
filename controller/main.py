"""ASGI entry point: uvicorn controller.main:app --reload."""

import asyncio
from contextlib import asynccontextmanager, suppress
from collections.abc import AsyncIterator

from fastapi import FastAPI

from common.config import positive_seconds
from controller.api import router
from controller.liveness import monitor_workers
from controller.registry import WorkerRegistry


def create_app() -> FastAPI:
    """Create an application with an independent in-memory registry."""
    timeout = positive_seconds("COLABCLUSTER_WORKER_TIMEOUT", 30)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Start one monitor and cancel/await it on shutdown or reload."""
        task = asyncio.create_task(monitor_workers(application.state.registry, timeout))
        application.state.liveness_task = task
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    application = FastAPI(title="ColabCluster Controller", version="0.1.0", lifespan=lifespan)
    application.state.worker_timeout = timeout
    application.state.registry = WorkerRegistry()
    application.include_router(router)
    return application


app = create_app()
