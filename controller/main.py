"""ASGI entry point: uvicorn controller.main:app --reload."""

from fastapi import FastAPI

from controller.api import router
from controller.registry import WorkerRegistry


def create_app() -> FastAPI:
    """Create an application with an independent in-memory registry."""
    application = FastAPI(title="ColabCluster Controller", version="0.1.0")
    application.state.registry = WorkerRegistry()
    application.include_router(router)
    return application


app = create_app()
