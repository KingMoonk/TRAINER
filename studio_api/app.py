"""create_app(services): the FastAPI app over a Services instance (WorkspaceServices or FakeServices)."""

import importlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from studio_api import events, jobs
from studio_api.auth import TailscaleAuth
from studio_api.static import ExportFiles
from studio_api.version import app_version
from studio_versions import NoCheckpoint

ROUTERS = ("versions", "queue", "storage", "passive", "data", "pretraining", "post_training", "chat", "inspect", "benchmark",
           "viewer", "node", "quicktest", "design", "cluster", "training_run")  # one module each in studio_api.routers
LOCAL_ORIGIN = r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
WEB_OUT = Path(__file__).resolve().parent.parent / "web" / "out"


def get_services(request: Request):
    return request.app.state.services


def create_app(services, open=False, **auth):
    """`auth` is TailscaleAuth's allow_tags, allow_users and whois (studio_cluster.allowed_callers gives the first two)."""
    @asynccontextmanager
    async def lifespan(app):
        yield
        app.state.jobs.close()
        app.state.broadcaster.close()

    app = FastAPI(title="Groundwork API", version=app_version(), lifespan=lifespan)
    app.state.services = services
    app.state.open = open
    app.state.broadcaster = events.Broadcaster(services)
    app.state.jobs = jobs.Jobs(services)
    app.add_middleware(TailscaleAuth, open=open, **auth)
    if open:  # loopback or --fake: let `pnpm dev` (http://localhost:3000) call this server from another origin
        app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NoCheckpoint)
    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    for name in ROUTERS:
        try:
            module = importlib.import_module(f"studio_api.routers.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"studio_api.routers.{name}":
                raise
            continue  # router not written yet
        app.include_router(module.router)
    app.include_router(events.router)
    app.include_router(jobs.router)
    if WEB_OUT.is_dir():
        app.mount("/", ExportFiles(directory=WEB_OUT, html=True), name="web")
    return app
