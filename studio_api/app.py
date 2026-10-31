"""create_app(services): the FastAPI app over a Services instance (WorkspaceServices or FakeServices)."""

import importlib
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from studio_api import events, jobs, traffic
from studio_api.auth import TailscaleAuth
from studio_api.hosts import ANY, HostCheck, HostNames
from studio_api.static import ExportFiles
from studio_api.version import app_version
from studio_versions import NoCheckpoint

ROUTERS = ("versions", "queue", "storage", "data", "pretraining", "post_training", "chat", "benchmark",
           "node", "quicktest", "design", "cluster", "collective", "distributed", "training_run", "state", "collect", "mlperf", "busy", "voice", "auto_run")  # one module each in studio_api.routers
THREADS = 128  # anyio's default is 40 threads for every plain `def` route; a few slow peers or long reads could fill them and stall the rest
LOCAL_ORIGIN = r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
WEB_OUT = Path(__file__).resolve().parent.parent / "web" / "out"


def get_services(request: Request):
    return request.app.state.services


def create_app(services, open=False, origins=(), hosts=(), fake_traffic=False, upload_admission_options=None, **auth):
    """`auth` is TailscaleAuth's allow_tags, allow_users, whois and token (studio_cluster.allowed_callers gives the first two,
    studio_cluster.funnel_token the token). `origins` are the web origins (the Vercel site) allowed to call this API from a
    browser: their requests get the CORS headers only once authenticated, and only their preflight is answered without.
    `hosts` are the names the Host header may carry besides localhost and IP addresses (studio_api.hosts.own_names); the default
    allows only those. Pass hosts=ANY to skip the check (tests over a TestClient, the OpenAPI dump). `fake_traffic` (--fake) has a made-up
    peer take part in the fake run, so the traffic console has DiLoCo exchanges to show (studio_api.traffic.simulate)."""
    @asynccontextmanager
    async def lifespan(app):
        anyio.to_thread.current_default_thread_limiter().total_tokens = THREADS
        async def expire_upload_credit():
            while True:
                await anyio.sleep(1.)
                app.state.upload_admission.expire()

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(expire_upload_credit)
                try:
                    yield
                finally:
                    group.cancel_scope.cancel()
        finally:
            app.state.upload_admission.close()
        if isinstance(hosts, HostNames):
            hosts.stop.set()  # the Tailscale-name refresh thread
        app.state.jobs.close()
        app.state.broadcaster.close()
        app.state.traffic.close()

    app = FastAPI(title="Groundwork API", version=app_version(), lifespan=lifespan)
    app.state.services = services
    # Receiver grant accounting is shared by requests handled in this process.
    from studio_upload_admission import UploadAdmission
    from studio_peer import _RECEIVER_INGRESS
    admission_options = dict(upload_admission_options or {})
    admission_options.setdefault("ingress_budget", _RECEIVER_INGRESS)
    admission_options.setdefault("on_event", lambda kind, event: services.emit("collective_upload", {"kind": kind, **event}))
    app.state.upload_admission = UploadAdmission(**admission_options)
    app.state.upload_apply_limiter = anyio.CapacityLimiter(2)
    app.state.download_build_limiter = anyio.CapacityLimiter(2)
    app.state.open = open
    app.state.hosts = hosts  # a HostNames knows this node's tailnet name (GET /cluster/nodes)
    app.state.broadcaster = events.Broadcaster(services)
    app.state.jobs = jobs.Jobs(services)
    app.state.traffic = tap = traffic.Traffic(emit=services.emit)  # GET /traffic; the middleware below fills it
    tap.watch_peers()  # the calls studio_peer makes from this process
    if fake_traffic:
        tap.own(traffic.simulate(tap, services))
    origins = tuple(origins)
    cors = dict(allow_methods=["*"], allow_headers=["*"])
    local_pages = open and not auth.get("token")  # a node with a token is not open: a localhost page could forge Tailscale-User-Login
    if origins:  # ONE CORS layer, inside the auth layer: a response the auth layer refused carries no CORS headers
        app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_origin_regex=LOCAL_ORIGIN if local_pages else None, **cors)
    app.add_middleware(TailscaleAuth, open=open, origins=origins, **auth)
    if local_pages and not origins:  # loopback or --fake: let `pnpm dev` (http://localhost:3000) call this server from another origin
        app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN, **cors)
    app.add_middleware(traffic.TrafficMiddleware, traffic=tap)  # outside auth and CORS: a refused request is traffic too
    if hosts is not ANY:  # outermost: a request for a foreign name never reaches auth, CORS or a route
        app.add_middleware(HostCheck, hosts=hosts, origins=origins)

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
    from studio_api.routers import pretraining
    services.build_pretrain = lambda settings, version: pretraining.auto_run_pretrain(
        services, settings, version, peer_url=lambda: importlib.import_module("studio_api.routers.distributed").peer_address(app))  # the Auto Run's pretrain phase
    app.include_router(events.router)
    app.include_router(traffic.router)
    app.include_router(jobs.router)
    if WEB_OUT.is_dir():
        app.mount("/", ExportFiles(directory=WEB_OUT, html=True), name="web")
    return app
