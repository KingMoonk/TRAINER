"""Viewer: the 3D scene for a browser renderer (studio_scene_json), and the controls of the Panda3D viewer process.

The scene routes keep nothing between calls. The browser holds the controls (`ViewControls`, the keys of web/src/lib/viewer.ts's
SceneQuery plus the rest of the viewer's state), and every call rebuilds the SceneModel from the version's design and loads the
snapshot it names. The snapshots are the ones the server knows (the version's training snapshots and the ones "Run one real
step" saved): a client names one by index, never by path. At the Model level the scene also carries the version's saved
whole-context summary (POST /viewer/summary makes it) when it is current; otherwise the note says what to do.
"""

import copy
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import studio_scene_json as scene_json
from studio_api.app import get_services
from studio_api.models_work import JobOut, plain, start_job
from studio_ia.copy import INSIDE3D as T
from studio_scene import SceneModel
from studio_snapshot import load as load_snapshot

router = APIRouter(prefix="/viewer", tags=["viewer"])

Level = Literal["Model", "Window", "Head", "Weights"]
Mode = Literal["Build", "Train", "Ride a token"]
HeadStage = Literal["scores", "scale", "mask", "pattern", "values"]
TileKind = Literal["weight", "gradient", "update"]
Arrays = Literal["inline", "binary"]  # numbers in the JSON, or {shape, name} into the bytes of /viewer/binary
# What a browser control does; the other actions are messages the studio sends the viewer process (spec, summary, stats, replay,
# and snapshot paths), which a client has no business sending.
BROWSER_ACTIONS = ("scene", "level", "mode", "selection", "step", "show", "matrix", "swap", "next_block", "token", "head", "pan",
                   "cell", "motion", "numbers", "play", "advance", "scrub", "probe_step")


class ViewControls(BaseModel):
    """Everything the viewer keeps between frames. The first five are web/src/lib/viewer.ts's SceneQuery."""
    model_config = ConfigDict(populate_by_name=True)

    version: int | None = None  # None = the current version
    level: Level = "Model"
    mode: Mode = "Build"
    head_stage: HeadStage = Field("pattern", alias="headStage")
    tile_kind: TileKind = Field("weight", alias="tileKind")
    snapshot: int = 0  # which snapshot to show, from 0; past the end is the last one
    selection: dict[str, Any] | None = None  # the picked block, token, head, matrix and tile origin (SceneModel.selection)
    cell: list[int] = Field(default_factory=lambda: [0, 0], min_length=2, max_length=2)
    playing: bool = False
    reduce_motion: bool = False
    show_numbers: bool = True
    spec: dict[str, Any] | None = None  # a design the client changed (swap a mixer); None = the version's own
    note: str = ""
    probing: bool = False


class FrameRequest(ViewControls):
    since: dict[str, Any] | None = None  # the `cursor` of the last payload or frame the client holds; None = the whole scene
    arrays: Arrays = "inline"


class ActionRequest(FrameRequest):
    action: dict[str, Any]  # {"type": ..., ...}: the viewer's own messages (see studio_scene_json.ACTIONS)


class ActionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: ViewControls  # the controls after the action: send them with the next call
    answer: Any = None  # what the action asks the studio to do: probe_step True (run it: POST /viewer/probe), swap its message
    scene: dict[str, Any]  # the whole scene, or the frame against `since`


class ViewerStatus(BaseModel):
    available: bool
    running: bool
    connected: bool


class SnapshotOut(BaseModel):
    index: int
    name: str


class StartRequest(BaseModel):
    level: str | None = None
    mode: str | None = None


class ProbeRequest(BaseModel):
    version: int | None = Field(default=None, ge=1)
    text: str = Field(default=T["probe_text"], min_length=2)


class ProbeResult(BaseModel):
    snapshot: int  # the new snapshot's index in /viewer/snapshots
    note: str


class ViewerSummaryRequest(BaseModel):
    version: int | None = Field(default=None, ge=1)


class ViewerSummaryStatus(BaseModel):
    state: Literal["ready", "stale", "none", "no_checkpoint"]  # none/stale/no_checkpoint: the Model level draws without attention or stream sizes
    step: int | None = None  # the training step of the checkpoint the summary came from
    bins: int | None = None
    n_ctx: int | None = None
    note: str


class ViewerSummaryResult(BaseModel):
    step: int
    bins: int
    n_ctx: int
    note: str


def snapshot_paths(request: Request, number=None):
    """The snapshots a client may name: the version's training snapshots, then the ones a probe saved (oldest first)."""
    services = request.app.state.services
    version = pick_version(services, number)
    paths = [str(path) for path in services.viewer_snapshots(version.number)]
    return paths + [path for path in probes(request, version.number) if path not in paths]


def probes(request: Request, number):
    """The snapshots "Run one real step" saved since the server started, oldest first."""
    if not hasattr(request.app.state, "viewer_probes"):
        request.app.state.viewer_probes = {}
    return request.app.state.viewer_probes.setdefault(number, [])


def pick_version(services, number):
    version = services.current_version() if number is None else next((v for v in services.versions() if v.number == number), None)
    if version is None:
        raise ValueError("There is no version yet. Build one first." if number is None else f"There is no version {number}.")
    return version


def build(request: Request, state: ViewControls):
    """The SceneModel and ViewState for `state`: the version's design, the snapshot it names, and the rest of the controls."""
    services = request.app.state.services
    version = pick_version(services, state.version)
    spec = state.spec or version.spec
    if not spec:
        raise ValueError(f"Version {version.number} has no design to draw.")
    model = SceneModel(copy.deepcopy(spec))
    model.choose(**(state.selection or {}))  # ValueError for a field or a coordinate that is not one
    model.level, model.mode = state.level, state.mode
    model.reduce_motion, model.show_numbers = state.reduce_motion, state.show_numbers
    paths = snapshot_paths(request, version.number)
    view = scene_json.ViewState(head_stage=state.head_stage, tile_kind=state.tile_kind, cell=list(state.cell), playing=state.playing,
                                snapshot_paths=paths, note=state.note, probing=state.probing)
    scene_json.scrub_to(model, view, state.snapshot, known(paths))  # a snapshot that will not load is said in the note
    if state.spec:
        model.update_spec(copy.deepcopy(state.spec))  # a design the client changed wins over the snapshot's
    if state.level == "Model":
        add_summary(services, version.number, model, view)
    return model, view


SUMMARY_NOTES = {T[key] for key in ("summary_none", "summary_stale", "summary_no_checkpoint")}


def summary_of(services, number, n_ctx):
    """(state, summary) of the version's saved summary: ready only when it is for a context as long as the design drawn (`n_ctx`)."""
    state, summary = services.viewer_summary(number)
    return ("stale" if state == "ready" and summary["n_ctx"] != n_ctx else state), summary


def add_summary(services, number, model, view):
    """Give the Model level the version's whole-context summary. Without a current one (never made, made from another checkpoint,
    or no checkpoint) nothing is drawn for it and the note says what to do. A note from the controls that was only this advice is
    replaced, so it does not outlive the summary."""
    state, summary = summary_of(services, number, model.spec["dims"]["n_ctx"])
    if state == "ready":
        scene_json.set_summary(model, {"attention": summary["attention"], "stream_norms": summary["stream_norms"], "edges": summary["edges"]})
    if not view.note or view.note in SUMMARY_NOTES:
        view.note = "" if state == "ready" else T[f"summary_{state}"]


def known(paths):
    """A snapshot loader that opens only the files the server listed."""
    allowed = set(paths)

    def load(path):
        if str(path) not in allowed:
            raise ValueError("That is not one of this version's snapshots.")
        return load_snapshot(path)
    return load


def controls_after(state, model, view):
    return ViewControls(version=state.version, level=model.level, mode=model.mode, head_stage=view.head_stage, tile_kind=view.tile_kind,
                        snapshot=max(view.snapshot_index, 0), selection=asdict(model.selection), cell=list(view.cell),
                        playing=view.playing, reduce_motion=model.reduce_motion, show_numbers=model.show_numbers,
                        spec=copy.deepcopy(model.spec), note=view.note, probing=view.probing)


def controls_query(version: int | None = None, level: Level = "Model", mode: Mode = "Build",
                   headStage: HeadStage = "pattern", tileKind: TileKind = "weight", snapshot: int = 0):
    return ViewControls(version=version, level=level, mode=mode, head_stage=headStage, tile_kind=tileKind, snapshot=snapshot)


@router.get("/scene")
def scene(request: Request, state: Annotated[ViewControls, Depends(controls_query)], arrays: Arrays = "inline"):
    """The whole scene to draw from scratch (studio_scene_json.scene_payload). Query: version, level, mode, headStage, tileKind,
    snapshot, arrays; web/src/lib/viewer.ts's SceneQuery. A client that keeps more of the controls POSTs /viewer/frame."""
    model, view = build(request, state)
    return JSONResponse(scene_json.scene_payload(model, view, arrays))


@router.post("/frame")
def frame(request: Request, body: FrameRequest):
    """What changed since `since` (studio_scene_json.scene_frame); the whole scene when `since` is empty."""
    model, view = build(request, body)
    return JSONResponse(scene_json.scene_frame(model, body.since, view, body.arrays))


@router.post("/binary")
def binary(request: Request, body: FrameRequest):
    """The arrays of the scene (or of the frame against `since`) as little-endian float32 bytes. Their offsets are in
    POST /viewer/manifest, and the payload's {shape, name} entries name them: send both calls the same body."""
    model, view = build(request, body)
    data, _manifest = scene_json.scene_binary(model, view, body.since)
    return Response(data, media_type="application/octet-stream")


@router.post("/manifest")
def manifest(request: Request, body: FrameRequest):
    """{name: {offset, shape, dtype}} for the bytes of POST /viewer/binary with the same body."""
    model, view = build(request, body)
    _data, layout = scene_json.scene_binary(model, view, body.since)
    return layout


@router.post("/action", response_model=ActionOut)
def action(request: Request, body: ActionRequest):
    """Do what a viewer control does (studio_scene_json.apply_action) to the state in the body, and answer with the new state and
    the scene (the whole one, or the frame against `since`)."""
    kind = body.action.get("type")
    if kind not in BROWSER_ACTIONS:
        raise ValueError(f"The browser can send these actions: {', '.join(BROWSER_ACTIONS)}.")
    model, view = build(request, body)
    answer = scene_json.apply_action(model, view, body.action, known(view.snapshot_paths))
    return ActionOut(state=controls_after(body, model, view), answer=plain(answer),
                     scene=scene_json.scene_frame(model, body.since, view, body.arrays))


@router.get("/snapshots", response_model=list[SnapshotOut])
def snapshots(request: Request, version: int | None = None):
    """The snapshots a scene can show, oldest first: a scene's `snapshot` is an index into this list."""
    return [SnapshotOut(index=index, name=path.replace("\\", "/").rsplit("/", 1)[-1]) for index, path in enumerate(snapshot_paths(request, version))]


@router.post("/probe", response_model=JobOut, status_code=202)
def probe(request: Request, body: ProbeRequest | None = None):
    """Run one real training step of the current version's checkpoint on a sentence, without touching training. A job: the result
    is {snapshot, note}, the new snapshot's index (see /viewer/snapshots) and the sentence to show."""
    services = request.app.state.services
    options = body or ProbeRequest()
    version = pick_version(services, options.version)

    def run(stop):
        path = services.viewer_probe_step(options.text, version.number)
        if path is None:
            raise ValueError(T["probe_no_checkpoint"])
        if str(path) not in probes(request, version.number):
            probes(request, version.number).append(str(path))
        state, _ = summary_of(services, version.number, (version.spec or {}).get("dims", {}).get("n_ctx"))
        if state in ("none", "stale"):  # the same checkpoint, so the same numbers: only a missing or old summary is made again
            try:
                services.viewer_summarize(version.number)
            except ValueError:
                pass  # the probe worked; the Model level's note says why there is no summary
        return {"snapshot": snapshot_paths(request, version.number).index(str(path)), "note": T["probe_done"].format(n=version.number)}
    return start_job(request, "viewer.probe", run, ProbeResult)


@router.get("/summary", response_model=ViewerSummaryStatus)
def summary(request: Request, version: int | None = None):
    """Whether the version's whole-context summary (the Model level's attention between positions and stream sizes) is there: ready,
    or what is missing. POST /viewer/summary makes it."""
    services = request.app.state.services
    picked = pick_version(services, version)
    state, found = summary_of(services, picked.number, (picked.spec or {}).get("dims", {}).get("n_ctx"))
    ready = state == "ready"
    return ViewerSummaryStatus(state=state, step=found["step"] if found else None, bins=len(found["edges"]) - 1 if found else None,
                         n_ctx=found["n_ctx"] if found else None, note=T["summary_done"].format(n=picked.number) if ready else T[f"summary_{state}"])


@router.post("/summary", response_model=JobOut, status_code=202)
def make_summary(request: Request, body: ViewerSummaryRequest | None = None):
    """Compute the version's whole-context summary as a job: one inference pass over a full context window of its held-back text
    (studio_summary.whole_context), saved beside the run's snapshots with the checkpoint and step it came from."""
    services = request.app.state.services
    picked = pick_version(services, (body or ViewerSummaryRequest()).version)

    def run(stop):
        made = services.viewer_summarize(picked.number)
        return {"step": made["step"], "bins": len(made["edges"]) - 1, "n_ctx": made["n_ctx"], "note": T["summary_done"].format(n=picked.number)}
    return start_job(request, "viewer.summary", run, ViewerSummaryResult)


@router.get("/status", response_model=ViewerStatus)
def status(services=Depends(get_services)):
    """Whether the Panda3D viewer process can be launched on this computer, is running, and is connected."""
    return ViewerStatus(**services.viewer_status())


@router.post("/start", response_model=ViewerStatus)
def start(body: StartRequest | None = None, services=Depends(get_services)):
    """Launch the Panda3D viewer process on the server's computer (no-op if running) with the current version's design."""
    scene_choice = {key: value for key, value in (body or StartRequest()).model_dump().items() if value}
    services.viewer_start(scene_choice or None)
    return ViewerStatus(**services.viewer_status())


@router.post("/stop", response_model=ViewerStatus)
def stop(services=Depends(get_services)):
    services.viewer_stop()
    return ViewerStatus(**services.viewer_status())


@router.post("/send", status_code=204)
def send(message: dict[str, Any], services=Depends(get_services)):
    """Queue one message (e.g. {"type": "scene", "level": "Window", "mode": "Train"}) for the running viewer process."""
    services.viewer_send(message)
    return Response(status_code=204)


@router.post("/poll")
def poll(services=Depends(get_services)):
    """The messages the viewer process sent back since the last call. Reading them empties the list."""
    return plain(services.viewer_poll())
