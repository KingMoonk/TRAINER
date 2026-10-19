"""The training-run page's one read: the run now (or the current version's latest run), with the page logic the Tk page
keeps in studio_ia/training_logic.py (phase_at, history_of, switch_note) applied, so the browser does not repeat it.

The run is found from the queue and the version store, never from a path the client sends.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException

from studio_api.app import get_services
from studio_api.models_queue_data import Model
from studio_api.routers.pretraining import CurveModel, ReferencesModel, _absolute, _run_of_version

router = APIRouter(prefix="/training-run", tags=["training-run"])
Services = Annotated[Any, Depends(get_services)]

EMPTY_CURVE = {"steps": [], "smooth": [], "heldout": [], "lr": [], "phases": [], "total": 0}


class SwitchNote(Model):
    """Which line the page says about Chat switching (studio_ia.training_logic.switch_note): the key is the copy's, gap is
    in bits per character, docs and need are set for gate_few_docs."""

    key: Literal["gate_few_docs", "gate_better", "gate_not_sure", "gate_not_better", "no_band_yet", "held_back_below", "held_back_inside"]
    gap: str | None = None
    docs: int | None = None
    need: int | None = None


class Ended(Model):
    """Why a run ended itself (the run's `run.json`): `plateau` is the validation loss going flat (studio_plateau)."""

    reason: str
    text: str
    step: int | None = None


def _ended(services, state, version):
    """The finished run's own reason for ending, when it wrote one; None while a job runs or for a run that simply finished."""
    if state.job is not None or version is None:
        return None
    from pathlib import Path

    from studio_core import read_json
    try:
        path = Path(_run_of_version(services, version.number)) / "run.json"
        found = read_json(path).get("ended") if path.exists() else None
    except (HTTPException, OSError, ValueError):
        return None
    return Ended.model_validate(found) if isinstance(found, dict) and found.get("text") else None


class RunPageModel(Model):
    version: int | None = None  # the version being trained (the newest)
    parent: int | None = None  # the version it continues from, if one exists
    change: str = ""
    running: bool = False
    job_id: str | None = None  # the running queue job
    step: int = 0
    total: int = 0
    phase: str = ""  # warm-up | steady | cool-down; "" without a schedule
    status: Literal["learning", "flat", "memorising"] | None = None  # null: no points yet
    curve: CurveModel
    references: ReferencesModel
    chat_version: int | None = None  # the version Chat uses
    gate_vs: int | None = None  # the version the finished run was weighed against
    switch: SwitchNote
    ended: Ended | None = None  # why the finished run ended itself, when it did


def _current(services):
    try:
        return services.current_version()
    except NotImplementedError:
        return None


def _curve(services, state, version):
    """(curve dict, job): the running job's run, else the version's latest run, else nothing."""
    job = state.job
    run_dir = job.params.get("output_dir") if job else None
    if run_dir:
        return services.run_curve(_absolute(services, run_dir))
    if version is not None:
        try:
            return services.run_curve(_run_of_version(services, version.number))
        except HTTPException:  # no run yet
            pass
    return dict(EMPTY_CURVE)


@router.get("")
def training_run(services: Services) -> RunPageModel:
    """The run the page shows: a running job's live run, else the newest version's latest run."""
    import studio_eval
    from studio_ia import training_logic as page

    state = services.queue()
    version = _current(services)
    parent = next((p for p in services.versions() if version and p.number == version.parent), None)
    curve = _curve(services, state, version)
    steps, held = curve.get("steps") or [], curve.get("heldout") or []
    running = state.job is not None
    step = state.step if running else (int(steps[-1][0]) if steps else 0)
    phase = (state.phase if running else "") or page.phase_at(curve.get("phases") or [], step)
    history = curve.get("history") or page.history_of(steps, held)
    try:
        references = services.references(parent or version) if version else {}
    except NotImplementedError:
        references = {}
    try:
        gate = services.gate_result(version) if version and not running else None
    except NotImplementedError:
        gate = None
    chat = services.best_version() if version else None
    key, extra = page.switch_note(running, held[-1][1] if held else None, references.get("best_so_far_bpc"), gate)
    return RunPageModel(
        version=version.number if version else None, parent=parent.number if parent else None, change=version.change if version else "",
        running=running, job_id=state.job.id if running else None, step=step, total=state.total if running else curve.get("total") or 0,
        phase=phase, status=studio_eval.run_status(history) if history else None, curve=CurveModel.model_validate(curve),
        references=ReferencesModel.model_validate(references), chat_version=(chat.number if chat else parent.number if parent else None),
        gate_vs=gate["vs"] if gate else None, switch=SwitchNote(key=key, **extra), ended=_ended(services, state, version),
    )
