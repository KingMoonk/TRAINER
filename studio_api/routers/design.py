"""The Builder's design maths over HTTP (studio_ia.design): the module registry, and one stateless POST per thing the Tk
page works out while editing. The draft lives in the browser: every route takes the spec or choices it means and holds
nothing. The ones that need the version history or this machine read Services or the OS, as the page does.

Choices are the page's draft dict (read_spec): L blocks, H heads, W thinking room, V word list, d and T, M the module of
every part and O the per-block picks. O's (block, slot) keys become a list of {block, slot, id} on the wire.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import studio_registry as registry
from studio_api.app import get_services
from studio_api.models_work import plain
from studio_ia import design
from studio_slots import BLOCK_SLOTS, SLOTS
from studio_spec import validate_spec

router = APIRouter(prefix="/design", tags=["design"])
Services = Annotated[Any, Depends(get_services)]

Slot = Literal["embedding", "position", "head", "norm", "mixer", "channel"]
TOWER_SLOTS = (*design.PART_SLOTS, *BLOCK_SLOTS)


def maths(fn, *args, **kwargs):
    """Call a design function; a spec it cannot read is the browser's mistake (422), not the server's (500)."""
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


# ---------------------------------------------------------------------------------------------- the choices on the wire

class Own(BaseModel):
    block: int = Field(ge=0)
    slot: Literal["norm", "mixer", "channel"]
    id: str


class Parts(BaseModel):
    embedding: str
    position: str
    head: str
    norm: str
    mixer: str
    channel: str


class Choices(BaseModel):
    L: int = Field(ge=1)
    H: int = Field(ge=1)
    W: int = Field(ge=0, le=2)
    V: int = Field(ge=0, le=3)
    d: int = Field(ge=1)
    T: int = Field(ge=1)
    M: Parts
    O: list[Own] = []


def choices_in(c: Choices):
    return dict(L=c.L, H=c.H, W=c.W, V=c.V, d=c.d, T=c.T, M=c.M.model_dump(), O={(o.block, o.slot): o.id for o in c.O})


def choices_out(s) -> Choices:
    own = [Own(block=block, slot=slot, id=short) for (block, slot), short in sorted(s["O"].items())]
    return Choices(L=s["L"], H=s["H"], W=s["W"], V=s["V"], d=s["d"], T=s["T"], M=Parts(**s["M"]), O=own)


# ---------------------------------------------------------------------------------------------- the registry

class ModuleInfo(BaseModel):
    short: str
    name: str
    blurb: str
    tip: str  # module_tip: the glossary's why, then the registry's blurb
    fields: list[dict]  # registry.fields: the settings it takes


class SlotInfo(BaseModel):
    slot: str
    name: str  # SLOTS' title
    tip: str  # part_tip
    modules: list[ModuleInfo]


class Registry(BaseModel):
    part_slots: list[str]  # the model-wide parts the tower shows
    block_slots: list[str]  # every block has these
    slots: list[SlotInfo]
    words: list[str]  # the V choices
    widths: list[str]  # the W choices
    heads: list[int]  # the H choices


@router.get("/registry")
def design_registry() -> Registry:
    """The modules every part can take, with their hover help and settings, plus the knob labels."""
    slots = [SlotInfo(slot=slot, name=SLOTS[slot][0], tip=design.part_tip(slot),
                      modules=[ModuleInfo(short=m.short, name=m.name, blurb=m.blurb, tip=design.module_tip(slot, m.short),
                                          fields=plain(registry.fields(m))) for m in registry.for_slot(slot)])
             for slot in TOWER_SLOTS]
    return Registry(part_slots=list(design.PART_SLOTS), block_slots=list(BLOCK_SLOTS), slots=slots, words=list(design.WORDS),
                    widths=list(design.WIDTHS), heads=list(design.HEADS))


# ---------------------------------------------------------------------------------------------- reading and writing the draft

class SpecBody(BaseModel):
    spec: dict


@router.post("/read")
def read(body: SpecBody) -> Choices:
    """read_spec: the editable choices of a spec (defaults filled in)."""
    return choices_out(maths(design.read_spec, body.spec))


class WriteBody(BaseModel):
    base: dict
    choices: Choices
    was: Choices | None = None  # the choices `base` was read as; absent: base's own


@router.post("/write")
def write(body: WriteBody) -> dict:
    """write_spec: `base` with only the choices that changed rewritten."""
    was = choices_in(body.was) if body.was else None
    return maths(design.write_spec, body.base, choices_in(body.choices), was)


class ChooseBody(BaseModel):
    choices: Choices
    slot: Slot
    short: str
    block: int | None = Field(None, ge=0)  # absent: every block


@router.post("/choose")
def choose(body: ChooseBody) -> Choices:
    """choose: the choices after picking a module for a part, for every block or for one."""
    return choices_out(design.choose(choices_in(body.choices), body.slot, body.short, body.block))


class ResizeBody(BaseModel):
    choices: Choices
    n: int = Field(ge=1)


@router.post("/resize")
def resize(body: ResizeBody) -> Choices:
    """resize: the choices with n blocks."""
    return choices_out(design.resize(choices_in(body.choices), body.n))


class Block(BaseModel):
    block: int
    parts: dict[str, str]  # block_parts: slot -> module id, the all-blocks choice or its own
    own: list[str]  # own_slots: the slots it has its own module in


class Knobs(BaseModel):
    H: bool  # knob_applies: whether look-back heads mean anything for the all-blocks mixer
    W: bool  # and thinking room for the channel mixer


class Tower(BaseModel):
    blocks: list[Block]
    knobs: Knobs


class ChoicesBody(BaseModel):
    choices: Choices


@router.post("/tower")
def tower(body: ChoicesBody) -> Tower:
    """What the tower paints: every block's parts (block_parts, own_slots) and which knobs apply (knob_applies)."""
    s = choices_in(body.choices)
    blocks = [Block(block=i, parts=design.block_parts(s, i), own=design.own_slots(s, i)) for i in range(s["L"])]
    return Tower(blocks=blocks, knobs=Knobs(H=maths(design.knob_applies, s, "H"), W=maths(design.knob_applies, s, "W")))


class OptionsBody(BaseModel):
    base: dict
    choices: Choices
    was: Choices | None = None
    slot: Slot
    block: int | None = Field(None, ge=0)


class Option(BaseModel):
    short: str
    name: str
    blurb: str
    why: str  # why it cannot be picked here, "" when it can


@router.post("/part-options")
def part_options(body: OptionsBody) -> list[Option]:
    """part_options: one part's menu, with why_not's reason on every module the design cannot take."""
    was = choices_in(body.was) if body.was else None
    found = maths(design.part_options, body.base, choices_in(body.choices), was, body.slot, body.block)
    return [Option(short=short, name=name, blurb=blurb, why=why) for short, name, blurb, why in found]


# ---------------------------------------------------------------------------------------------- what it costs

class SizesBody(BaseModel):
    spec: dict
    v: int = Field(ge=0, le=3)  # the word list


class Sizes(BaseModel):
    params: int
    table: int
    time: float
    memory: int
    activations: int
    parts: dict[str, int]  # param_parts: parameters in the word table, position, blocks and head
    fits: bool | None  # fits: in this machine's memory at ROWS rows; null when the OS does not say


@router.post("/sizes")
def sizes(body: SizesBody) -> Sizes:
    """sizes: size, time and memory of a design under word list v, and whether it fits this computer (fits)."""
    c = design.sizes(maths(validate_spec, body.spec), body.v)
    return Sizes(**c, fits=design.fits(c))


class Preset(BaseModel):
    key: str
    term: str  # the glossary entry behind it
    name: str
    spec: dict


@router.get("/presets")
def presets() -> list[Preset]:
    """preset_specs: the ready-made designs, GPT-2's with its word list when that is on this computer."""
    return [Preset(key=key, term=design.preset_term(key), name=spec["name"], spec=spec)
            for key, spec in design.preset_specs(design.gpt2_words_ready())]


class PresetCost(BaseModel):
    params: int
    need: int
    fits: bool | None
    rows: int
    step: float
    slow: bool
    lines: list[str]
    recipe: str  # recipe_text


@router.post("/preset-cost")
def preset_cost(body: SpecBody) -> PresetCost:
    """preset_cost: what a design costs on this machine before it is loaded, and its training recipe in one line."""
    spec = maths(validate_spec, body.spec)
    return PresetCost(**design.preset_cost(spec, gpu=design.gpu_kind()), recipe=design.recipe_text(spec))


class ComputeBody(BaseModel):
    spec: dict
    v: int = Field(ge=0, le=3)
    preview: dict | None = None  # GET /pretraining/preview's answer; null while the text is still being counted


class Line(BaseModel):
    text: str
    bad: bool


@router.post("/compute-lines")
def compute_lines(body: ComputeBody) -> list[Line]:
    """compute_lines: how much text a design needs against the text you have, and how long a run should take."""
    spec = maths(validate_spec, body.spec)
    params = design.sizes(spec, body.v)["params"]
    lines = design.compute_lines(params, spec["dims"]["n_ctx"], spec["dims"]["n_layer"], body.preview, design.gpu_kind())
    return [Line(text=text, bad=bad) for text, bad in lines]


class LrBody(BaseModel):
    result: dict | None = None  # the finished lr_range_test job's result
    current: float | None = None  # the rate the design trains with now


class Text(BaseModel):
    text: str


@router.post("/lr-text")
def lr_text(body: LrBody) -> Text:
    """lr_text: the sentence under "Find a learning rate"."""
    return Text(text=design.lr_text(body.result, body.current))


# ---------------------------------------------------------------------------------------------- what to expect

class ExpectBody(BaseModel):
    cur: Choices  # the version the draft started from
    draft: Choices
    base: dict  # its spec
    draft_spec: dict | None = None  # write's answer for the draft; absent when it cannot be written


class Expectations(BaseModel):
    headline: str  # design_expectation
    details: str
    lines: list[str]  # expectations: one per change, then the advice the draft gets


@router.post("/expectations")
def expectations(body: ExpectBody, services: Services) -> Expectations:
    """design_expectation for the draft, and expectations for its changes. "Last time blocks changed" reads the history
    (Services.versions, run_facts, compare_runs), only when the block count differs, as the page does."""
    cur, draft = choices_in(body.cur), choices_in(body.draft)
    spec = maths(validate_spec, body.draft_spec or body.base)
    headline, details = design.design_expectation(draft, spec)
    issues = design.new_issues(body.base, body.draft_spec) if body.draft_spec else ()
    versions = services.versions()
    facts, compares = (services.run_facts, services.compare_runs) if draft["L"] != cur["L"] else (None, None)
    lines = design.expectations(cur, draft, versions, issues, facts, compares)
    return Expectations(headline=headline, details=details, lines=lines)


class ChangesBody(BaseModel):
    base: dict
    draft_spec: dict
    cur: Choices
    draft: Choices


class Warm(BaseModel):
    kind: str  # scratch | heads | other | removed | same
    carries: bool
    preserving: bool
    chip: str
    state: str
    text: str


class Changes(BaseModel):
    chips: list[str]  # change_chips: the not-saved chips
    parts_changed: bool
    warm: Warm | None  # warm_outlook: how the old weights carry into the draft; null when nothing changed


@router.post("/changes")
def changes(body: ChangesBody) -> Changes:
    """change_chips, parts_changed and warm_outlook for a draft against the version it started from."""
    cur, draft = choices_in(body.cur), choices_in(body.draft)
    base, draft_spec = maths(validate_spec, body.base), maths(validate_spec, body.draft_spec)
    chips = design.change_chips(base, draft_spec, cur, draft)
    warm = design.warm_outlook(cur, draft)
    return Changes(chips=chips, parts_changed=design.parts_changed(cur, draft), warm=Warm(**warm) if warm else None)
