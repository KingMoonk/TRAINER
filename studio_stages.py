"""Where a DiLoCo run goes next when its validation loss flattens, built on the `next_dataset` pivot of studio_diloco_rules (#509).

The pivot already moves a run, with the same weights and round count, to the next of the data sources it was given. This module is
the policy behind it, attached to a Coordinator as `coordinator.stages` (job params `stages`):
- the pool of sources is the workspace's plain-text sets as they are when a switch is due (`text_sets`, read fresh from disk), so
  sets a collector added while the run trained join it;
- the sources not yet read are ordered by what the weights do worst on (`rank`): the held-out loss of each source (`SCORERS`; a fixed
  eval suite, #428, registers its own scorer there and is named in `stages.scorer`);
- each source keeps a stable held-out slice (`split`) that no round trains on, so it can be measured;
- a later source's rounds also deal a replay share of the earlier sources' shards, against forgetting (`replay_ids`);
- every switch is recorded in `stages.json` and the run's journal: which set, why, and the loss on the earlier sets before and after;
- when no unseen text is left, or the model stopped learning the newest text, the run ends with a plain reason and the suggestion of
  a bigger design.

Nothing here touches the engine; the coordinator calls `before_observe`, `after_observe` and `replay_ids` under its own lock."""
from __future__ import annotations

import copy
import json
import random
import zlib
from pathlib import Path

HELD_EVERY = 20  # one text in this many of every source is kept out of training and used to measure that source
SAMPLE = 48  # held-out texts scored per source
REPLAY = .2  # share of a round's shards drawn from the sources before the current one (forgetting guard, as studio_curriculum does)
LEARNED = .02  # a new source whose held-out loss did not fall by this share while it was read was not learnable by this model
JOURNAL = "stages.json"
SUGGESTION = "To learn more, continue with a bigger design (a warm start keeps what it learned) or with more text."
NOTHING_LEFT = "No unseen text is left to move on to. "
TOO_SMALL = "On the newest text the model's held-out loss barely fell, so this design has learned what it can. "


def split(rows):
    """(train rows, held-out rows) of one source: stable, so every round and every restart agrees which texts are held out."""
    train, held = [], []
    for row in rows:
        (held if zlib.crc32(row["text"].encode("utf-8")) % HELD_EVERY == 0 else train).append(row)
    if not held and len(train) > 1:  # a small source still gets something to be measured on
        held.append(train.pop())
    return train, held


def text_sets(root, skip=()):
    """[{"id", "name"}] of the plain-text sets in the workspace under `root` right now (a fresh read), without catalogue validation and
    test splits and the ids in `skip`."""
    from studio_core import Workspace
    from studio_data_kinds import classify, is_draft
    from studio_ia.pretraining_plan import held_out
    workspace = Workspace(root)
    found = []
    for dataset in workspace.state["datasets"]:
        if dataset["id"] in skip or held_out(dataset["id"]):
            continue
        for example in workspace.iter_examples(dataset["id"]):
            if classify(example) == "text" and not is_draft(example) and example.get("text"):
                found.append({"id": dataset["id"], "name": dataset.get("name") or dataset["id"]})
                break
    return found


def held_out_losses(checkpoint, held, limit=SAMPLE):
    """{source id: the checkpoint's next-token loss on that source's held-out texts} (None where there are none to read). `held` is
    {source id: rows}. The model runs on the CPU: this is a short look between rounds, not training."""
    import studio_model
    data = studio_model.load_checkpoint(checkpoint, weights_only=True)
    cfg = studio_model.validate_config({**data["config"], "device": "cpu", "objective": "all"})
    tokenizer = studio_model.Tokenizer.restore(data["tokenizer"])
    model = studio_model.ScratchTransformer(cfg, tokenizer.vocab_size)
    model.load_weights(data["model"])
    out = {}
    for source, rows in held.items():
        samples = studio_model.samples_from([{"text": r["text"]} for r in rows[:limit]], tokenizer, cfg)
        total, count = 0., 0
        for start in range(0, len(samples), 8):
            x, y = studio_model.batch_arrays([samples[i] for i in range(start, min(start + 8, len(samples)))])
            targets = int((y != -100).sum())
            total += model.loss(x, y) * targets
            count += targets
        out[source] = total / count if count else None
    return out


SCORERS = {"held_out": held_out_losses}  # name -> scorer(checkpoint, {source id: held-out rows}) -> {source id: loss}; #428 adds its suite here


def rank(losses, unseen):
    """[(id, loss)] of the `unseen` sets, the one the weights do worst on first; a set with no measurable loss comes last."""
    return sorted(((s, losses.get(s)) for s in unseen), key=lambda item: (item[1] is None, -(item[1] or 0.)))


def mean(losses, ids):
    found = [losses[i] for i in ids if losses.get(i) is not None]
    return sum(found) / len(found) if found else None


def write_journal(folder, journal):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / JOURNAL).write_text(json.dumps(journal), encoding="utf-8")


def read_journal(folder):
    """The journal in `folder`: {"sources": [[set ids]...], "switches": [...]} or None."""
    try:
        found = json.loads((Path(folder) / JOURNAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and "switches" in found else None


class Policy:
    """The stages policy of one Coordinator. `loader(set id)` gives a set's plain-text rows, `discover(skip)` the sets that exist now
    ([{"id", "name"}]), `scorer` is a name in SCORERS. `first` are the ids of the sets the run started on (one source)."""

    def __init__(self, first, loader, discover, scorer="held_out", replay=REPLAY, held=None):
        self.loader, self.discover, self.scorer, self.replay = loader, discover, scorer, replay
        self.names = [list(first)]  # per source: the set ids it holds ("" for a source that was handed over without a name)
        self.held = dict(held or {})  # set id -> its held-out rows
        self.known = set(first)
        self.switches, self.pending, self.too_small, self.exhausted = [], None, False, False

    # -- measuring
    def _losses(self, c, ids):
        """The loss of the coordinator's current weights on the held-out slice of each of `ids`."""
        step = c.next_round * c.settings.inner_steps
        path = c.directory / "stage_probe.npz"
        c._write_npz(path, c._checkpoint_arrays(step))
        try:
            return SCORERS[self.scorer](path, {i: self.held[i] for i in ids if self.held.get(i)})
        finally:
            path.unlink(missing_ok=True)

    def _align(self, c):
        """One name list per source: sources handed over by the job (`later_examples`) have none."""
        while len(self.names) < len(c.spans):
            self.names.append([])

    def _earlier(self, index):
        return [i for names in self.names[:index] for i in names]

    # -- the pool
    def _grow(self, c):
        """Add the sets that now exist and have not been read as sources, then put the unseen sources in the order the weights do worst on."""
        import studio_model as sm
        self._align(c)
        for found in self.discover(set(self.known)):
            rows = self.loader(found["id"])
            train, held = split(rows)
            if not train:
                continue
            windows = c._windows_of(sm, train, c.tokenizer, c.cfg, c.directory / "windows" / f"source{len(c.spans)}")
            c.add_source(windows)
            self.names.append([found["id"]])
            self.held[found["id"]] = held
            self.known.add(found["id"])
        first = c.rules.source + 1
        if len(c.spans) - first > 1:
            keys = [names[0] if names else "" for names in self.names[first:]]
            named = [k for k in keys if k]
            place = {k: n for n, (k, _) in enumerate(rank(self._losses(c, named), named))}
            order = sorted(range(len(keys)), key=lambda j: place.get(keys[j], len(keys) + j))  # unnamed sources last, as handed over
            c.spans[first:], c.later[first - 1:], self.names[first:] = ([seq[start + j] for j in order] for seq, start in
                                                                         ((c.spans, first), (c.later, first - 1), (self.names, first)))

    # -- the coordinator's hooks
    def sources(self, c):
        """How many sources the rules may move through: not past the current one once the model stopped learning the newest text."""
        return c.rules.source + 1 if self.too_small else len(c.spans)

    def before_observe(self, c, row, verdict_for):
        """Called as a round closes, before its verdict. Measures the last switch once a round has been read on the new text, and, when
        the rules are about to move on (`verdict_for(sources)` is the verdict they would give with that many sources), reads the
        pool and orders it."""
        self._align(c)
        if c.finished:
            return
        if self.pending is not None and row["round"] > self.pending["round"] and self.pending.get("new_loss_after") is None:
            self._report_after(c, row)
        if self.pending is not None and self.pending.get("new_loss_after") is not None and not self.too_small:
            before, after = self.pending["new_loss_before"], self.pending["new_loss_after"]
            if before and after is not None and (before - after) / before < LEARNED:
                self.too_small = True
        pretend = verdict_for(len(c.spans) + (0 if self.too_small else 1))
        if self.too_small or pretend["action"] != "pivot" or pretend["name"] != "next_dataset":
            return
        self._grow(c)
        if len(c.spans) <= c.rules.source + 1:
            self.exhausted = True

    def after_observe(self, c, row, verdict):
        """Called after the verdict is applied: record a switch, or give a stop the reason that fits the policy."""
        if verdict["action"] == "pivot" and verdict["name"] == "next_dataset":
            self._record_switch(c, row, verdict)
        elif verdict["action"] == "stop" and verdict.get("rule") == "pivots" and (self.exhausted or self.too_small):
            extra = TOO_SMALL if self.too_small else NOTHING_LEFT
            c.rules.stop_text = f"{c.rules.stop_text} {extra}{SUGGESTION}"
        self._save(c)

    def _record_switch(self, c, row, verdict):
        index = c.rules.source
        names = self.names[index] if index < len(self.names) else []
        earlier = self._earlier(index)
        ids = [i for i in names if i] + [i for i in earlier if self.held.get(i)]
        losses = self._losses(c, ids) if ids else {}
        entry = {"round": row["round"], "step": row["step"], "from_sets": earlier, "to_sets": [i for i in names if i],
                 "why": (f"the best validation loss ({verdict['best']:.4g}) stopped improving"
                         + (f"; of the text it could read next it did worst on {names[0]}" if names and losses.get(names[0]) is not None else "")),
                 "old_loss_before": mean(losses, earlier), "new_loss_before": mean(losses, names), "replay": self.replay,
                 "old_loss_after": None, "new_loss_after": None}
        self.switches.append(entry)
        self.pending = entry
        what = ", ".join(entry["to_sets"]) or "the next supplied source"
        before = f"{entry['old_loss_before']:.4g}" if entry["old_loss_before"] is not None else "unmeasured"
        c._event("pivot", "", row["round"], f"Round {row['round']}: moving on to {what}, {entry['why']}. The loss on the earlier sets is {before}; it is "
                                             f"measured again after the next round, and {int(self.replay * 100)}% of each round replays them.")

    def _report_after(self, c, row):
        entry = self.pending
        ids = entry["from_sets"] + entry["to_sets"]
        losses = self._losses(c, [i for i in ids if self.held.get(i)])
        entry["old_loss_after"], entry["new_loss_after"] = mean(losses, entry["from_sets"]), mean(losses, entry["to_sets"])
        old = f"{entry['old_loss_after']:.4g}" if entry["old_loss_after"] is not None else "unmeasured"
        new = f"{entry['new_loss_after']:.4g}" if entry["new_loss_after"] is not None else "unmeasured"
        was = f"{entry['old_loss_before']:.4g}" if entry["old_loss_before"] is not None else "unmeasured"
        c._event("pivot", "", row["round"], f"Round {row['round']}: after a round on {', '.join(entry['to_sets']) or 'the new source'}, the loss on "
                                             f"the earlier sets is {old} (it was {was}); on the new text {new}.")
        self._save(c)

    def replay_ids(self, c, number):
        """Shard ids of the sources before the current one, enough that they are `replay` of the round's shards."""
        index = min(c.rules.source, len(c.spans) - 1)
        if index == 0 or not self.replay:
            return []
        pool = [shard for first, count in c.spans[:index] for shard in range(first, first + count)]
        count = min(len(pool), max(1, round(c.spans[index][1] * self.replay / (1 - self.replay))))
        return sorted(random.Random(c.cfg["seed"] * 1009 + number).sample(pool, count))

    def _save(self, c):
        write_journal(c.directory, {"sources": self.names, "switches": copy.deepcopy(self.switches), "exhausted": self.exhausted,
                                    "too_small": self.too_small})

    @classmethod
    def for_job(cls, params, services):
        """(policy, the run's training examples) for job params with `stages`: the chosen sets read one by one, each with its held-out
        slice kept out of training. `params["datasets"]` names the sets the run starts on."""
        spec = params["stages"] if isinstance(params["stages"], dict) else {}
        ids = list(params.get("datasets") or [s["id"] for s in text_sets(services.root)])
        rows, held = [], {}
        for source in ids:
            train, held[source] = split(services.examples({"datasets": [source], "heldout_fraction": params.get("heldout_fraction", .05)}))
            rows += train
        policy = cls(ids, lambda source: services.examples({"datasets": [source], "heldout_fraction": params.get("heldout_fraction", .05)}),
                     lambda skip: text_sets(services.root, skip), spec.get("scorer", "held_out"), spec.get("replay", REPLAY), held)
        return policy, rows
