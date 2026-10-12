"""Bounded, reproducible mixtures for training on large local dataset libraries."""
from __future__ import annotations

import math
import random
import re

from studio_core import fingerprint, messages_of, normalize


def _quotas(total, weights, capacities):
    """Weighted allocation with capacity limits and deterministic integer rounding."""
    result = {key: 0 for key in weights}
    remaining = min(total, sum(capacities[key] for key in weights if weights[key] > 0))
    while remaining:
        active = [key for key in weights if weights[key] > 0 and result[key] < capacities[key]]
        if not active:
            break
        largest = max(weights[key] for key in active)
        normalized = {key: weights[key] / largest for key in active}
        weight_sum = sum(normalized.values())
        shares = {key: remaining * normalized[key] / weight_sum for key in active}
        increments = {key: min(capacities[key] - result[key], int(shares[key])) for key in active}
        assigned = sum(increments.values())
        for key, count in increments.items():
            result[key] += count
        remaining -= assigned
        # Distribute the rounding remainder in input order on equal fractions.
        if remaining and assigned == 0:
            for key in sorted(active, key=lambda item: shares[item], reverse=True):
                if remaining and result[key] < capacities[key]:
                    result[key] += 1
                    remaining -= 1
    return result


MAX_RECORDS = 1000000
BALANCES = ("records", "tokens")
_CALCULATOR = re.compile(r"<<.*?>>")


def clean_example(row):
    """Strip gsm8k-style <<a*b=c>> calculator annotations from assistant text, in place.

    Returns "cleaned" when text changed, "empty" when no assistant text is left (the row is unusable), else "".
    """
    changed = False
    for message in row.get("messages") or []:
        if message.get("role") == "assistant" and "<<" in message["content"]:
            text = _CALCULATOR.sub("", message["content"]).strip()
            changed = changed or text != message["content"]
            message["content"] = text
    if row.get("messages"):
        row["messages"] = [m for m in row["messages"] if m.get("role") != "assistant" or len(m["content"]) >= 1]
        if not any(m.get("role") == "assistant" for m in row["messages"]):
            return "empty"
        if changed:
            row["assistant"] = "\n".join(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return "cleaned" if changed else ""


def merge_duplicate_lineage(kept, duplicate):
    """Keep aliases of duplicate rows so their descendants cannot leak across splits."""
    provenance = dict(kept["provenance"]) if isinstance(kept.get("provenance"), dict) else {}
    other = duplicate["provenance"] if isinstance(duplicate.get("provenance"), dict) else {}

    def values(mapping, key):
        value = mapping.get(key, [])
        return [str(item) for item in value if isinstance(item, (str, int)) and item] if isinstance(value, (list, tuple)) else []

    source_ids = set(values(provenance, "source_ids"))
    source_ids.update(values(other, "source_ids"))
    if duplicate.get("id"):
        source_ids.add(str(duplicate["id"]))
    if source_ids:
        provenance["source_ids"] = sorted(source_ids)
    groups = set(values(provenance, "group_ids"))
    groups.update(values(other, "group_ids"))
    if duplicate.get("groupId"):
        groups.add(str(duplicate["groupId"]))
    if groups:
        provenance["group_ids"] = sorted(groups)
    if provenance:
        kept["provenance"] = provenance


def build_training_set(workspace, dataset_ids, max_records=20000, seed=42,
                       reviewed_only=False, weights=None, stop=None, emit=None,
                       balance="tokens", use_all=False, clean=True):
    """Select a bounded mixture without materializing complete dataset collections.

    Returns {"examples": [...], "report": {...}}; the report has per-source counts, "warnings", "cleaning".
    balance "tokens" (default) divides each source's weight by its average characters per record so sources
    contribute an equal share of text (weights still scale emphasis); "records" splits the budget by record
    count. use_all=True sets the budget to every eligible record (max_records is ignored, no 1,000,000 cap).
    reviewed_only keeps a source that has no reviewed rows (all of it, with a warning) instead of failing.
    clean=True strips <<calculator>> annotations from assistant text and drops emptied answers.
    A short source's quota is redistributed to other sources. Sampling examines at most twice the requested budget, then
    removes exact duplicates across sources. The report exposes any shortfall;
    it never silently grows the requested training budget to fill that shortfall.
    """
    emit = emit or (lambda event: None)

    def cancelled():
        if stop is not None and stop.is_set():
            raise InterruptedError("Training preparation cancelled.")

    cancelled()
    max_records = int(max_records)
    seed = int(seed)
    if balance not in BALANCES:
        raise ValueError("Balance must be records or tokens.")
    if not use_all and not 1 <= max_records <= MAX_RECORDS:
        raise ValueError("Training record budget must be between 1 and 1,000,000.")
    dataset_ids = list(dict.fromkeys(dataset_ids))
    if not dataset_ids:
        raise ValueError("Select at least one dataset for the training mixture.")
    datasets = {row["id"]: row for row in workspace.state["datasets"]}
    if any(key not in datasets for key in dataset_ids):
        raise ValueError("A selected dataset no longer exists. Refresh the dataset library.")
    weights = {key: float((weights or {}).get(key, 1)) for key in dataset_ids}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Dataset weights must be finite, nonnegative numbers.")
    if not any(weights.values()):
        raise ValueError("At least one dataset must have a weight greater than zero.")
    sources, warnings, strict = {}, [], {}
    for key in dataset_ids:
        cancelled()
        audit = workspace.audit(key)
        records = int(audit["records"])
        reviewed = max(0, records - int(audit.get("review", 0)))
        # A source with no reviewed rows is kept whole (with a warning) rather than failing the plan.
        strict[key] = bool(reviewed_only and reviewed > 0)
        eligible = reviewed if strict[key] else records
        name = datasets[key].get("name", key)
        if reviewed_only and records and not reviewed:
            warnings.append(f"{name} has no reviewed examples; all {records:,} of its examples are included.")
        characters = int(audit.get("characters") or 0)
        sources[key] = {"name": name, "records": records, "reviewed": reviewed, "available": eligible,
                        "average_characters": characters / records if records and characters else 0.,
                        "weight": weights[key], "requested": 0, "sampled": 0, "selected": 0, "duplicates": 0,
                        "invalid": 0, "cleaned": 0, "emptied": 0, "characters": 0,
                        "reviewed_fallback": bool(reviewed_only and records and not reviewed)}
    capacities = {key: info["available"] for key, info in sources.items()}
    if use_all:
        max_records = max(1, sum(capacities[key] for key in dataset_ids if weights[key] > 0))
    shares = dict(weights)
    balanced = balance == "tokens" and len(dataset_ids) > 1 and all(sources[key]["average_characters"] for key in dataset_ids)
    if balanced:
        mean = sum(sources[key]["average_characters"] for key in dataset_ids) / len(dataset_ids)
        shares = {key: weights[key] * mean / sources[key]["average_characters"] for key in dataset_ids}
    weights = shares
    requested = _quotas(max_records, weights, capacities)
    extras = _quotas(max_records, weights, {key: capacities[key] - requested[key] for key in dataset_ids})
    candidates = {}
    for index, key in enumerate(dataset_ids):
        cancelled()
        sources[key]["requested"] = requested[key]
        limit = requested[key] + extras[key]
        if limit:
            emit({"type": "info", "text": f"Sampling {sources[key]['name']}: up to {limit:,} records."})
            cancelled()
            candidates[key] = workspace.sample(key, limit=limit, seed=seed + index, reviewed_only=strict[key])
        else:
            candidates[key] = []
        sources[key]["sampled"] = len(candidates[key])
    cancelled()
    selected, seen = [], {}
    positions = {key: 0 for key in dataset_ids}

    def take(key, desired):
        added = 0
        while added < desired and positions[key] < len(candidates[key]):
            cancelled()
            original = candidates[key][positions[key]]
            positions[key] += 1
            row = normalize(original, dataset_id=key)
            if row is None or (strict[key] and row.get("quality") != "Good"):
                sources[key]["invalid"] += 1
                continue
            if clean:
                outcome = clean_example(row)
                if outcome == "empty":
                    sources[key]["emptied"] += 1
                    continue
                sources[key]["cleaned"] += outcome == "cleaned"
            content_hash = fingerprint(row)
            metadata = original if isinstance(original, dict) else {}
            row["id"] = str(metadata.get("id") or "ex_" + content_hash[:20])
            if metadata.get("updatedAt"):
                row["updatedAt"] = metadata["updatedAt"]
            if content_hash in seen:
                sources[key]["duplicates"] += 1
                merge_duplicate_lineage(seen[content_hash], row)
                continue
            seen[content_hash] = row
            selected.append(row)
            added += 1
            sources[key]["selected"] += 1
            sources[key]["characters"] += len(row["text"]) if row.get("text") else sum(len(m["content"]) for m in messages_of(row))

    for key in dataset_ids:
        take(key, requested[key])
    # Compensate for duplicates and invalid sampled rows, while staying within
    # the bounded candidate pool and respecting zero-weight exclusions.
    while len(selected) < max_records:
        remaining = {key: len(candidates[key]) - positions[key] for key in dataset_ids}
        extra = _quotas(max_records - len(selected), weights, remaining)
        if not any(extra.values()):
            break
        for key in dataset_ids:
            take(key, extra[key])
    if not selected:
        qualifier = " reviewed" if reviewed_only else " usable"
        raise ValueError(f"No{qualifier} examples were found in the selected datasets.")
    random.Random(seed).shuffle(selected)
    total_characters = sum(info["characters"] for info in sources.values()) or 1
    for info in sources.values():
        info["character_share"] = info["characters"] / total_characters
    report = {"requested": max_records, "selected": len(selected), "sources": sources, "warnings": warnings,
              "balance": balance if balanced else "records", "use_all": bool(use_all),
              "cleaning": {"enabled": bool(clean), "annotated_rows": sum(i["cleaned"] for i in sources.values()),
                           "empty_answers_dropped": sum(i["emptied"] for i in sources.values())},
              "duplicates": sum(info["duplicates"] for info in sources.values()),
              "invalid": sum(info["invalid"] for info in sources.values()),
              "characters": sum(info["characters"] for info in sources.values()),
              "sampled": sum(info["sampled"] for info in sources.values()),
              "reviewed_only": bool(reviewed_only), "seed": seed}
    for warning in warnings:
        emit({"type": "info", "text": "Warning: " + warning})
    if report["cleaning"]["annotated_rows"] or report["cleaning"]["empty_answers_dropped"]:
        emit({"type": "info", "text": f"Cleaned {report['cleaning']['annotated_rows']:,} answers (calculator markup); dropped {report['cleaning']['empty_answers_dropped']:,} empty."})
    emit({"type": "info", "text": f"Prepared {len(selected):,} examples from {sum(info['selected'] > 0 for info in sources.values())} datasets; removed {report['duplicates']:,} sampled duplicates."})
    if len(selected) < max_records:
        emit({"type": "info", "text": f"The bounded sample yielded {len(selected):,} of the requested {max_records:,} unique records."})
    return {"examples": selected, "report": report}
