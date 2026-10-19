"""Licence classes: what may be kept, and what the person has to opt in to.

Classes: permissive (MIT, Apache, CC BY, CC0 ...), share-alike (CC BY-SA, ODbL ...), copyleft (GPL family), non-commercial
(CC BY-NC, no-derivatives ...) and unknown (missing, "other", custom). Permissive and share-alike are kept by default; the
rest are dropped unless the person opts in to that class.
"""
from __future__ import annotations

import re

PERMISSIVE, SHARE_ALIKE, COPYLEFT, NON_COMMERCIAL, UNKNOWN = "permissive", "share-alike", "copyleft", "non-commercial", "unknown"
CLASSES = (PERMISSIVE, SHARE_ALIKE, COPYLEFT, NON_COMMERCIAL, UNKNOWN)  # from the most to the least free
DEFAULT_ALLOWED = (PERMISSIVE, SHARE_ALIKE)
OPT_IN = {NON_COMMERCIAL: "non_commercial", UNKNOWN: "unknown", COPYLEFT: "copyleft"}  # class -> the setting that allows it

_PERMISSIVE = {"mit", "mit-0", "apache-2.0", "apache-1.1", "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "isc", "cc0-1.0", "cc0",
               "unlicense", "wtfpl", "zlib", "bsl-1.0", "0bsd", "odc-by", "pddl", "afl-3.0", "artistic-2.0", "python-2.0", "ncsa",
               "cdla-permissive-1.0", "cdla-permissive-2.0", "public-domain", "pd", "ecl-2.0", "ofl-1.1", "postgresql",
               "cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0", "cc-by"}
_SHARE_ALIKE = {"odbl", "cdla-sharing-1.0", "gfdl", "gfdl-1.3", "mpl-2.0", "epl-1.0", "epl-2.0", "eupl-1.1", "eupl-1.2", "osl-3.0"}
_NAMES = {"apache license 2.0": "apache-2.0", "apache 2.0": "apache-2.0", "mit license": "mit", "creative commons attribution 4.0": "cc-by-4.0",
          "creative commons attribution-sharealike 4.0": "cc-by-sa-4.0", "cc by 4.0": "cc-by-4.0", "cc by-sa 4.0": "cc-by-sa-4.0",
          "cc by-sa 3.0": "cc-by-sa-3.0", "cc0 1.0": "cc0-1.0", "public domain": "public-domain", "bsd 3-clause": "bsd-3-clause",
          "bsd 2-clause": "bsd-2-clause", "gnu general public license v3.0": "gpl-3.0", "gnu general public license v2.0": "gpl-2.0"}
_UNKNOWN = {"other", "unknown", "none", "noassertion", "null", "unspecified", "n/a"}


def normalise(raw):
    """A licence id as lower-case SPDX-like text ("Apache License 2.0" -> "apache-2.0"); "" for a missing one."""
    text = str(raw or "").strip().lower()
    if text in _NAMES:
        return _NAMES[text]
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-license$", "", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def classify(raw):
    """(normalised id, class). A list of ids (a dataset card may name several) takes the most restrictive class."""
    if isinstance(raw, (list, tuple)):
        found = [classify(item) for item in raw if item]
        if not found:
            return "", UNKNOWN
        return "+".join(i for i, _ in found), max((c for _, c in found), key=CLASSES.index)
    ident = normalise(raw)
    if not ident or ident in _UNKNOWN or ident.startswith(("llama", "gemma", "openrail", "creativeml", "bigscience", "research")):
        return ident, UNKNOWN
    if re.search(r"(^|-)nc(-|$)", ident) or re.search(r"(^|-)nd(-|$)", ident) or ident.startswith(("non-commercial", "noncommercial")):
        return ident, NON_COMMERCIAL
    if ident in _SHARE_ALIKE or re.search(r"(^|-)sa(-|$)", ident):
        return ident, SHARE_ALIKE
    if ident.startswith(("gpl", "agpl", "lgpl")):
        return ident, COPYLEFT
    if ident in _PERMISSIVE or ident.startswith(("cc-by", "apache", "bsd", "mit")):
        return ident, PERMISSIVE
    return ident, UNKNOWN


def allowed(licence_class, opt_in):
    """Whether a class may be kept, given the person's opt-ins ({non_commercial, unknown, copyleft} booleans)."""
    if licence_class in DEFAULT_ALLOWED:
        return True
    return bool((opt_in or {}).get(OPT_IN.get(licence_class, ""), False))


def refusal(ident, licence_class):
    """The plain-words reason a licence is not kept."""
    if licence_class == UNKNOWN:
        return f"licence {ident or 'not stated'} is unknown"
    return f"licence {ident} is {licence_class}"
