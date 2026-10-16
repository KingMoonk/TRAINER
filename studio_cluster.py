"""Cluster: which machines there are, what a job needs, and which machine should run it. Pure Python: no Tk, no server.

Every API is a coordinator and a worker; a node alone is a cluster of one. The pieces here are what the queue will call
(placement is not wired in yet):

* the registry: nodes are listed by hand in the workspace config (`studioConfig.nodes` in data/workspace.json, tailnet names
  or URLs; absent means none), `own_info()` describes this machine and `poll()` asks the others for their `GET /node` payload;
* requirements: what a kind of job needs (`defaults_for`), as a device preference, a minimum of memory and a dtype;
* placement: `place()` picks the best free node that fits, or the node the user pinned.

Polling and the local probe are slow (network, a GPU check): call them from a worker, never the Tk thread.
"""
from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from studio_peer import PeerClient, base_url

DEFAULT_PORT = 8787  # the API's port when a node is listed as a bare name; studio_api (Phase 1) must serve on the same one
POLL_TIMEOUT = 2.  # seconds: a peer that does not answer this fast is offline for now
DEFAULT_TAG = "tag:trainer"  # the tag a node carries to call its peers (docs/cluster.md)
LOCAL = "local"  # the address of this machine's own NodeInfo

# dtypes a device kind trains in. Conservative: bfloat16 on Apple's GPU is not listed, because torch's Metal kernels cover it unevenly.
DTYPES = {"cuda": ("float32", "bfloat16", "float16"), "mps": ("float32", "float16"), "cpu": ("float32", "bfloat16")}


@dataclass(frozen=True)
class NodeInfo:
    """One machine, as `GET /node` reports it. `online` is False for a peer that did not answer; `error` then says why."""
    address: str  # how we reach it: the config entry, or LOCAL for this machine
    hostname: str = ""
    version: str = ""
    device: str = "cpu"  # "cuda" | "mps" | "cpu": the device jobs run on (the GPU when the probe found one)
    device_name: str = ""
    memory_bytes: int = 0  # video memory on CUDA; the computer's memory on MPS and CPU (they share it); 0 if unknown
    busy: bool = False
    dtypes: tuple = ("float32",)
    online: bool = True
    error: str = ""

    @property
    def free(self):
        return self.online and not self.busy

    def payload(self):
        """The body of `GET /node`: what a peer needs from this machine."""
        out = asdict(self)
        for key in ("address", "online", "error"):  # the asker knows the address and whether it answered
            del out[key]
        out["dtypes"] = list(self.dtypes)
        return out


def info_from_payload(address, payload):
    """A peer's NodeInfo from its `GET /node` JSON. Raises ValueError for a body that is not a node description."""
    if not isinstance(payload, dict) or not isinstance(payload.get("hostname"), str) or payload.get("device") not in DTYPES:
        raise ValueError("not a node description")
    memory = payload.get("memory_bytes", 0)
    dtypes = payload.get("dtypes")
    return NodeInfo(address=address, hostname=payload["hostname"], version=str(payload.get("version", "")), device=payload["device"],
                    device_name=str(payload.get("device_name", "")), memory_bytes=memory if isinstance(memory, int) and memory > 0 else 0,
                    busy=bool(payload.get("busy", False)),
                    dtypes=tuple(str(d) for d in dtypes) if isinstance(dtypes, list) and dtypes else ("float32",))


def offline(address, error):
    return NodeInfo(address=address, online=False, error=error)


# ---- the registry --------------------------------------------------------------------------------------------------

def configured_nodes(state):
    """The peers listed in the workspace config, in order and without duplicates. `state` is `Workspace.state`.

    `studioConfig.nodes` is a list of tailnet hostnames ("windows-box", "windows-box.tail1234.ts.net") or URLs
    ("http://100.64.0.7:8787"); absent or empty means this machine works alone."""
    config = state.get("studioConfig") if isinstance(state, dict) else None
    nodes = (config or {}).get("nodes", [])
    if not isinstance(nodes, list) or any(not isinstance(n, str) for n in nodes):
        raise ValueError("studioConfig.nodes in the workspace must be a list of machine names or URLs.")
    out = []
    for name in nodes:
        name = name.strip()
        if name and name not in out:
            out.append(name)
    return out


def node_url(address):
    """The `GET /node` URL of a configured entry: a URL as given (any path is replaced by /node, so https://box.ts.net stays on
    443 where `tailscale serve` listens), a bare name on DEFAULT_PORT."""
    base = base_url(address)
    if base.startswith("http://") and ":" not in base.partition("//")[2].rsplit("]", 1)[-1]:
        base += f":{DEFAULT_PORT}"  # plain http without a port is the API's own port; https is serve's 443
    return f"{base}/node"


def allowed_callers(state):
    """(tags, users) a direct caller may have, from `studioConfig.cluster` in the workspace config:
    `{"allowTags": ["tag:trainer"], "allowUsers": ["me@example.com"]}`. Absent allowTags means tag:trainer; absent allowUsers means
    nobody. Raises ValueError with a plain sentence for a bad value."""
    config = state.get("studioConfig") if isinstance(state, dict) else None
    cluster = (config or {}).get("cluster", {})
    if not isinstance(cluster, dict):
        raise ValueError('studioConfig.cluster in the workspace must be an object like {"allowTags": ["tag:trainer"], "allowUsers": []}.')
    out = []
    for key, default in (("allowTags", [DEFAULT_TAG]), ("allowUsers", [])):
        values = cluster.get(key, default)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"studioConfig.cluster.{key} in the workspace must be a list of non-empty text values.")
        values = [v.strip() for v in values]
        if key == "allowTags" and any(not v.startswith("tag:") for v in values):
            raise ValueError('studioConfig.cluster.allowTags entries must look like "tag:trainer".')
        out.append(tuple(values))
    return tuple(out)


def own_info(busy=False, version="", status=None, memory=None):
    """This machine's NodeInfo. `status` is `studio_backend.gpu_status()` and `memory` the computer's memory in bytes; both are
    probed when not given. `busy` is whether the queue is running a job; the caller knows, this module does not."""
    if status is None:
        try:
            from studio_backend import gpu_status
            status = gpu_status()
        except Exception as exc:  # torch missing or broken: a node without a usable GPU is still a node
            status = {"available": False, "kind": "", "name": "", "memory_bytes": 0, "reason": str(exc)}
    kind = status.get("kind") if status.get("available") else "cpu"
    if memory is None and kind != "cuda":
        try:
            from studio_passive import total_memory
            memory = total_memory()
        except Exception:
            memory = 0
    size = int(status.get("memory_bytes") or 0) if kind == "cuda" else int(memory or 0)
    return NodeInfo(address=LOCAL, hostname=socket.gethostname(), version=version, device=kind, device_name=status.get("name") or "",
                    memory_bytes=size, busy=bool(busy), dtypes=DTYPES[kind])


def fetch_json(url, timeout=POLL_TIMEOUT):
    """GET `url` (a configured entry's /node URL) through studio_peer.PeerClient. One attempt: a peer that does not answer is
    offline for now, and the next survey asks again."""
    base, _, path = url.rpartition("/")
    return PeerClient(base, timeout=timeout, retries=0).get_json(path)


def poll(addresses, fetch=None, timeout=POLL_TIMEOUT):
    """Ask each address for its `GET /node`; one NodeInfo per address, in order. Never raises: a peer that is unreachable, slow,
    or answers nonsense is reported offline with the reason. `fetch(url, timeout)` returns the parsed JSON (default: urllib).
    Peers are asked at the same time, so a dead one costs one timeout, not one each."""
    fetch = fetch or fetch_json

    def ask(address):
        try:
            return info_from_payload(address, fetch(node_url(address), timeout))
        except Exception as exc:
            return offline(address, f"{type(exc).__name__}: {exc}")

    addresses = list(addresses)
    if not addresses:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(addresses))) as pool:
        return list(pool.map(ask, addresses))


def survey(state, own, fetch=None, timeout=POLL_TIMEOUT):
    """The whole cluster: this node first, then every configured peer (a node alone is a cluster of one)."""
    return [own, *poll(configured_nodes(state), fetch, timeout)]


# ---- requirements --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirements:
    """What a job asks of a node. `device` is a preference ("cuda", "mps", "cpu" or "any"): a node of that kind is chosen over
    another that fits, but any node that fits is better than none. `min_memory_bytes` and `dtype` are hard: a node below the
    memory, or that does not train in the dtype, never gets the job. `most_memory` ranks by memory after the device."""
    device: str = "any"
    min_memory_bytes: int = 0
    dtype: str = ""  # "" = any
    most_memory: bool = False


ANY = Requirements()
# Queue kinds (studio_queue.KINDS) plus the long reads that become jobs with the API (rewrite.md, Phase 1).
DEFAULTS = {
    "pretrain": Requirements(device="cuda"),
    "post_round": Requirements(device="cuda"),  # RLHF
    "restore_answers": Requirements(device="cuda"),  # RLHF
    "pretrain_distributed": Requirements(device="cuda"),  # the owner: the coordinator plus one local member (studio_diloco)
    "diloco_member": ANY,
    "lora": ANY,
    "passive": ANY,
    "eval": ANY,
    "score": ANY,
    "benchmark": ANY,
    "datagen": ANY,
    "quicktest": ANY,
    "compare": ANY,
    "import_pretrained": ANY,
    "checkpoint_now": ANY,
    "inspect": Requirements(most_memory=True),
}


def defaults_for(kind):
    """The requirements a kind of job carries unless the user changes them."""
    try:
        return DEFAULTS[kind]
    except KeyError:
        raise ValueError(f"Unknown job kind: {kind}. Known: {', '.join(sorted(DEFAULTS))}.") from None


# ---- placement -----------------------------------------------------------------------------------------------------

def fits(requirements, node):
    """Whether `node` is online, free, has the memory and trains in the dtype."""
    return (node.free and node.memory_bytes >= requirements.min_memory_bytes
            and (not requirements.dtype or requirements.dtype in node.dtypes))


def matches(pinned, node):
    return bool(pinned) and pinned.strip().lower() in (node.address.lower(), node.hostname.lower())


def place(requirements, nodes, pinned=None):
    """The node a job should run on, or None when none fits (the job waits). Pure.

    pinned: the address or hostname the user chose. That node is returned if it fits (free, memory, dtype), else None: a pin
    is never overridden by a better node. Otherwise every node that fits is ranked by (device preference met, then the most
    memory when the job asks for it) and the first in `nodes` order wins a tie, so list this machine first to keep work local."""
    if pinned:
        return next((n for n in nodes if matches(pinned, n) and fits(requirements, n)), None)
    candidates = [n for n in nodes if fits(requirements, n)]
    if not candidates:
        return None
    wanted = requirements.device

    def rank(node):
        return (wanted in ("any", node.device), node.memory_bytes if requirements.most_memory else 0)

    return max(candidates, key=rank)  # max returns the first of equals
