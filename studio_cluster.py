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

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit

from studio_peer import PeerClient, PeerError, base_url, is_page

DEFAULT_PORT = 8787  # the API's port when a node is listed as a bare name; studio_api (Phase 1) must serve on the same one
GATEWAY_ERRORS = (502, 503, 504)  # what a proxy answers when the server behind it is not there
POLL_TIMEOUT = 2.  # seconds: a peer that does not answer this fast is slow; one slow SLOW_POLLS polls in a row is offline for now
SLOW_POLLS = 2  # a peer that answered before and is too slow this many polls in a row is offline; a busy API stalls now and then
DEFAULT_TAG = "tag:trainer"  # the tag a node carries to call its peers (docs/cluster.md)
LOCAL = "local"  # the address of this machine's own NodeInfo
SPEED_TIE = 1.05  # measured seconds within this ratio count as equal: the earlier node in the list keeps the job
MAX_NODES = 32  # peers in the list: each is polled on its own thread

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
    speeds: dict = field(default_factory=dict)  # design shape (studio_probe.shape_key) -> {device, seconds: {device: s}, tokens, age_s}: what this node measured

    def seconds_for(self, shape):
        """(device, seconds per step) this node measured for the design `shape`, or None when it has not (or reports none). The device
        is the one the node's own `auto` picks for it, which may be its CPU on a machine with a GPU."""
        hit = self.speeds.get(shape) if shape else None
        if not hit:
            return None
        return hit["device"], hit["seconds"][hit["device"]]

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
                    dtypes=tuple(str(d) for d in dtypes) if isinstance(dtypes, list) and dtypes else ("float32",),
                    speeds=clean_speeds(payload.get("speeds")))


def clean_speeds(value):
    """The `speeds` of a peer's report, keeping only well-formed entries (a peer may be a different version): a measured pick that has
    a positive seconds-per-step, and an age within studio_probe.MAX_AGE_S."""
    from studio_probe import MAX_AGE_S
    out = {}
    for shape, hit in (value.items() if isinstance(value, dict) else ()):
        try:
            seconds = {str(d): float(v) for d, v in hit["seconds"].items() if float(v) > 0}
            if hit["device"] in seconds and 0 <= float(hit["age_s"]) <= MAX_AGE_S:
                out[str(shape)] = {"device": hit["device"], "seconds": seconds, "tokens": int(hit["tokens"]), "age_s": float(hit["age_s"])}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


def offline(address, error, hostname=""):
    return NodeInfo(address=address, hostname=hostname, online=False, error=error)


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


def check_address(address):
    """A node address as typed, stripped, or a ValueError saying in plain words what is wrong. Accepts what `node_url` reaches: a
    machine name or IP ("windows-box", "100.64.0.7"), with a port ("windows-box:8787", "[fd7a::1]:8787") or an http(s) URL."""
    text = address.strip()

    def bad(why):
        return ValueError(f'"{text}" is not a machine address: {why}.')

    scheme, sep, rest = text.partition("://")
    if sep and scheme not in ("http", "https"):
        raise bad("a URL must start with http:// or https://")
    netloc = (rest if sep else text).split("/", 1)[0]
    if not netloc:
        raise bad("it has no machine name")
    if any(c.isspace() for c in text) or "@" in netloc or "?" in text or "#" in text:
        raise bad("write a name or a URL, without spaces, a login, a query or a fragment")
    if netloc.count(":") > 1 and not netloc.startswith("["):
        raise bad("put an IPv6 address in brackets, like [fd7a::1]:8787")
    try:
        parts = urlsplit(f"//{netloc}")
    except ValueError:
        raise bad("the brackets around the IPv6 address do not match") from None
    if not parts.hostname:
        raise bad("it has no machine name")
    try:
        port = parts.port
    except ValueError:
        port = 0
    if port == 0 or (port is None and netloc.rsplit("]", 1)[-1].endswith(":")):
        raise bad("the port must be a number from 1 to 65535")
    return text


def clean_nodes(addresses):
    """The list as `studioConfig.nodes` keeps it: each entry checked and stripped, blank and repeated ones dropped, in order. Raises
    ValueError with a sentence for a bad entry or for more than MAX_NODES."""
    out = []
    for address in addresses:
        if address.strip():
            address = check_address(address)
            if address not in out:
                out.append(address)
    if len(out) > MAX_NODES:
        raise ValueError(f"A cluster can list up to {MAX_NODES} other machines; this list has {len(out)}.")
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


FUNNEL_TOKEN_MIN = 32


def funnel_token(state):
    """The shared token `tailscale funnel` callers send as `X-Groundwork-Token`, from `studioConfig.cluster.funnelToken` in the
    workspace config; "" (no Funnel access) when absent. It must be printable ASCII (it travels in a header) and at least
    FUNNEL_TOKEN_MIN characters, as `openssl rand -hex 32` makes. Raises ValueError naming the setting, never the value."""
    config = state.get("studioConfig") if isinstance(state, dict) else None
    cluster = (config or {}).get("cluster", {})
    if not isinstance(cluster, dict) or "funnelToken" not in cluster:
        return ""
    token = cluster["funnelToken"]
    token = token.strip() if isinstance(token, str) else ""
    if len(token) < FUNNEL_TOKEN_MIN or not (token.isascii() and token.isprintable()):
        raise ValueError(f"studioConfig.cluster.funnelToken in the workspace must be at least {FUNNEL_TOKEN_MIN} printable ASCII "
                         "characters, such as the output of `openssl rand -hex 32` (or be left out).")
    return token


def own_info(busy=False, version="", status=None, memory=None, speeds=None):
    """This machine's NodeInfo. `status` is `studio_backend.gpu_status()` and `memory` the computer's memory in bytes; both are
    probed when not given. `speeds` is `studio_probe.reported(status)`, read from the cache when not given. `busy` is whether the queue is running a job; the caller knows, this module does not."""
    if status is None:
        try:
            from studio_backend import gpu_status
            status = gpu_status()
        except Exception as exc:  # torch missing or broken: a node without a usable GPU is still a node
            status = {"available": False, "kind": "", "name": "", "memory_bytes": 0, "reason": str(exc)}
    kind = status.get("kind") if status.get("available") else "cpu"
    if memory is None and kind != "cuda":
        try:
            from studio_memory import total_memory
            memory = total_memory()
        except Exception:
            memory = 0
    size = int(status.get("memory_bytes") or 0) if kind == "cuda" else int(memory or 0)
    if speeds is None:
        try:
            import studio_probe
            speeds = studio_probe.reported(status)
        except Exception:  # a probe cache that cannot be read is no speeds, not a node that cannot report
            speeds = {}
    return NodeInfo(address=LOCAL, hostname=socket.gethostname(), version=version, device=kind, device_name=status.get("name") or "",
                    memory_bytes=size, busy=bool(busy), dtypes=DTYPES[kind], speeds=speeds)


def fetch_json(url, timeout=POLL_TIMEOUT):
    """GET `url` (a configured entry's /node URL) through studio_peer.PeerClient. One attempt: a peer that does not answer is
    offline for now, and the next survey asks again."""
    base, _, path = url.rpartition("/")
    return PeerClient(base, timeout=timeout, retries=0).get_json(path)


def run_together(items, work, timeout=POLL_TIMEOUT):
    """`work(item)` for every item at the same time, one result per item in order, the whole call ending `timeout` seconds in.
    Never raises: a result is the return value, the exception `work` raised, or a TimeoutError for a call still running at the
    deadline (a socket timeout covers one read, not a slow drip or a name lookup; that thread is left to finish on its own)."""
    items = list(items)
    if not items:
        return []
    pool = ThreadPoolExecutor(max_workers=len(items))
    try:
        futures = [pool.submit(work, item) for item in items]
        wait(futures, timeout=timeout)
        return [(f.exception() or f.result()) if f.done() else TimeoutError() for f in futures]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


_ANSWERED = {}  # address -> (its last answer, slow polls since): what `poll` reports for a peer that is slow once
_ANSWERED_LOCK = threading.Lock()


def slow(exc):
    """Whether a failed ask failed only by taking too long (not refused, not a wrong answer): the peer may be up and busy."""
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, PeerError) and exc.status is None and "timed out" in str(exc).lower()


def poll(addresses, fetch=None, timeout=POLL_TIMEOUT):
    """Ask each address for its `GET /node`; one NodeInfo per address, in order. Never raises: a peer that is unreachable, slow,
    or answers nonsense is reported offline with the reason. `fetch(url, timeout)` returns the parsed JSON (default: urllib).
    Peers are asked at the same time, and the whole poll ends `timeout` seconds in: a peer that has not answered by then is slow. A
    peer that answered its last poll and is slow in fewer than SLOW_POLLS in a row keeps its last answer: one stall is not offline."""
    fetch = fetch or fetch_json
    addresses = list(addresses)
    answers = run_together(addresses, lambda a: info_from_payload(a, fetch(node_url(a), timeout)), timeout)
    out = []
    with _ANSWERED_LOCK:
        for address, answer in zip(addresses, answers):
            if not isinstance(answer, BaseException):
                _ANSWERED[address] = (answer, 0)
                out.append(answer)
                continue
            last, misses = _ANSWERED.pop(address, (None, 0))
            if last is not None and slow(answer) and misses + 1 < SLOW_POLLS:
                _ANSWERED[address] = (last, misses + 1)
                out.append(last)
            else:
                out.append(offline(address, failure(address, answer, timeout)))
    return out


def gateway_down(exc):
    """True for an empty gateway answer (502/503/504 with no body): a proxy such as `tailscale serve` is there but the app behind it is
    not, as while the app restarts. The app never had the request, so it is as unanswered as a refused connection."""
    return isinstance(exc, PeerError) and exc.status in GATEWAY_ERRORS and not exc.detail.strip()


def failure(address, exc, timeout=POLL_TIMEOUT):
    """Why a peer is offline, as a sentence for the person looking at the node list."""
    if isinstance(exc, PeerError) and exc.status is not None and is_page(exc.status, exc.detail):
        return (f"{address} answered with a web page, not the app's API: the app's API is not at that address. Check the address "
                f"(the machine's https://<name>.ts.net, or http://<machine>:{DEFAULT_PORT}) and that the app is running there.")
    if isinstance(exc, PeerError):
        text = f"{str(exc).rstrip('.')}."
        if gateway_down(exc):
            text += (f" Something answers at {address} (usually `tailscale serve`) but the app is not behind it: start the app on that "
                     f"machine, or list it by its direct address (http://<machine>:{DEFAULT_PORT}).")
        return text
    if isinstance(exc, TimeoutError):
        return f"{address} did not answer within {timeout:g} seconds."
    if isinstance(exc, ConnectionRefusedError):
        return f"{address} refused the connection: the app is not running there, or listens on another port."
    if isinstance(exc, json.JSONDecodeError) or "not a node description" in str(exc):
        return f"{address} answered, but not like a node of this app."
    return f"{address} could not be asked: {type(exc).__name__}: {exc}."


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
    "mlperf": ANY,  # a training-speed run (studio_mlperf); the device it times is named in its params
    "rlhf_loop": Requirements(device="cuda"),  # the loop (studio_loop); never moved whole: its folder is on its owner
    "pretrain_distributed": Requirements(device="cuda"),  # the owner: the coordinator plus one local member (studio_diloco)
    "diloco_member": ANY,
    "lora": ANY,
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


def pinned_nodes(pinned, nodes):
    """The machines a typed pin answers to: the one whose address it is, else every one with that hostname (case-insensitive). More
    than one means the pin is ambiguous (two machines can share a hostname, never an address): the job is not placed."""
    key = pinned.strip().lower() if pinned else ""
    if not key:
        return []
    return [n for n in nodes if n.address.lower() == key] or [n for n in nodes if n.hostname.lower() == key]


def place(requirements, nodes, pinned=None, shape=None):
    """The node a job should run on, or None when none fits (the job waits). Pure. See `place_why`."""
    return place_why(requirements, nodes, pinned, shape)[0]


def place_why(requirements, nodes, pinned=None, shape=None):
    """(node, why): the node a job should run on (None when none fits: the job waits) and, when its speed decided, a sentence saying so.

    pinned: the address or hostname the user chose. That node is returned if it fits (free, memory, dtype), else None: a pin
    is never overridden by a better node. A hostname that two nodes share names no node, so the job waits (see `pinned_nodes`).

    Otherwise every node that fits is ranked. `shape` is the job's design shape (`studio_probe.shape_key`): when every node that
    meets the device preference has measured it (`NodeInfo.speeds`), the fastest of them wins (the device a node picks for itself
    counts, so a GPU machine whose probe chose its CPU does not meet a "cuda" preference); on a tie, and whenever any of them has
    not measured it, the rule is the old one: the device preference, then the most memory when the job asks for it, and the first in
    `nodes` order, so list this machine first to keep work local. An unmeasured shape is therefore never guessed at."""
    if pinned:
        found = pinned_nodes(pinned, nodes)
        return (found[0] if len(found) == 1 and fits(requirements, found[0]) else None), ""
    candidates = [n for n in nodes if fits(requirements, n)]
    if not candidates:
        return None, ""
    wanted = requirements.device

    def device(node):
        measured = node.seconds_for(shape)
        return measured[0] if measured else node.device

    def preferred(node):
        return wanted in ("any", device(node))

    best = max(preferred(n) for n in candidates)
    tier = [n for n in candidates if preferred(n) == best]
    measured = {n.address: n.seconds_for(shape) for n in tier}
    if len(tier) > 1 and shape and all(measured.values()):
        fastest = min(m[1] for m in measured.values())
        tier = [n for n in tier if measured[n.address][1] <= fastest * SPEED_TIE]  # in `nodes` order
        winner = max(tier, key=lambda n: n.memory_bytes if requirements.most_memory else 0)
        slowest = max(m[1] for a, m in measured.items() if a != winner.address)
        mine = measured[winner.address]
        if slowest > mine[1] * SPEED_TIE:
            return winner, (f"{winner.hostname or winner.address} measured {mine[1]:.3g} s per step for this design on its "
                            f"{mine[0].upper()}, {slowest / mine[1]:.1f}x faster than the slowest free machine.")
        return winner, ""
    return max(tier, key=lambda n: n.memory_bytes if requirements.most_memory else 0), ""  # max returns the first of equals
