"""Record what the API answers over FakeServices as the fixtures of the web contract tests.

    python -m tests.web_fixtures           # rewrite web/src/contract/fixtures/*.json
    python -m tests.web_fixtures --check   # exit 1 if a committed fixture differs from what the API answers now

Each fixture is one request and the reply: {"request": {"method", "path", "body"}, "status", "body"}. Vitest (web/src/contract/)
feeds the replies through the web client and the pages, so a change to a response model that the client does not follow fails
there; tests/test_web_fixtures.py (part of `tasks.py test`, so part of CI) fails while a committed fixture is stale. Fixing
either is one command, the first above, followed by `pnpm test` in web/.

The steps run in order against one app, so later ones see the earlier ones' jobs. Anything that differs between runs or
machines (job ids, timestamps, the host's name and device) is replaced with a fixed value of the same type.
"""

import json
import re
import sys
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices

FIXTURES = Path(__file__).resolve().parent.parent / "web" / "src" / "contract" / "fixtures"
REGENERATE = "python -m tests.web_fixtures"

JOB_A = {"kind": "pretrain", "label": "Version 1 pretraining", "note": "First try", "params": {}}
JOB_B = {"kind": "pretrain", "label": "Version 2 pretraining", "note": "", "params": {}}
# (fixture name, method, path, body). `{job1}` in a path is the id of the first job enqueued, `{job2}` of the second.
STEPS = [
    ("queue-empty", "GET", "/queue", None),
    ("queue-next-step-empty", "GET", "/queue/next_step", None),
    ("queue-enqueue-first", "POST", "/queue/jobs", JOB_A),
    ("queue-enqueue-second", "POST", "/queue/jobs", JOB_B),
    ("queue-two-waiting", "GET", "/queue", None),
    ("queue-jobs", "GET", "/queue/jobs", None),
    ("queue-job-outlook", "GET", "/queue/jobs/{job2}/outlook", None),
    ("queue-outlook", "GET", "/queue/outlook", None),
    ("queue-hold", "POST", "/queue/jobs/{job2}/hold", {"held": True}),
    ("queue-move", "POST", "/queue/jobs/{job2}/move", {"index": 0}),
    ("queue-plan", "GET", "/queue/plan", None),
    ("queue-pause", "POST", "/queue/pause", None),
    ("queue-remove", "DELETE", "/queue/jobs/{job1}", None),
    ("queue-resume", "POST", "/queue/resume", None),
    ("jobs-empty", "GET", "/jobs", None),
    ("cluster-merged", "GET", "/cluster/merged", None),
    ("cluster-nodes", "GET", "/cluster/nodes", None),
    ("distributed-defaults", "GET", "/distributed/defaults", None),
    ("distributed-runs-empty", "GET", "/distributed/runs", None),
    ("node", "GET", "/node", None),
    ("traffic-rules", "GET", "/traffic/rules", None),
    ("storage", "GET", "/storage", None),
    ("storage-free", "GET", "/storage/free", None),
    ("storage-exports", "GET", "/storage/exports", None),
    ("data-has-text", "GET", "/data/has_text", None),
    ("data-drafts", "GET", "/data/drafts", None),
    ("data-problem-rules", "GET", "/data/problems/rules", None),
    ("data-add-text-row", "POST", "/data/text/rows", {"fields": {"text": "A first line of text.\nAnd a second."}}),
    ("data-text-rows", "GET", "/data/text/rows", None),
    ("data-conversation-rows", "GET", "/data/conversations/rows", None),
    ("pretraining-text-sets", "GET", "/pretraining/text_sets", None),
    ("pretraining-preview", "GET", "/pretraining/preview", None),
    ("post-training-options", "GET", "/post-training/options", None),
    ("post-training-import-models", "GET", "/post-training/import-models", None),
    ("quicktest", "GET", "/quicktest", None),
    ("training-run", "GET", "/training-run", None),
    ("versions", "GET", "/versions", None),
    ("versions-current", "GET", "/versions/current", None),
    ("versions-published-refused", "GET", "/versions/published", None),
    ("design-registry", "GET", "/design/registry", None),
    ("design-presets", "GET", "/design/presets", None),
    ("voice-status", "GET", "/voice", None),
    ("auto-runs-empty", "GET", "/auto-runs", None),
    ("auto-run-defaults", "GET", "/auto-runs/defaults?version=1", None),
]

# (fixture name, beat of the demo run's clock): the demo run's first round is about five beats, a second round follows at six
DEMO_MOMENTS = [("distributed-run-demo-training", 2.5), ("distributed-run-demo-collecting", 4.2), ("distributed-run-demo-second-round", 8.)]

HOST = {"hostname": "this-computer", "version": "v0.0.0", "device": "cpu", "device_name": "CPU", "memory_bytes": 17179869184, "dtypes": ["float32"]}
STAMP = 1700000000.0


def _steady(value, ids, host=HOST):
    """`value` with the parts that differ between runs and machines replaced by fixed ones of the same type (`host` holds the replacements for the machine's own facts)."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in host:
                out[key] = host[key]
            elif key.endswith("_at") and isinstance(item, (int, float)) and item > 1e9:
                out[key] = STAMP
            else:
                out[key] = _steady(item, ids, host)
        return out
    if isinstance(value, list):
        return [_steady(item, ids, host) for item in value]
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{8}", value):  # a job id
        return ids.setdefault(value, f"job{len(ids) + 1:04d}")
    return value


def record() -> dict[str, str]:
    """{fixture file name: its text}, from one pass of STEPS over a fresh FakeServices."""
    import studio_distributed

    kept =dict(studio_distributed.RUNS)  # runs another test left in the process are not part of a fresh app's answers
    studio_distributed.RUNS.clear()
    try:
        # The modern preset picks bfloat16 where this computer runs it; a recording must not depend on the GPU it was made on.
        cpu = mock.patch("studio_modules.precision.policy.likely_device", return_value="cpu")
        cpu.start()
        # /node reports this machine's device probes (speeds): the cache of whoever ran here, never part of the recorded answers
        with mock.patch("studio_probe.reported", return_value={}):
            services = FakeServices()
            client = TestClient(create_app(services, open=True, hosts=ANY))
            ids: dict[str, str] = {}
            jobs: list[str] = []
            out = {}
            for name, method, path, body in STEPS:
                sent = path.format(job1=jobs[0] if jobs else "", job2=jobs[1] if len(jobs) > 1 else "")
                response = client.request(method, sent, json=body)
                reply = response.json() if response.content else None
                if name.startswith("queue-enqueue"):
                    jobs.append(reply["id"])
                fixture = {"request": {"method": method, "path": path, "body": body}, "status": response.status_code, "body": reply}
                out[f"{name}.json"] = json.dumps(_steady(fixture, ids), indent=2, sort_keys=True) + "\n"
            out.update(_demo(client, services, ids))
    finally:
        cpu.stop()
        studio_distributed.RUNS.clear()
        studio_distributed.RUNS.update(kept)
    return out


class Beat:
    """A clock that stands still, in beats of one second."""

    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def _demo(client, services, ids):
    """The demo run (`studio_api --fake` opens on it) read at fixed moments of its first round, on a clock that does not move."""
    import studio_diloco
    import studio_distributed

    out, clock = {}, Beat(0.)
    run = services.demo_run(clock=clock)
    try:
        for name, now in DEMO_MOMENTS:
            clock.now = now
            response = client.get(f"/distributed/runs/{run.id}")
            fixture = {"request": {"method": "GET", "path": f"/distributed/runs/{run.id}", "body": None}, "status": response.status_code,
                       "body": response.json()}
            out[f"{name}.json"] = json.dumps(_steady(fixture, ids, host={}), indent=2, sort_keys=True) + "\n"  # host={}: the demo machines are the simulation's own, one CUDA and one MPS
    finally:
        studio_distributed.RUNS.pop(run.id, None)
        studio_diloco.COORDINATORS.pop(run.id, None)
    return out


def stale() -> list[str]:
    """The fixture files that are missing, differ from the API's answer, or no longer have a step."""
    recorded = record()
    problems = [f"{name}: missing" if not (FIXTURES / name).is_file() else f"{name}: differs"
                for name, text in recorded.items()
                if not (FIXTURES / name).is_file() or (FIXTURES / name).read_text(encoding="utf-8").replace("\r\n", "\n") != text]
    problems += [f"{path.name}: no longer recorded" for path in sorted(FIXTURES.glob("*.json")) if path.name not in recorded]
    return problems


def main(argv):
    if "--check" in argv:
        problems = stale()
        if problems:
            print("The web contract fixtures are stale:\n  " + "\n  ".join(problems) + f"\nRewrite them with:\n  {REGENERATE}\nthen run `pnpm test` in web/ and commit web/src/contract/fixtures/.")
            return 1
        print("web contract fixtures are current")
        return 0
    recorded = record()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for path in FIXTURES.glob("*.json"):
        if path.name not in recorded:
            path.unlink()
    for name, text in recorded.items():
        (FIXTURES / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(recorded)} fixtures to {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
