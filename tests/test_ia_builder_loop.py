"""Build your model's learning loop (#83): a guess before saving, checked once the version has trained; one change per version;
and the lab notebook of every version. Plain functions and page methods on stand-ins: no window opens."""
import json
import tempfile
import unittest
from pathlib import Path

import studio_registry as registry
from studio_core import Workspace
from studio_ia import design as D
from studio_ia.services import FakeServices, Version, WorkspaceServices
from studio_spec import default_spec
from studio_versions import VersionStore

CUR = D.read_spec(default_spec())


def spec(**choices):
    return D.write_spec(default_spec(), dict(CUR, **choices), CUR)


def deeper(n):
    return spec(**D.resize(CUR, n))


def version(number, s, bpc=None, prediction=None, parent=None, method=None, trained=True):
    card = {} if bpc is None else {"bpc": bpc, **({"bpc_method": method} if method else {})}
    return Version(number, spec=s, scorecard=card, prediction=prediction or {}, parent=parent,
                   best_checkpoint=f"/runs/{number}/best.npz" if trained else None)


def heads_free_mixer():
    """A look-back module without heads, so a heads change means nothing to it."""
    return next(m.short for m in registry.for_slot("mixer") if "heads" not in D.fields_of("mixer", m.short)
                and not D.why_not(default_spec(), CUR, CUR, "mixer", m.short))


class Stored(unittest.TestCase):
    def test_the_guess_is_kept_with_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VersionStore(Path(tmp) / "versions.json")
            store.save_version(deeper(2))
            row = store.save_version(deeper(4), "more blocks", {"error": "lower"})
            self.assertEqual(store.get(row["number"])["prediction"], {"error": "lower"})
            self.assertEqual(store.save_version(deeper(2))["prediction"], {})

    def test_services_hand_it_back_on_the_version(self):
        fake = FakeServices()
        self.assertEqual(fake.save_version(deeper(4), "x", {"step": "slower"}).prediction, {"step": "slower"})
        with tempfile.TemporaryDirectory() as tmp:
            svc = WorkspaceServices(Workspace(tmp))
            self.addCleanup(svc.close)
            svc.save_version(deeper(2), "first")
            svc.save_version(deeper(4), "more blocks", {"memory": "more"})
            self.assertEqual([v.prediction for v in svc.versions()], [{}, {"memory": "more"}])


class Speed(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Workspace(tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)

    def speed_of(self, rows):
        d = Path(self.ws.run_dir) / "a"
        d.mkdir(parents=True)
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.svc.store.save_version(deeper(2))
        self.svc.store.add_run(1, d)
        return self.svc.run_speed(1)

    def test_seconds_per_step_from_the_logged_speed(self):
        rows = [dict(step=0, tokens_per_second=0.)] + \
            [dict(step=s, tokens_per_second=t, trained_tokens=s * 1000) for s, t in ((5, 9000.), (10, 10000.), (15, 11000.))]
        self.assertAlmostEqual(self.speed_of(rows), 1000 / 10000)  # 1000 pieces a step at the median 10,000 a second

    def test_a_resumed_run_counts_only_its_own_steps(self):
        rows = [dict(step=s, tokens_per_second=10000., trained_tokens=(s - 100) * 500) for s in (105, 110, 120)]
        self.assertAlmostEqual(self.speed_of(rows), 500 / 10000)

    def test_no_run_no_speed(self):
        self.assertIsNone(self.svc.run_speed(1))
        self.assertIsNone(FakeServices().run_speed(1))


if __name__ == "__main__":
    unittest.main()
