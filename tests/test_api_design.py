"""The /design routes: every one answers over FakeServices, one result matches the function it wraps, and a spec the
maths cannot read is a 422 with a plain sentence, not a 500."""

from studio_ia import design as D
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

from tests.api_work import small_spec
from tests.sse_harness import ApiCase


class DesignRouterTests(ApiCase):
    prefix = "/design"

    def test_every_route_answers(self):
        base = small_spec()
        reg = self.call("GET", "/design/registry")
        self.assertEqual(reg["part_slots"], list(D.PART_SLOTS))
        self.assertEqual([s["slot"] for s in reg["slots"]], ["embedding", "position", "head", "norm", "mixer", "channel"])
        mixer = next(s for s in reg["slots"] if s["slot"] == "mixer")
        attention = next(m for m in mixer["modules"] if m["short"] == "attention")
        self.assertEqual(attention["tip"], D.module_tip("mixer", "attention"))
        self.assertEqual({f["name"] for f in attention["fields"]}, D.fields_of("mixer", "attention"))
        self.assertEqual((reg["words"], reg["heads"]), (list(D.WORDS), list(D.HEADS)))

        cur = self.call("POST", "/design/read", json={"spec": base})
        s = D.read_spec(base)
        self.assertEqual((cur["L"], cur["d"], cur["T"], cur["M"], cur["O"]), (s["L"], s["d"], s["T"], s["M"], []))

        more = self.call("POST", "/design/resize", json={"choices": cur, "n": 3})
        self.assertEqual(more["L"], 3)
        own = self.call("POST", "/design/choose", json={"choices": more, "slot": "norm", "short": "rmsnorm", "block": 1})
        self.assertEqual(own["O"], [{"block": 1, "slot": "norm", "id": "rmsnorm"}])
        every = self.call("POST", "/design/choose", json={"choices": own, "slot": "norm", "short": "rmsnorm"})
        self.assertEqual((every["M"]["norm"], every["O"]), ("rmsnorm", []))

        tower = self.call("POST", "/design/tower", json={"choices": own})
        self.assertEqual(len(tower["blocks"]), 3)
        self.assertEqual((tower["blocks"][1]["parts"]["norm"], tower["blocks"][1]["own"]), ("rmsnorm", ["norm"]))
        self.assertEqual(tower["blocks"][0]["own"], [])
        self.assertTrue(tower["knobs"]["H"])

        written = self.call("POST", "/design/write", json={"base": base, "choices": own, "was": cur})
        self.assertEqual(written, D.write_spec(base, D.read_spec(base) | dict(L=3, O={(1, "norm"): "rmsnorm"})))
        self.assertEqual(written["dims"]["n_layer"], 3)

        options = self.call("POST", "/design/part-options", json={"base": base, "choices": own, "slot": "position"})
        self.assertEqual([o["short"] for o in options], [m.short for m in D.registry.for_slot("position")])
        self.assertEqual(set(options[0]), {"short", "name", "blurb", "why"})

        c = self.call("POST", "/design/sizes", json={"spec": base, "v": s["V"]})
        want = D.sizes(base, s["V"])
        self.assertEqual({k: c[k] for k in want}, want)
        self.assertIn(c["fits"], (True, False, None))

        presets = self.call("GET", "/design/presets")
        self.assertEqual([p["key"] for p in presets], [key for key, _ in D.preset_specs()])
        self.assertTrue(all(p["term"] and p["name"] and p["spec"] for p in presets))

        cost = self.call("POST", "/design/preset-cost", json={"spec": presets[0]["spec"]})
        self.assertEqual(set(cost), {"params", "need", "fits", "rows", "step", "slow", "lines", "recipe"})
        self.assertIn("AdamW", cost["recipe"])

        lines = self.call("POST", "/design/compute-lines", json={"spec": base, "v": s["V"], "preview": None})
        self.assertEqual(lines, [{"text": D.T["compute_counting"], "bad": False}])
        preview = self.call("GET", "/pretraining/preview")
        lines = self.call("POST", "/design/compute-lines", json={"spec": base, "v": s["V"], "preview": preview})
        self.assertEqual(len(lines), 2)
        self.assertIn("steps", lines[1]["text"])

        self.assertEqual(self.call("POST", "/design/lr-text", json={"result": None}), {"text": ""})
        text = self.call("POST", "/design/lr-text", json={"result": {"error": "boom"}, "current": 0.001})["text"]
        self.assertEqual(text, D.lr_text({"error": "boom"}, 0.001))

        expect = self.call("POST", "/design/expectations", json={"cur": cur, "draft": own, "base": base, "draft_spec": written})
        self.assertTrue(expect["headline"])
        self.assertIn(D.EXPECT["more"], expect["lines"])

        same = self.call("POST", "/design/changes", json={"base": base, "draft_spec": base, "cur": cur, "draft": cur})
        self.assertEqual((same["parts_changed"], same["warm"]), (False, None))
        changed = self.call("POST", "/design/changes", json={"base": base, "draft_spec": written, "cur": cur, "draft": own})
        self.assertEqual(changed["chips"], D.change_chips(base, written, s, D.read_spec(written)))
        self.assertEqual((changed["parts_changed"], changed["warm"]["kind"]), (True, "other"))
        self.learning_routes()
        self.assertAllRoutesHit()

    def learning_routes(self):
        base = small_spec()
        cur = self.call("POST", "/design/read", json={"spec": base})
        draft = {**cur, "L": 4, "W": 2}
        new_spec = D.write_spec(base, {**D.read_spec(base), "L": 4, "W": 2})
        questions = self.call("POST", "/design/questions", json={"old_spec": base, "new_spec": new_spec})
        self.assertEqual([q["key"] for q in questions], D.prediction_questions(base, new_spec))
        self.assertEqual(questions[0]["options"], [{"value": "lower", "label": "Lower"}, {"value": "higher", "label": "Higher"}])

        self.services._versions.clear()
        old = self.services.save_version(base, "base")
        old.best_checkpoint = "parent/best.npz"
        old.scorecard = {"bpc": 2.4}
        body = {"base": base, "was": cur, "cur": cur, "draft": draft, "parent_version": old.number,
                "datasets": ["text"], "predictions": [{"error": "lower"}, {}]}
        with tempfile.TemporaryDirectory() as root:
            self.services.workspace = SimpleNamespace(run_dir=root)
            try:
                preview = self.call("POST", "/design/split-plan", json=body)
            finally:
                del self.services.workspace
        steps, jobs = preview["steps"], preview["jobs"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(len(jobs), 2)
        self.assertEqual([j["params"]["version"] for j in jobs], [2, 3])
        self.assertEqual(jobs[0]["params"]["init_from"], old.best_checkpoint)
        self.assertEqual(jobs[1]["params"]["init_from"], str(Path(jobs[0]["params"]["output_dir"]) / "best.npz"))
        self.assertEqual(jobs[1]["params"]["datasets"], ["text"])
        self.assertEqual([s["prediction"] for s in steps], [{"error": "lower"}, {}])
        self.assertEqual(len(self.services.versions()), 1)  # preview never saves or queues
        self.assertEqual(len(self.services.queue().up_next), 0)
        self.assertEqual(steps[-1]["spec"], new_spec)

        new = self.services.save_version(new_spec, "more blocks, wider", {"error": "lower"})
        new.scorecard = {"bpc": 2.2}
        speeds = {old.number: .2, new.number: .3}
        with mock.patch.object(self.services, "run_speed", side_effect=lambda number: speeds[number]):
            check = self.job("POST", "/design/guess-check", json={"old_version": old.number, "new_version": new.number})
            self.assertEqual(check["status"], "done", check)
            result = check["result"]
            self.assertEqual(result["check"], D.guess_check(old, new, speeds))
            self.assertEqual(result["check"]["hits"], 1)
            self.assertEqual(result["tray"], list(D.check_text(old, new, speeds)))
            self.assertTrue(result["results"]["step"]["measured"])
            notebook = self.job("POST", "/design/notebook")
            self.assertEqual(notebook["status"], "done", notebook)
            rows = D.notebook([old, new], speeds)
            self.assertEqual(notebook["result"], {"rows": rows, "lines": D.notebook_lines(rows)})

    def test_learning_routes_reject_bad_inputs_and_handle_empty_history(self):
        self.services._versions.clear()
        self.call("POST", "/design/questions", 422, json={"old_spec": {"dims": {"n_layer": "no"}}, "new_spec": {}})
        self.call("POST", "/design/guess-check", 404, json={"old_version": 1, "new_version": 2})
        self.call("POST", "/design/guess-check", 422, json={"old_version": 0, "new_version": 2})
        base = small_spec()
        cur = self.call("POST", "/design/read", json={"spec": base})
        body = {"base": base, "was": cur, "cur": cur, "draft": cur}
        self.assertEqual(self.call("POST", "/design/split-plan", json=body), {"steps": [], "jobs": []})
        self.call("POST", "/design/split-plan", 404, json={**body, "parent_version": 20})
        self.call("POST", "/design/split-plan", 422, json={**body, "predictions": [{"error": "maybe"}]})
        self.call("POST", "/design/split-plan", 422, json={**body, "predictions": [{"error": "lower"}]})
        result = self.job("POST", "/design/notebook")
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["result"], {"rows": [], "lines": D.notebook_lines([])})

    def test_bad_requests(self):
        bad = {"dims": {"n_layer": "no"}}
        for route in ("/design/read", "/design/preset-cost"):
            answer = self.call("POST", route, 422, json={"spec": bad})
            self.assertEqual(answer["detail"], "n_layer must be a whole number.")
        answer = self.call("POST", "/design/sizes", 422, json={"spec": {"block": 3}, "v": 0})
        self.assertEqual(answer["detail"], "block must be an object.")
        cur = self.call("POST", "/design/read", json={"spec": {}})
        answer = self.call("POST", "/design/write", 422, json={"base": {"block": {"mixer": {"id": "nope"}}}, "choices": cur})
        self.assertEqual(answer["detail"], "Unknown token mixer module: 'nope'.")
        self.call("POST", "/design/choose", 422, json={"choices": cur, "slot": "tokenizer", "short": "byte"})
        self.call("POST", "/design/read", 422, json={})
