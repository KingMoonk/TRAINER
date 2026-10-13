"""Post-training page: renders on FakeServices, a click on a round changes the card kicker, buttons enqueue post_round jobs."""
import tkinter as tk
import unittest
from tkinter import ttk

from studio_ia.copy import FEEDBACK as T
from studio_ia.services import FakeServices
from studio_ia.shell import Studio


def labels(widget):
    own = [str(widget.cget("text"))] if isinstance(widget, (tk.Label, ttk.Label)) else []
    return own + [t for child in widget.winfo_children() for t in labels(child)]


def find(widget, text):
    if isinstance(widget, (tk.Label, ttk.Label)) and str(widget.cget("text")) == text:
        return widget
    return next((w for child in widget.winfo_children() if (w := find(child, text))), None)


class PostTrainingTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServices()
        try:
            self.app = Studio(self.fake)
        except tk.TclError as error:
            self.skipTest(f"no display: {error}")
        self.addCleanup(self.app.destroy)
        self.page = self.app.show("post_training")
        self.app.update()

    def click(self, widget):
        widget.event_generate("<Button-1>")
        self.app.update()

    def test_renders_stage_line_chip_and_gate(self):
        text = labels(self.page)
        for want in (T["h1"], T["lead"], T["starts_from"], T["at_this_size"], T["click_round"], T["verdict_title"], T["chat_moves"],
                     T["held_back_check"], T["problems_check"], T["until_then"]):
            self.assertIn(want, text)
        for name in ("Practice answers", "Practice with checks", "Learn from a teacher", "Your votes"):
            self.assertIn(name, text)

    def test_clicking_a_round_changes_the_card_kicker(self):
        self.assertIn(T["round_1"], labels(self.page))   # nothing is done yet: the first round is open
        for kicker, name in ((T["round_3"], "Learn from a teacher"), (T["round_4"], "Your votes"), (T["round_2"], "Practice with checks")):
            self.click(find(self.page, name))
            text = labels(self.page)
            self.assertIn(kicker, text)
            self.assertEqual(sum(1 for t in text if t.startswith("ROUND ") and " · " in t), 1)

    def test_each_round_shows_its_visual(self):
        want = {"Practice answers": T["few_thousand"], "Practice with checks": T["rules_check"], "Learn from a teacher": T["teacher_note"],
                "Your votes": T["leash_note"]}
        for name, sentence in want.items():
            self.click(find(self.page, name))
            self.assertIn(sentence, labels(self.page), name)

    def test_live_events_replace_the_samples(self):
        self.click(find(self.page, "Practice with checks"))
        self.page.on_event("train", {"type": "group", "prompt": "Copy: yes", "rule": "copy", "mean": .5, "skipped": False,
                                     "answers": [{"text": "yes", "pass": True, "score": 1., "push": .5}, {"text": "no", "pass": False, "score": 0., "push": -.5}]})
        self.app.update()
        text = labels(self.page)
        self.assertIn("↑ +0.50 · pushed up", text)
        self.assertNotIn(T["illustrative"].upper(), text)
        self.page.on_event("train", {"type": "gate", "round": 2, "ok": False, "verdict": "v", "checks": [
            {"name": "a", "detail": "x", "ok": False}, {"name": "b", "detail": "y", "ok": True}]})
        self.app.update()
        text = labels(self.page)
        self.assertIn("RESULT · ROUND 2", [t.upper() for t in text])
        self.assertIn(T["result_kept"], text)

    def test_add_to_the_queue_enqueues_post_round_jobs(self):
        self.click(find(self.page, "Your votes"))
        button = next(w for w in self._walk(self.page) if isinstance(w, ttk.Button) and w.cget("text") == T["add_votes"])
        button.invoke()
        job = self.fake.queue().up_next[-1]
        self.assertEqual((job.kind, job.params["round"]), ("post_round", 4))
        self.assertEqual(job.params["beta"], .1)

    def _walk(self, widget):
        yield widget
        for child in widget.winfo_children():
            yield from self._walk(child)


if __name__ == "__main__":
    unittest.main()
