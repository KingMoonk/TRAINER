"""Test data stays out of training: rewrites share their source's group, and 13-gram overlap leaves items out of evaluation."""
import tempfile
import unittest

import studio_data_kinds as dk
import studio_decontam as decontam
from studio_catalog import _entry, convert_record
from studio_core import Workspace
from studio_api.routers.post_training import T as FEEDBACK
from studio_ia.services import WorkspaceServices
from studio_model import partition_examples

QUESTION = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends daily with four."
SOURCE = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends daily with four. How many eggs does she sell?"
REWRITE = "Janet's ducks lay 16 eggs per day, and she eats three for breakfast. Suppose she sells the rest at $2 each; how much does she make?"


def words(prefix, count):
    return " ".join(f"{prefix}{i}" for i in range(count))


def rows(parser, raw):
    return list(convert_record(raw, _entry(parser)))


class Groups(unittest.TestCase):
    def test_a_rewrite_and_its_source_share_a_group(self):
        original = rows("gsm8k", {"question": SOURCE, "answer": "16 - 3 = 13\n#### 13"})[0]
        rewrite = rows("metamathqa", {"type": "GSM_Rephrased", "query": REWRITE, "original_question": SOURCE, "response": "13. The answer is: 13"})[0]
        self.assertEqual(rewrite["groupId"], original["groupId"])

    def test_checkable_problems_share_the_group_too(self):
        original = rows("gsm8k_check", {"question": SOURCE, "answer": "#### 13"})[0]
        rewrite = rows("metamath_check", {"query": REWRITE, "original_question": SOURCE, "response": "The answer is: 13"})[0]
        conversation = rows("gsm8k", {"question": SOURCE, "answer": "#### 13"})[0]
        self.assertEqual({original["groupId"], rewrite["groupId"]}, {conversation["groupId"]})

    def test_the_group_ignores_case_punctuation_and_spacing(self):
        loud = rows("metamathqa", {"query": "x", "original_question": SOURCE.upper().replace(" ", "  "), "response": "y"})[0]
        self.assertEqual(loud["groupId"], rows("gsm8k", {"question": SOURCE, "answer": "#### 1"})[0]["groupId"])

    def test_different_questions_and_missing_sources(self):
        other = rows("gsm8k", {"question": "What is 2 + 3?", "answer": "#### 5"})[0]
        self.assertNotEqual(other["groupId"], rows("gsm8k", {"question": SOURCE, "answer": "#### 13"})[0]["groupId"])
        plain = rows("metamathqa", {"query": REWRITE, "response": "13"})[0]  # no original_question: the group is the query's own
        self.assertEqual(plain["groupId"], decontam.group_id(REWRITE))

    def test_a_split_never_separates_a_rewrite_from_its_source(self):
        import random
        examples = []
        for i in range(40):
            question = f"Question number {i} about {words('thing', 20)} {i}?"
            examples.append({**rows("gsm8k", {"question": question, "answer": f"#### {i}"})[0], "id": f"g{i}"})
            for kind in ("a", "b"):
                examples.append({**rows("metamathqa", {"query": f"{kind} {question}", "original_question": question, "response": f"{i}"})[0], "id": f"m{i}{kind}"})
        for seed in range(5):
            _, held, _ = partition_examples(examples, .3, random.Random(seed))
            sides = {}
            for row in examples:
                sides.setdefault(row["groupId"], set()).add(row["id"] in {h["id"] for h in held})
            self.assertTrue(all(len(side) == 1 for side in sides.values()))
            self.assertTrue(held)


class Overlap(unittest.TestCase):
    def test_text_sharing_thirteen_words_is_flagged(self):
        index = decontam.Index().add_all(["The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills today."])
        self.assertTrue(index.overlaps("Notice: THE quick, brown fox jumps over the lazy dog while the sun sets slowly -- and then?"))

    def test_clean_text_and_near_misses_pass(self):
        index = decontam.Index().add_all([SOURCE, words("train", 200)])
        self.assertFalse(index.overlaps(words("other", 100)))
        self.assertFalse(index.overlaps(" ".join(decontam.words(SOURCE)[:12])))  # twelve words are one short of a 13-gram
        self.assertFalse(index.overlaps(""))

    def test_a_short_item_flags_only_when_the_whole_item_appears(self):
        index = decontam.Index().add_all(["What is 2 + 3?", words("train", 50)])
        self.assertTrue(index.overlaps("what is 2 + 3"))
        self.assertFalse(index.overlaps("What is 2 + 4?"))

    def test_the_index_stays_bounded_and_says_it_sampled(self):
        index = decontam.Index(max_grams=500)
        index.add_all(words(f"doc{d}x", 300) for d in range(40))
        self.assertLessEqual(len(index._grams), 500)
        self.assertTrue(index.sampled and "1 in" in index.note)
        self.assertTrue(index.overlaps(words("doc7x", 300)))  # a long overlap is still found
        self.assertEqual(decontam.Index().note, "")

    def test_exclude_overlap_reports_what_it_left_out(self):
        index = decontam.Index().add_all([SOURCE])
        clean, report = decontam.exclude_overlap([{"q": REWRITE}, {"q": SOURCE + " Extra words."}, {"q": words("x", 40)}], index, lambda item: item["q"])
        self.assertEqual([item["q"] for item in clean], [REWRITE, words("x", 40)])
        self.assertEqual((report.checked, report.left_out), (3, 1))


class Excluded(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)

    def test_a_held_back_problem_that_appears_in_training_is_left_out(self):
        problems = self.ws.create_dataset("problems")
        dk.add_problems(self.ws, problems, [{"prompt": f"Problem {i}: {words(f'p{i}w', 30)}", "rule": "final_number", "params": {"answer": str(i)}, "answer": str(i)}
                                            for i in range(300)])
        held = self.svc.post_rows("heldout_problems")
        self.assertGreater(len(held), 2)
        self.assertEqual(self.svc.post_training(None)["overlap"]["problems"], {"checked": len(held), "left_out": 0, "note": ""})
        leaked = held[0]["prompt"]
        talk = self.ws.create_dataset("talk")
        for attempt in range(3):  # a copy that hashes into the held-back share is not trained on, so add a few
            self.ws.add(talk, [{"user": leaked, "assistant": f"copy {attempt}"}])
        kept = self.svc.post_rows("heldout_problems")
        self.assertEqual(len(kept), len(held) - 1)
        self.assertNotIn(leaked, [p["prompt"] for p in kept])
        self.assertEqual(self.svc.post_training(None)["overlap"]["problems"], {"checked": len(held), "left_out": 1, "note": ""})
        self.assertIn("also appear in training", FEEDBACK["overlap_left_out"])

    def test_a_held_back_text_that_appears_in_training_is_left_out(self):
        stories = self.ws.create_dataset("stories")
        base = [{"text": f"story {i} {words(f'w{i}x', 40)}"} for i in range(40)]
        self.ws.add(stories, base + [{"text": row["text"] + " and one more sentence"} for row in base[:20]])  # 20 near-copies
        train = self.svc.examples({"datasets": [stories], "heldout_fraction": .3})
        held = self.svc._heldout()
        self.assertTrue(held)
        self.assertFalse(set(held) & {e["text"] for e in train})
        report = self.svc.post_training(None)["overlap"]["text"]
        self.assertGreater(report["left_out"], 0)
        self.assertEqual(report["left_out"] + len(held), report["checked"])
        for text in held:  # nothing that is scored shares a 13-gram with what is trained on
            self.assertFalse(decontam.Index().add_all(e["text"] for e in train).overlaps(text))


class ChoiceRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Workspace(self.tmp.name)
        self.svc = WorkspaceServices(self.ws)
        self.addCleanup(self.svc.close)

    def _add(self, count, options, prefix="Q"):
        ds = self.ws.create_dataset(f"{prefix}{options}")
        rows = []
        for i in range(count):
            lines = "\n".join(f"{'ABCDEFGHIJ'[j]}) {prefix}{i} option {j}" for j in range(options))
            rows.append({"user": f"Answer the question. Reply with the letter and the option text.\n\n{prefix}{i} {words(f'{prefix}{i}w', 20)}?\n\n{lines}",
                         "assistant": "A) x", "metadata": {"rule": "choice", "params": {"answer": "A"}}})
        self.ws.add(ds, rows)

    def test_options_are_reshuffled_by_seed_with_the_key_following(self):
        self._add(120, 4)
        a, b = self.svc.post_rows("problems", seed=1), self.svc.post_rows("problems", seed=2)
        self.assertTrue(a)
        self.assertEqual(self.svc.post_rows("problems", seed=1), a)
        self.assertNotEqual([r["prompt"] for r in a], [r["prompt"] for r in b])
        for row in a:
            lines = row["prompt"].rpartition("\n\n")[2].split("\n")
            self.assertTrue(next(l for l in lines if l.startswith(row["params"]["answer"] + ")")).endswith("option 0"))
        self.assertGreater(len({r["params"]["answer"] for r in a}), 1)

    def test_two_option_sets_are_left_out_of_problems_unless_asked(self):
        self._add(120, 2, "B")
        self.assertEqual(self.svc.post_rows("problems"), [])
        self.assertTrue(self.svc.post_rows("problems", binary=True))
        self.assertTrue(self.svc.post_rows("heldout_problems"))

    def test_no_held_back_prompt_is_a_training_prompt(self):
        self._add(150, 4)
        held = self.svc.post_rows("heldout_problems")
        self.assertTrue(held)
        prompt = self.svc._kind_rows("problems", True)[0]["prompt"]
        talk = self.ws.create_dataset("talk")
        for attempt in range(4):  # copies hash to either side, so add a few until one is trained on
            self.ws.add(talk, [{"user": prompt, "assistant": f"copy {attempt}"}])
        trained = {r.get("prompt") or r.get("user") for r in self.svc._kind_rows("problems", False) + self.svc._kind_rows("conversations", False)}
        self.assertFalse({r["prompt"] for r in self.svc.post_rows("heldout_problems")} & trained)


if __name__ == "__main__":
    unittest.main()
