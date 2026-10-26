"""studio_rules and studio_data_kinds: rule checks, generators, classification, near-duplicates, summaries."""
import itertools
import random
import tempfile
import time
import unittest

import studio_core
import studio_data_kinds as dk
import studio_rules as rules

CASES = {  # rule: (prompt, passing answer, failing answers)
    "copy": ("Copy: groundwork", "  groundwork\n", ["Groundwork", "groundwor", ""]),
    "reverse": ("Reverse: groundwork", " krowdnuorg ", ["groundwork", "Krowdnuorg"]),
    "sort": ("Sort these letters: d b a c", "a b c d", ["a b d c", "a b c", ""]),
    "count": ('How many r in "mirror"?', "There are 3.", ["2", "no idea"]),
    "small_addition": ("17 + 25 =", " 42 ", ["41", "seventeen", ""]),
    "max_length": ("Describe rain in under 20 characters.", "Soft grey water.", ["Water falling from the sky in long grey lines.", ""]),
    "ends_cleanly": ("Write a sentence.", 'He said "stop."', ["It was (not quite", "Ends with no stop", 'She said "hi.', ""]),
}


class Checks(unittest.TestCase):
    def test_every_rule_positive_and_negative(self):
        self.assertEqual(set(CASES), set(rules.RULES))
        for rule, (prompt, good, bad) in CASES.items():
            self.assertTrue(rules.check(rule, prompt, good), rule)
            for answer in bad:
                self.assertFalse(rules.check(rule, prompt, answer), (rule, answer))

    def test_unknown_rule_and_wrong_prompt_shape(self):
        with self.assertRaises(ValueError):
            rules.check("nope", "x", "y")
        self.assertFalse(rules.check("reverse", "no label here", "x"))

    def test_keyed_rules_compare_with_the_stored_key(self):
        number = {"answer": "1200"}
        for ok in ("1200", "So the total is 1,200.", "The answer is 1200.0"):
            self.assertTrue(rules.check("final_number", "q", ok, number), ok)
        for bad in ("120", "1200 then 3", "twelve hundred", ""):
            self.assertFalse(rules.check("final_number", "q", bad, number), bad)
        self.assertTrue(rules.check("final_number", "q", "about 0.5", {"answer": "0.5"}))
        for ok in ("B) water", "B", " (B) water", "B. water"):
            self.assertTrue(rules.check("choice", "q", ok, {"answer": "B"}), ok)
        for bad in ("A) rock", "water", "Because B", ""):
            self.assertFalse(rules.check("choice", "q", bad, {"answer": "B"}), bad)
        # no key stored: nothing can pass
        self.assertFalse(rules.check("final_number", "q", "3"))
        self.assertFalse(rules.check("choice", "q", "B"))
        self.assertEqual(set(rules.CHECKABLE), set(rules.RULES) | {"final_number", "choice"})

    def test_ends_cleanly_details(self):
        for ok in ("Done.", "Really?", "He said (quietly).", "Done.\n", "It's fine."):
            self.assertTrue(rules.check("ends_cleanly", "", ok), ok)
        for bad in ("Done", "Done,", "(Done.", "Say \"hi."):
            self.assertFalse(rules.check("ends_cleanly", "", bad), bad)


class Generators(unittest.TestCase):
    def test_round_trip(self):
        for rule in rules.RULES:
            items = rules.generate(rule, random.Random(3), 25)
            self.assertEqual(len(items), 25)
            for item in items:
                self.assertEqual(item["rule"], rule)
                self.assertIn("params", item)
                if "answer" in item:
                    self.assertTrue(rules.check(rule, item["prompt"], item["answer"]), item)
                else:
                    self.assertIn(rule, ("max_length", "ends_cleanly"))

    def test_deterministic_for_a_seed(self):
        self.assertEqual(rules.generate("sort", random.Random(5), 4), rules.generate("sort", random.Random(5), 4))


def workspace():
    return studio_core.Workspace(tempfile.mkdtemp())


class Kinds(unittest.TestCase):
    def test_classify(self):
        ws = workspace()
        d = ws.create_dataset("mixed")
        ws.add(d, ["a plain document about tides", {"user": "hi", "assistant": "hello"}])
        dk.add_problems(ws, d, rules.generate("reverse", random.Random(1), 2))
        got = sorted(dk.classify(e) for e in ws.examples(d))
        self.assertEqual(got, ["conversations", "problems", "problems", "text"])

    def test_problem_survives_edit(self):
        ws = workspace()
        d = ws.create_dataset("p")
        dk.add_problems(ws, d, rules.generate("count", random.Random(1), 1))
        example = ws.examples(d)[0]
        ws.update_example({**example, "quality": "Review"})
        self.assertEqual(dk.classify(ws.examples(d)[0]), "problems")
        self.assertEqual(dk.rule_of(ws.examples(d)[0]), "count")

    def test_keyless_problem_is_stored(self):
        ws = workspace()
        d = ws.create_dataset("p")
        self.assertEqual(dk.add_problems(ws, d, rules.generate("ends_cleanly", random.Random(1), 3)), (3, 0))
        self.assertEqual(dk.add_problems(ws, d, rules.generate("ends_cleanly", random.Random(1), 3)), (0, 3))

    def test_conversations_never_enter_pretraining(self):
        ws = workspace()
        a, b = ws.create_dataset("a"), ws.create_dataset("b")
        ws.add(a, ["one document", {"user": "q", "assistant": "r"}])
        dk.add_problems(ws, b, rules.generate("copy", random.Random(1), 2))
        texts = list(dk.pretraining_examples(ws, [a, b]))
        self.assertEqual([e["text"] for e in texts], ["one document"])

    def test_unreviewed_text_drafts_never_enter_pretraining(self):
        ws = workspace()
        a = ws.create_dataset("a")
        ws.add(a, ["a kept document", {"text": "a draft document", "quality": "Review", "synthetic": True}])
        self.assertEqual([e["text"] for e in dk.pretraining_examples(ws, [a])], ["a kept document"])

    def test_text_sample_reads_a_bounded_spread_and_sizes_from_counts(self):
        ws = workspace()
        disk, small = ws.create_dataset("disk", storage="sqlite"), ws.create_dataset("small")
        docs = [f"document {i:04d} " + "x" * 90 for i in range(400)]
        ws.add(disk, docs + [{"user": "q", "assistant": "a conversation, never text"}])
        ws.add(small, ["a short note kept whole"])
        sample, total = dk.text_sample(ws, [disk, small], budget=4000, exclude={docs[0]})
        self.assertEqual(total, sum(map(len, docs)) + len("q") + len("a conversation, never text") + len("a short note kept whole"))
        self.assertIn("a short note kept whole", sample)
        from_disk = [d for d in sample if d.startswith("document")]
        self.assertTrue(4 <= len(from_disk) <= 80, len(from_disk))  # about budget / 100 characters, not all 400
        self.assertGreater(int(from_disk[-1][9:13]), 300)  # spread across the set, not just its start
        self.assertNotIn(docs[0], sample)
        self.assertEqual(dk.summarize(ws, "text").characters, sum(map(len, docs)) + len("a short note kept whole"))  # counted once

    def test_text_characters_sums_stored_counts_without_sampling(self):
        ws = workspace()
        disk, small = ws.create_dataset("disk", storage="sqlite"), ws.create_dataset("small")
        docs = [f"document {i:04d} " + "x" * 90 for i in range(400)]
        ws.add(disk, docs)
        ws.add(small, ["a short note kept whole", "held back"])
        self.assertEqual(dk.text_characters(ws, [disk, small], exclude={"held back"}), sum(map(len, docs)) + len("a short note kept whole"))
        self.assertEqual(dk.text_characters(ws, []), 0)

    def test_text_characters_counts_only_training_text_of_disk_sets(self):
        ws = workspace()
        plain, heavy = ws.create_dataset("plain", storage="sqlite"), ws.create_dataset("heavy", storage="sqlite")
        docs = [f"document {i:03d} " + "x" * 40 for i in range(30)]
        ws.add(plain, docs)
        ws.add(heavy, [{"text": d, "metadata": {"note": "m" * 500}, "intent": "i" * 300} for d in docs]
               + [{"user": "q" * 200, "assistant": "a" * 200}, {"text": "an unreviewed draft", "quality": "Review", "synthetic": True}])
        dk.add_problems(ws, heavy, rules.generate("copy", random.Random(1), 3))
        self.assertEqual(dk.text_characters(ws, [heavy]), sum(map(len, docs)))
        self.assertEqual(dk.text_characters(ws, [plain]), dk.text_characters(ws, [heavy]))  # equal text, equal shares
        in_memory = ws.create_dataset("memory")
        ws.add(in_memory, docs + [{"user": "q" * 200, "assistant": "a" * 200}])
        self.assertEqual(dk.text_characters(ws, [in_memory]), dk.text_characters(ws, [heavy]))

    def test_text_characters_leaves_out_held_back_rows_of_disk_sets(self):
        ws = workspace()
        disk = ws.create_dataset("disk", storage="sqlite")
        docs = [f"document {i:03d} " + "x" * 40 for i in range(30)]
        ws.add(disk, docs)
        held = {docs[0], docs[7], "text that is in no set"}
        self.assertEqual(dk.text_characters(ws, [disk], exclude=held), sum(map(len, docs)) - len(docs[0]) - len(docs[7]))

    def test_interleaved_shares_follow_text_not_record_weight(self):
        ws = workspace()
        a, b = ws.create_dataset("a", storage="sqlite"), ws.create_dataset("b", storage="sqlite")
        docs = [f"document {i:03d} " + "x" * 40 for i in range(20)]
        ws.add(a, docs)
        ws.add(b, [{"text": d, "metadata": {"note": "m" * 5000}} for d in docs])
        shares = {i: dk.text_characters(ws, [i]) for i in (a, b)}
        self.assertEqual(shares[a], shares[b])
        first = list(itertools.islice(dk.interleaved_examples(ws, [a, b], shares), 20))
        self.assertEqual([sum(e["datasetId"] == i for e in first) for i in (a, b)], [10, 10])

    def test_sqlite_storage_problems(self):
        ws = workspace()
        d = ws.create_dataset("disk", storage="sqlite")
        dk.add_problems(ws, d, rules.generate("sort", random.Random(2), 3))
        self.assertEqual(ws.count(d), 3)
        self.assertEqual({dk.classify(e) for e in ws.examples(d)}, {"problems"})


STORY = "the old lighthouse keeper climbed the long spiral stairs every evening to light the great lamp above the grey sea"


class NearDuplicates(unittest.TestCase):
    def test_finds_near_copy_not_unrelated(self):
        edited = STORY.replace("grey", "gray") + " and then went home"
        other = "photosynthesis converts light energy into chemical energy stored in glucose inside the chloroplasts of green plants"
        found = dk.near_duplicates([("a", STORY), ("b", edited), ("c", other), ("d", STORY)])
        self.assertEqual(found.get("d"), "a")
        self.assertNotIn("c", found)
        self.assertNotIn("a", found)

    def test_threshold_boundary(self):
        base = " ".join(f"w{i}" for i in range(60))
        far = " ".join(f"w{i}" for i in range(30)) + " " + " ".join(f"x{i}" for i in range(30))
        self.assertEqual(dk.near_duplicates([("a", base), ("b", far)]), {})

    def test_cost_stays_bounded(self):
        docs = [(i, f"document number {i} " + " ".join(f"tok{(i * 7 + j) % 997}" for j in range(40))) for i in range(3000)]
        start = time.time()
        dk.near_duplicates(docs)
        self.assertLess(time.time() - start, 20)


class Summary(unittest.TestCase):
    def test_summaries_and_statuses(self):
        ws = workspace()
        t = ws.create_dataset("Stories")
        ws.add(t, [STORY, STORY + " today", "a different text about cooking pasta with garlic and olive oil slowly"])
        c = ws.create_dataset("Chats")
        ws.add(c, [{"user": "hi", "assistant": "hello", "quality": "Good"}, {"user": "yo", "assistant": "sup"}])
        ws.add(c, [{"user": "fix", "assistant": "fixed"}], source="chat")
        p = ws.create_dataset("Generated from rules")
        dk.add_problems(ws, p, rules.generate("small_addition", random.Random(1), 4))
        dk.add_problems(ws, p, rules.generate("copy", random.Random(1), 2), quality="Review")

        text = dk.summarize(ws, "text")
        self.assertEqual(text.count, 3)
        self.assertEqual([s["name"] for s in text.sets], ["Stories"])
        self.assertEqual(text.near_duplicates, 1)
        self.assertIn(text.rows[1]["status"], ("Near-duplicate", "Held back"))
        self.assertTrue(text.count_line.endswith("1 sets"))

        convo = dk.summarize(ws, "conversations")
        self.assertEqual(convo.count, 3)
        self.assertLessEqual({r["status"] for r in convo.rows}, {"Good", "Review", "From Chat", "Held back"})
        self.assertIn("From Chat", [r["status"] for r in convo.rows])

        prob = dk.summarize(ws, "problems")
        self.assertEqual(prob.count, 6)
        self.assertEqual(prob.rules, {"small_addition": 4, "copy": 2})
        self.assertEqual({r["status"] for r in prob.rows}, {"Ready", "Review"})
        self.assertEqual(len(dk.summarize(ws, "problems", query="copy").rows), 2)
        self.assertEqual(len(dk.summarize(ws, "problems", limit=1).rows), 1)


if __name__ == "__main__":
    unittest.main()
