"""Teacher-LLM synthesis with a mocked OllamaTeacher: every mode, bad JSON, grading, stop, safety of computed data."""
import re
import threading
import unittest
from unittest.mock import patch

import studio_teacher_synth as teacher_synth
from studio_core import normalize
from studio_synthesis import generate_batch, summarize_batch, validate_config


class FakeTeacher:
    """Routes by the system prompt; `handlers[kind](self, user)` overrides the default behaviour."""
    def __init__(self, **handlers):
        self.handlers, self.log, self.counter = handlers, [], 0

    @staticmethod
    def kind(system):
        for name, marker in [("grade", "strict reviewer"), ("dialogue", "short conversations"), ("evolve", "rewrite prompts"),
                             ("paraphrase", "rephrase requests"), ("answer", "ideal reply"), ("topic", "learns from scratch")]:
            if marker in system:
                return name
        raise AssertionError("unexpected system prompt: " + system[:80])

    def chat(self, system, user, json_mode=True, temperature=0.):
        kind = self.kind(system)
        self.log.append((kind, user, json_mode, temperature))
        if kind in self.handlers:
            return self.handlers[kind](self, user)
        return getattr(self, "default_" + kind)(user)

    def calls(self, kind):
        return [entry for entry in self.log if entry[0] == kind]

    def default_topic(self, user):
        items = []
        for _ in range(int(re.search(r"Write (\d+)", user).group(1))):
            self.counter += 1
            items.append({"prompt": f"Question number {self.counter} about cats?", "answer": f"**Answer** {self.counter} with `code`."})
        return {"items": items}

    def default_answer(self, user):
        return "Teacher: " + user.splitlines()[-1][:60]

    def default_evolve(self, user):
        return {"items": [{"id": int(i), "prompt": "Harder: " + p} for i, p in re.findall(r"id (\d+): operation = .*\nprompt: (.*)", user)]}

    def default_paraphrase(self, user):
        return {"items": [{"id": int(i), "prompt": "Please help me. " + p} for i, p in re.findall(r"^id (\d+): (.*)$", user, re.M)]}

    def default_dialogue(self, user):
        conversations, turns = int(re.search(r"Write (\d+) different", user).group(1)), int(re.search(r"exactly (\d+)", user).group(1))
        out = []
        for _ in range(conversations):
            self.counter += 1
            out.append({"turns": [{"user": f"Chat {self.counter} question {t}?", "assistant": f"Chat {self.counter} answer {t}."} for t in range(turns)]})
        return {"conversations": out}

    def default_grade(self, user):
        return {"grades": [{"id": int(i), "score": 2 if "BAD" in a else 9, "reason": "ok"}
                           for i, a in re.findall(r"id (\d+)\nPROMPT: .*\nANSWER: (.*)", user)]}


def run(config, teacher, **kwargs):
    base = {"teacher_model": "fake:1b", "seed": 3}
    return generate_batch({**base, **config}, teacher=teacher, **kwargs)


SEEDS = [{"id": f"s{i}", "user": f"What is {i} plus {i}?", "assistant": str(i * 2), "quality": "Good" if i % 2 else "Review"} for i in range(1, 9)]


class TeacherModes(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(teacher_synth, "RETRY_DELAY", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_topic_mode_rows_provenance_plain_text_and_rotation(self):
        fake = FakeTeacher()
        result = run({"mode": "topic", "count": 12, "topics": "cats\ndogs", "audience": "children", "batch_size": 4}, fake)
        self.assertEqual(len(result["rows"]), 12)
        self.assertEqual(result["stats"]["accepted"], 12)
        row = result["rows"][0]
        self.assertEqual(row["assistant"][:7], "Answer ")
        self.assertNotIn("*", row["assistant"])
        self.assertNotIn("`", row["assistant"])
        self.assertEqual(row["quality"], "Review")
        self.assertFalse(row["computedVerified"])
        provenance = row["provenance"]
        self.assertEqual((provenance["generator"], provenance["teacher_model"], provenance["mode"], provenance["template_id"]),
                         ("teacher-synthesis-v1", "fake:1b", "topic", "topic-v1"))
        self.assertEqual(provenance["batch_id"], result["batch_id"])
        self.assertEqual(provenance["recipe"], "topic")
        self.assertIn("teacher_model", row["qualityReport"])
        # task types rotate between calls, both topics are used, and later calls list earlier prompts as already used
        calls = fake.calls("topic")
        self.assertGreaterEqual(len(calls), 3)
        task_lines = {re.search(r"Task type: (.*?) \(adapt", user).group(1) for _, user, *_ in calls}
        self.assertEqual(len(task_lines), len(calls))
        self.assertTrue(any("cats" in user for _, user, *_ in calls) and any("dogs" in user for _, user, *_ in calls))
        self.assertIn("Question number 1 about cats?", calls[-1][1])
        self.assertIn("children", calls[0][1])

    def test_answer_length_cap_trims_at_sentence_or_rejects(self):
        long_sentence = "This is a sentence. " * 30
        no_stop = "word " * 100
        def items(fake, user):
            return {"items": [{"prompt": "Short one?", "answer": long_sentence}, {"prompt": "No stop?", "answer": no_stop},
                              {"prompt": "Fine?", "answer": "Yes."}]}
        result = run({"mode": "topic", "count": 2, "topics": "x", "batch_size": 3, "answer_length": "short"}, FakeTeacher(topic=items))
        by_prompt = {row["user"]: row for row in result["rows"]}
        self.assertLessEqual(len(by_prompt["Short one?"]["assistant"]), 220)
        self.assertTrue(by_prompt["Short one?"]["assistant"].endswith("sentence."))
        self.assertTrue(by_prompt["Short one?"]["provenance"]["derivation"]["trimmed"])
        self.assertNotIn("No stop?", by_prompt)
        self.assertEqual(result["stats"]["reasons"]["answer_too_long"], 1)
        self.assertEqual(result["stats"]["trimmed"], 1)
        # medium keeps a 330-character answer whole
        varied = " ".join(f"Sentence number {i} is here." for i in range(11))
        result = run({"mode": "topic", "count": 1, "topics": "x", "batch_size": 1, "answer_length": "medium"},
                     FakeTeacher(topic=lambda f, u: {"items": [{"prompt": "P?", "answer": varied}]}))
        self.assertEqual(result["rows"][0]["assistant"], varied)
        self.assertGreater(len(varied), 220)
        self.assertEqual(result["stats"]["trimmed"], 0)

    def test_refusals_echoes_and_long_prompts_are_filtered(self):
        def items(fake, user):
            return {"items": [{"prompt": "Tell me a secret?", "answer": "I'm sorry, but I cannot help with that."},
                              {"prompt": "Say hi", "answer": "say hi"}, {"prompt": "p" * 400, "answer": "ok"},
                              {"prompt": "Good one?", "answer": "Yes."}]}
        result = run({"mode": "topic", "count": 1, "topics": "x", "batch_size": 4}, FakeTeacher(topic=items))
        self.assertEqual([row["user"] for row in result["rows"]], ["Good one?"])
        self.assertEqual(result["stats"]["reasons"], {"refusal": 1, "echo": 1, "prompt_too_long": 1})

    def test_evolve_mutates_dataset_prompts_then_answers(self):
        fake = FakeTeacher()
        result = run({"mode": "evolve", "count": 6, "batch_size": 3}, fake, seeds=SEEDS, existing=SEEDS)
        self.assertEqual(len(result["rows"]), 6)
        for row in result["rows"]:
            self.assertTrue(row["user"].startswith("Harder: What is "))
            self.assertTrue(row["assistant"].startswith("Teacher: Harder:"))
            self.assertIn(row["provenance"]["derivation"]["operation"], teacher_synth.EVOLVE_OPS)
            self.assertEqual(len(row["provenance"]["source_ids"]), 1)
            self.assertTrue(row["provenance"]["source_ids"][0].startswith("s"))
            self.assertEqual(row["provenance"]["template_id"], "evolve-v1")
        self.assertEqual(len(fake.calls("evolve")), 2)
        self.assertEqual(len(fake.calls("answer")), 6)
        self.assertTrue(all(call[2] is False for call in fake.calls("answer")))      # answers are plain text, not JSON
        # needs something to evolve
        with self.assertRaises(ValueError):
            run({"mode": "evolve", "count": 3}, fake, seeds=[])

    def test_evolve_from_pasted_prompts_and_unchanged_prompts_rejected(self):
        keep = lambda fake, user: {"items": [{"id": 0, "prompt": "Explain gravity."}, {"id": 1, "prompt": "Harder: Why is the sky blue?"}]}
        result = run({"mode": "evolve", "count": 1, "batch_size": 2, "prompts": "Explain gravity.\nWhy is the sky blue?"},
                     FakeTeacher(evolve=keep), seeds=[])
        self.assertEqual(result["stats"]["accepted"], 1)
        self.assertEqual(result["rows"][0]["user"], "Harder: Why is the sky blue?")
        self.assertEqual(result["stats"]["reasons"].get("unchanged"), 1)

    def test_answer_mode_distils_without_prompt_conflicts_and_skips_computed(self):
        seeds = [dict(row) for row in SEEDS]
        seeds[0]["computedVerified"] = True
        fake = FakeTeacher()
        # existing contains the very same prompts with different answers: answering them is not a conflict
        result = run({"mode": "answer", "count": 20, "batch_size": 4}, fake, seeds=seeds, existing=seeds)
        self.assertEqual(len(result["rows"]), 7)                       # 8 sources minus the computed one
        self.assertEqual(result["stats"]["skipped_computed"], 1)
        self.assertEqual(result["stats"]["exhausted_families"], ["answer"])
        self.assertNotIn("s1", {sid for row in result["rows"] for sid in row["provenance"]["source_ids"]})
        self.assertEqual(result["stats"]["reasons"].get("prompt_conflict", 0), 0)
        # filter: only rows that are not yet Good
        result = run({"mode": "answer", "count": 20, "answer_filter": "review"}, FakeTeacher(), seeds=SEEDS)
        self.assertEqual(len(result["rows"]), 4)
        self.assertTrue(all(row["provenance"]["derivation"]["source_quality"] != "Good" for row in result["rows"]))
        # filter: only missing answers (pasted prompts)
        result = run({"mode": "answer", "count": 20, "answer_filter": "missing", "prompts": "Name a prime.\nName a colour."}, FakeTeacher(), seeds=SEEDS)
        self.assertEqual({row["user"] for row in result["rows"]}, {"Name a prime.", "Name a colour."})

    def test_paraphrase_keeps_exact_answers_and_computed_flag(self):
        seeds = [{"id": f"c{i}", "user": f"Calculate {i} + {i * 3}. Return only the integer.", "assistant": str(i * 4),
                  "computedVerified": True, "quality": "Review"} for i in range(1, 6)]
        fake = FakeTeacher()
        result = run({"mode": "paraphrase", "count": 5, "batch_size": 5}, fake, seeds=seeds, existing=seeds)
        self.assertEqual(len(result["rows"]), 5)
        originals = {s["id"]: s for s in seeds}
        for row in result["rows"]:
            source = originals[row["provenance"]["source_ids"][0]]
            self.assertEqual(row["assistant"], source["assistant"])
            self.assertNotEqual(row["user"], source["user"])
            self.assertTrue(row["user"].startswith("Please help me."))
            self.assertTrue(row["computedVerified"])
            self.assertTrue(row["qualityReport"]["computed_target"])
        self.assertEqual(fake.calls("grade"), [])
        # changed numbers, unchanged text, and empty rewrites are refused
        def bad(fake, user):
            listed = dict(re.findall(r"^id (\d+): (.*)$", user, re.M))
            return {"items": [{"id": 0, "prompt": listed["0"] + " Add 99."}, {"id": 1, "prompt": listed["1"]}, {"id": 2, "prompt": ""}]}
        result = run({"mode": "paraphrase", "count": 3, "batch_size": 3, "attempt_multiplier": 1}, FakeTeacher(paraphrase=bad), seeds=seeds[:3])
        reasons = result["stats"]["reasons"]
        self.assertEqual(result["stats"]["accepted"], 0)
        self.assertEqual(reasons.get("meaning_changed"), 1)
        self.assertEqual(reasons.get("unchanged"), 1)
        self.assertEqual(reasons.get("malformed_item"), 1)

    def test_paraphrase_never_alters_unicode_or_markdown_answers(self):
        answer = "**Bold** `code` — café\n- item"
        seed = [{"id": "x", "user": "Say it in 3 words.", "assistant": answer}]
        result = run({"mode": "paraphrase", "count": 1, "plain_text": True}, FakeTeacher(), seeds=seed)
        self.assertEqual(result["rows"][0]["assistant"], answer)

    def test_dialogue_mode_builds_alternating_turns(self):
        fake = FakeTeacher()
        result = run({"mode": "dialogue", "count": 5, "topics": "weather", "min_turns": 2, "max_turns": 3}, fake)
        self.assertEqual(len(result["rows"]), 5)
        for row in result["rows"]:
            roles = [m["role"] for m in row["messages"]]
            self.assertEqual(roles[-1], "assistant")
            self.assertIn(len(roles), (4, 6))
            self.assertEqual(roles, ["user", "assistant"] * (len(roles) // 2))
            self.assertEqual(row["provenance"]["template_id"], "dialogue-v1")
        # one-turn "conversations" are rejected as the wrong shape
        short = lambda f, u: {"conversations": [{"turns": [{"user": "Hi?", "assistant": "Hello."}]}]}
        result = run({"mode": "dialogue", "count": 1, "topics": "x", "attempt_multiplier": 1}, FakeTeacher(dialogue=short))
        self.assertEqual(result["stats"]["accepted"], 0)
        self.assertEqual(result["stats"]["reasons"]["bad_shape"], 1)
        # a system prompt is added to the student's rows, never to the teacher request
        result = run({"mode": "dialogue", "count": 1, "topics": "x", "system_prompt": "Be brief."}, FakeTeacher())
        self.assertEqual(result["rows"][0]["messages"][0], {"role": "system", "content": "Be brief."})


class TeacherRobustness(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(teacher_synth, "RETRY_DELAY", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_malformed_json_is_retried_skipped_and_counted(self):
        state = {"calls": 0}
        def flaky(fake, user):
            state["calls"] += 1
            if state["calls"] <= 3:          # the first request fails all 1 + 2 attempts
                raise RuntimeError("The teacher did not return JSON: oops")
            if state["calls"] == 4:          # then one retry-able blip
                raise RuntimeError("temporary")
            return FakeTeacher.default_topic(fake, user)
        result = run({"mode": "topic", "count": 5, "topics": "x", "batch_size": 5}, FakeTeacher(topic=flaky))
        stats = result["stats"]
        self.assertEqual(stats["accepted"], 5)
        self.assertEqual(stats["teacher_errors"], 1)
        self.assertEqual(stats["teacher_retries"], 3)
        self.assertIn("failed on 1 request", summarize_batch(result))

    def test_malformed_items_are_counted_not_fatal(self):
        items = lambda f, u: {"items": [{"prompt": "no answer here"}, "junk", {"answer": "orphan"}, {"prompt": "Okay?", "answer": "Yes."}]}
        result = run({"mode": "topic", "count": 1, "topics": "x"}, FakeTeacher(topic=items))
        self.assertEqual(result["stats"]["accepted"], 1)
        self.assertEqual(result["stats"]["malformed"], 2)

    def test_alternative_json_shapes_are_accepted(self):
        shapes = [lambda f, u: [{"instruction": "Q one?", "output": "A one."}],
                  lambda f, u: {"data": [{"question": "Q one?", "response": "A one."}]},
                  lambda f, u: {"prompt": "Q one?", "answer": "A one."}]
        for shape in shapes:
            result = run({"mode": "topic", "count": 1, "topics": "x", "batch_size": 1}, FakeTeacher(topic=shape))
            self.assertEqual([(r["user"], r["assistant"]) for r in result["rows"]], [("Q one?", "A one.")])

    def test_unreachable_teacher_raises_a_plain_error(self):
        def down(fake, user):
            raise RuntimeError("Cannot reach Ollama at http://localhost:1. Is it running? (refused)")
        with self.assertRaisesRegex(ValueError, "Cannot reach Ollama"):
            run({"mode": "topic", "count": 3, "topics": "x"}, FakeTeacher(topic=down))
        with self.assertRaises(ValueError):                    # model must be chosen when no teacher object is injected
            generate_batch({"mode": "topic", "count": 1, "topics": "x"})

    def test_persistent_malformed_replies_stop_instead_of_looping(self):
        calls = {"n": 0}
        def junk(fake, user):
            calls["n"] += 1
            raise RuntimeError("The teacher did not return JSON: ???")
        with self.assertRaisesRegex(ValueError, "could not be used"):
            run({"mode": "topic", "count": 50, "topics": "x"}, FakeTeacher(topic=junk))
        self.assertEqual(calls["n"], 4 * 3)                   # 4 failed requests, 3 attempts each

    def test_stop_between_calls_keeps_partial_batch(self):
        stop = threading.Event()
        def first_then_stop(fake, user):
            reply = FakeTeacher.default_topic(fake, user)
            stop.set()
            return reply
        fake = FakeTeacher(topic=first_then_stop)
        result = run({"mode": "topic", "count": 40, "topics": "x", "batch_size": 5}, fake, stop=stop)
        self.assertTrue(result["stats"]["cancelled"])
        self.assertEqual(len(fake.calls("topic")), 1)
        self.assertEqual(len(result["rows"]), 5)
        self.assertIn("Stopped early", summarize_batch(result))
        # already stopped: no teacher calls at all
        fake = FakeTeacher()
        stopped = threading.Event()
        stopped.set()
        result = run({"mode": "topic", "count": 5, "topics": "x"}, fake, stop=stopped)
        self.assertEqual((result["rows"], fake.log), ([], []))

    def test_stop_inside_evolve_answer_loop(self):
        stop = threading.Event()
        def answer_then_stop(fake, user):
            stop.set()
            return "An answer."
        fake = FakeTeacher(answer=answer_then_stop)
        result = run({"mode": "evolve", "count": 6, "batch_size": 6}, fake, seeds=SEEDS, stop=stop)
        self.assertEqual(len(fake.calls("answer")), 1)
        self.assertTrue(result["stats"]["cancelled"])

    def test_progress_events_and_result_shape(self):
        events = []
        result = run({"mode": "topic", "count": 4, "topics": "x", "batch_size": 2}, FakeTeacher(), emit=events.append)
        self.assertEqual(set(result), {"rows", "rejected", "stats", "config", "batch_id"})
        self.assertTrue(events and all(e["type"] == "progress" for e in events))
        self.assertEqual(events[-1]["completed"], 4)
        for key in ("requested", "attempted", "accepted", "duplicates", "conflicts", "rejected", "reasons", "coverage",
                    "shortfall", "cancelled", "teacher_errors", "teacher_calls"):
            self.assertIn(key, result["stats"])

    def test_duplicates_and_existing_rows_are_filtered(self):
        existing = [normalize({"user": "Question number 1 about cats?", "assistant": "Answer 1 with code."})]
        result = run({"mode": "topic", "count": 3, "topics": "x", "batch_size": 3}, FakeTeacher(), existing=existing)
        self.assertEqual(result["stats"]["reasons"].get("duplicate"), 1)
        self.assertEqual(result["stats"]["accepted"], 3)
        self.assertIn("1 duplicates", summarize_batch(result))


class SelfGrading(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(teacher_synth, "RETRY_DELAY", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_threshold_filters_and_scores_are_stored(self):
        def items(fake, user):
            out = []
            for _ in range(int(re.search(r"Write (\d+)", user).group(1))):
                fake.counter += 1
                out.append({"prompt": f"Q{fake.counter}?", "answer": ("BAD " if fake.counter % 2 == 0 else "") + f"A{fake.counter}."})
            return {"items": out}
        fake = FakeTeacher(topic=items)
        result = run({"mode": "topic", "count": 4, "topics": "x", "batch_size": 4, "self_grade": True, "grade_threshold": 6}, fake)
        self.assertEqual(result["stats"]["accepted"], 4)
        self.assertTrue(all("BAD" not in row["assistant"] for row in result["rows"]))
        for row in result["rows"]:
            self.assertEqual(row["qualityReport"]["teacher_score"], 9.)
            self.assertEqual(row["qualityReport"]["teacher_rationale"], "ok")
        self.assertGreater(result["stats"]["reasons"]["low_score"], 0)
        self.assertEqual(result["stats"]["mean_score"] is not None, True)
        self.assertIn("low teacher score", summarize_batch(result))
        # one grading call covers a small batch of candidates
        self.assertLessEqual(len(fake.calls("grade")), len(fake.calls("topic")) * 2)

    def test_grading_disabled_makes_no_grade_calls_and_failure_fails_open(self):
        fake = FakeTeacher()
        run({"mode": "topic", "count": 3, "topics": "x"}, fake)
        self.assertEqual(fake.calls("grade"), [])
        broken = FakeTeacher(grade=lambda f, u: (_ for _ in ()).throw(RuntimeError("boom")))
        result = run({"mode": "topic", "count": 3, "topics": "x", "self_grade": True}, broken)
        self.assertEqual(result["stats"]["accepted"], 3)
        self.assertGreater(result["stats"]["grade_errors"], 0)
        self.assertNotIn("teacher_score", result["rows"][0]["qualityReport"])
        self.assertIn("Self-grade unavailable for this example.", result["rows"][0]["qualityReport"]["warnings"])

    def test_computed_answers_are_never_graded_or_changed(self):
        seeds = [{"id": f"c{i}", "user": f"Calculate {i} * 2. Return only the integer.", "assistant": str(i * 2), "computedVerified": True} for i in range(1, 4)]
        fake = FakeTeacher(grade=lambda f, u: {"grades": []})
        result = run({"mode": "paraphrase", "count": 3, "self_grade": True, "grade_threshold": 10}, fake, seeds=seeds)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(fake.calls("grade"), [])
        self.assertTrue(all(row["computedVerified"] for row in result["rows"]))


class Wiring(unittest.TestCase):
    def test_default_teacher_is_ollama_with_url_and_model(self):
        seen = []
        def fake_chat(self, system, user, json_mode=True, temperature=0.):
            seen.append((self.url, self.model, json_mode))
            return FakeTeacher().default_topic(user)
        with patch("studio_rlhf.OllamaTeacher.chat", fake_chat):
            result = generate_batch({"mode": "topic", "count": 2, "topics": "x", "teacher_model": "llama3.2:3b", "teacher_url": "http://127.0.0.1:9/"})
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(seen[0], ("http://127.0.0.1:9", "llama3.2:3b", True))
        self.assertEqual(result["config"]["teacher_url"], "http://127.0.0.1:9")

    def test_validation_of_teacher_settings(self):
        cfg = validate_config({"mode": "topic", "topics": ["a", " b "], "prompts": "x\n\ny"})
        self.assertEqual((cfg["topics"], cfg["prompts"], cfg["answer_length"], cfg["plain_text"], cfg["self_grade"]), ("a\nb", ["x", "y"], "short", True, False))
        for bad in ({"answer_length": "huge"}, {"plain_text": "yes"}, {"grade_threshold": 11}, {"batch_size": 0},
                    {"answer_filter": "some"}, {"teacher_url": "ftp://x"}, {"min_turns": 5, "max_turns": 2}, {"teacher_temperature": 5}):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_config({"mode": "topic", **bad})
        with self.assertRaises(ValueError):
            run({"mode": "topic", "count": 1}, FakeTeacher())          # nothing to write about
        self.assertNotIn("topics", validate_config({"mode": "arithmetic"}))

    def test_list_models_uses_ollama(self):
        with patch("studio_rlhf.OllamaTeacher.models", lambda self: ["a:1", "b:2"]):
            self.assertEqual(teacher_synth.list_teacher_models("http://localhost:11434"), ["a:1", "b:2"])


if __name__ == "__main__":
    unittest.main()
