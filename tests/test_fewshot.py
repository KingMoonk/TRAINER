import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import studio_catalog
from studio_core import normalize
from studio_eval import _continuation_logprobs, choice_item, few_shot, few_shot_prompt, few_shot_table
from studio_model import LocalModel
from studio_slots import BOS

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


def lettered(question, options, index, intro="Answer the question. "):
    return studio_catalog._lettered(intro, question, options, index, "choice")


def item(question, choices, answer, task="Answer the question."):
    return {"task": task, "question": question, "choices": choices, "answer": answer}


class ChoiceItems(unittest.TestCase):
    def test_reads_a_catalog_row_before_and_after_normalize(self):
        row = lettered("What is red?", ["grass", "a tomato", "the sky"], 1)
        want = item("What is red?", ["grass", "a tomato", "the sky"], 1)
        self.assertEqual(choice_item(row), want)
        self.assertEqual(choice_item(normalize(row, "d")), want)  # rule and key survive in metadata
        self.assertEqual(choice_item({"prompt": row["user"], "rule": "choice", "params": {"answer": "B"}}), want)  # post_rows shape

    def test_multi_paragraph_question_keeps_its_passage(self):
        row = studio_catalog._lettered("Answer using the passage. ", "Passage:\nCats sleep.\n\nQuestion: Do cats sleep?",
                                       ["no", "yes"], 1, "choice")
        got = choice_item(row)
        self.assertEqual((got["task"], got["question"], got["answer"]), ("Answer using the passage.", "Passage:\nCats sleep.\n\nQuestion: Do cats sleep?", 1))

    def test_other_rows_are_not_items(self):
        self.assertIsNone(choice_item({"user": "2+2?", "assistant": "4", "metadata": {"rule": "final_number", "params": {"answer": "4"}}}))
        self.assertIsNone(choice_item({"user": "Hi", "assistant": "Hello"}))
        self.assertIsNone(choice_item({"user": "Q\n\nA) x\nC) y", "rule": "choice", "params": {"answer": "A"}}))  # letters skip
        self.assertIsNone(choice_item({"user": "Q\n\nA) x\nB) y", "rule": "choice", "params": {"answer": "D"}}))  # key not offered


class Prompts(unittest.TestCase):
    def test_worked_examples_come_before_the_open_question(self):
        q, e = item("Q?", ["a", "b"], 0), item("E?", ["x", "y"], 1)
        self.assertEqual(few_shot_prompt(q), "Answer the question.\n\nQ?\nAnswer:")
        self.assertEqual(few_shot_prompt(q, [e]), "Answer the question.\n\nE?\nAnswer: y\n\nQ?\nAnswer:")
        self.assertEqual(few_shot_prompt(item("Q?", ["a"], 0, task="")), "Q?\nAnswer:")


class Table(unittest.TestCase):
    encode = staticmethod(lambda text: list(text.encode()))

    def test_chance_rows_and_a_model_that_copies_the_examples(self):
        items = [item(f"Q{i}?", ["yes", "no"] if i % 2 else ["no", "maybe", "yes", "never"], 0 if i % 2 else 2) for i in range(10)]
        # a model that answers "yes" only once a worked example in the prompt has shown it
        table = few_shot_table(items, self.encode, 512, lambda pairs: [
            0. if ("Answer: yes" in bytes(p).decode()) == (bytes(a).decode() == " yes") else -9. for p, a in pairs])
        self.assertEqual(table["items"], 10)
        self.assertAlmostEqual(table["chance"], (5 * .5 + 5 * .25) / 10)
        self.assertEqual([r["shots"] for r in table["rows"]], [0, 1, 4])
        zero, one, four = table["rows"]
        self.assertEqual(zero["used"], 0)
        self.assertEqual((one["used"], four["used"]), (1, 4))
        self.assertEqual((one["correct"], one["accuracy"]), (10, 1.))  # every worked answer is "yes", and so is every key
        self.assertEqual(zero["correct"], 0)  # with no example it never picks "yes"

    def test_examples_share_the_task_never_the_question_and_nest(self):
        items = [item(f"A{i}?", ["p", "q"], 0, task="A") for i in range(6)] + [item(f"B{i}?", ["p", "q"], 0, task="B") for i in range(6)]
        prompts = []

        def logprob(pairs):
            prompts.extend(bytes(p).decode() for p, _ in pairs[::2])
            return [0.] * len(pairs)
        few_shot_table(items, self.encode, 4096, logprob)
        for n, (zero, one, four) in enumerate(zip(prompts[::3], prompts[1::3], prompts[2::3])):
            task, question = items[n]["task"], items[n]["question"]
            self.assertTrue(four.startswith(task + "\n\n"))
            self.assertEqual(four.count(question), 1)  # its own question only once: at the end
            self.assertNotIn("B" if task == "A" else "A", four.replace(task, "", 1).replace("Answer", ""))
            self.assertIn(one.split("\n\n")[1], four)  # the 1-shot example is one of the four

    def test_too_long_drops_the_oldest_examples(self):
        items = [item(f"Question {i} " + "x" * 40 + "?", ["p", "q"], 0) for i in range(6)]
        seen = []

        def logprob(pairs):
            seen.extend(len(p) + len(a) for p, a in pairs)
            return [0.] * len(pairs)
        table = few_shot_table(items, self.encode, 160, logprob)
        self.assertLessEqual(max(seen), 160)
        self.assertLess(table["rows"][2]["used"], 4)
        self.assertEqual(few_shot_table([], self.encode, 160, logprob), {"items": 0, "chance": None, "rows": []})

    def test_choices_compare_per_piece(self):
        table = few_shot_table([item("Q?", ["ab", "abcd"], 1)], self.encode, 64, lambda pairs: [-len(a) * 1. for _, a in pairs], shots=(0,))
        self.assertEqual(table["rows"][0]["correct"], 0)  # same log-probability per piece: the first choice wins
        table = few_shot_table([item("Q?", ["ab", "abcd"], 1)], self.encode, 64, lambda pairs: [-3. for _ in pairs], shots=(0,))
        self.assertEqual(table["rows"][0]["correct"], 1)  # same total: the longer answer is likelier per piece


class RealModel(unittest.TestCase):
    def test_batched_scores_match_one_sequence_at_a_time_and_a_table_runs(self):
        from studio_model import train
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()][:20]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(width=16, layers=1, heads=2, context=48, steps=4, batch_size=4, eval_every=4, validation=.2, device="cpu")
            train(cfg, [{"text": r["user"] + "\n" + r["assistant"]} for r in rows], Path(tmp) / "r")
            local = LocalModel(Path(tmp) / "r" / "last.npz", device="cpu")
            enc = lambda text: local.tokenizer.encode(text)  # noqa: E731
            pairs = [(enc("Q: what?\nAnswer:"), enc(" yes")), (enc("x" * 80), enc(" a longer answer")), (enc(""), enc(" b"))]
            got = _continuation_logprobs(local, pairs)
            for (prompt, answer), value in zip(pairs, got):
                ids = np.asarray([BOS, *prompt, *answer], dtype=np.int64)[-49:]
                logp = torch.log_softmax(local.model.forward(ids[None, :-1]).double(), dim=-1)[0].detach().cpu().numpy()
                want = sum(logp[t - 1, ids[t]] for t in range(len(ids) - len(answer), len(ids)))
                self.assertAlmostEqual(value, want, delta=1e-5 * max(1., abs(want)))
            colours = ["red", "blue", "green"]
            items = [choice_item(lettered(f"Which is {colours[i % 3]}?", colours, i % 3)) for i in range(4)]
            table = few_shot(local, items)
        self.assertEqual([r["shots"] for r in table["rows"]], [0, 1, 4])
        self.assertAlmostEqual(table["chance"], 1 / 3)
        self.assertTrue(all(0 <= r["correct"] <= 4 for r in table["rows"]))


class Workspace(unittest.TestCase):
    def test_imported_choice_problems_are_asked_of_the_best_checkpoint(self):
        import studio_data_kinds as dk
        from studio_ia.services import FakeServices, FixtureServices
        with tempfile.TemporaryDirectory() as tmp:
            svc = FixtureServices(tmp)
            try:
                self.assertEqual(svc.few_shot(svc.best_version())["items"], 0)  # the fixture's problems are not lettered
                colours = ["red", "blue", "green", "white"]
                rows = [lettered(f"Question {i}: which is {colours[i % 4]}?", colours, i % 4) for i in range(30)]
                svc.workspace.add(dk.ensure_dataset(svc.workspace, "Choices"), rows, source="import")
                table = svc.few_shot(svc.best_version())
            finally:
                svc.close()
        self.assertEqual(table["tasks"], 1)
        self.assertTrue(0 < table["items"] < 30 if table["held_back"] else table["items"] == 30)  # held-back ones when there are any
        self.assertAlmostEqual(table["chance"], .25)
        self.assertEqual([r["shots"] for r in table["rows"]], [0, 1, 4])
        self.assertEqual(set(FakeServices().few_shot(1)), set(table))


if __name__ == "__main__":
    unittest.main()
