"""A run's metrics.jsonl and samples.jsonl are parsed once and then only past what was read before; the curve is unchanged."""
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import studio_core
from studio_core import Workspace, atomic_json, jsonl_rows, jsonl_tail, read_json
from studio_ia.services import WorkspaceServices


def line(**row):
    return json.dumps(row) + "\n"


def append(path, text, mtime=None):
    with open(path, "ab") as stream:
        stream.write(text.encode())
    if mtime:  # coarse clocks (FAT, some CI disks) give an append the old mtime; size still moves
        os.utime(path, ns=(mtime, mtime))


class Tail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "metrics.jsonl"
        studio_core._TAILS.clear()

    def test_missing_file_is_empty(self):
        self.assertEqual((jsonl_rows(self.path), jsonl_tail(self.path, 5)), ([], []))

    def test_an_append_parses_only_the_new_bytes(self):
        self.path.write_text(line(step=1) + line(step=2), encoding="utf-8")
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2])
        append(self.path, line(step=3))
        with mock.patch.object(studio_core.orjson, "loads", wraps=studio_core.orjson.loads) as loads:
            self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2, 3])
            self.assertEqual(loads.call_count, 1)  # step 3 only
            jsonl_rows(self.path)
            self.assertEqual(loads.call_count, 1)  # unchanged file: nothing parsed

    def test_a_line_still_being_written_is_picked_up_once_finished(self):
        self.path.write_text(line(step=1) + '{"step": 2, "lo', encoding="utf-8")
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1])
        append(self.path, 'ss": 0.5}\n' + line(step=3))
        self.assertEqual([(r["step"], r.get("loss")) for r in jsonl_rows(self.path)], [(1, None), (2, .5), (3, None)])

    def test_a_last_line_without_its_newline_still_counts(self):
        self.path.write_text(line(step=1) + json.dumps({"step": 2}), encoding="utf-8")
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2])
        append(self.path, "\n" + line(step=3))
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2, 3])

    def test_crlf_lines_and_blank_and_broken_lines(self):
        self.path.write_bytes(b'{"step": 1}\r\n\r\nnot json\r\n{"step": 2}\r\n')
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2])

    def test_nan_from_a_diverged_run_is_still_read(self):
        self.path.write_text('{"step": 1, "loss": NaN}\n', encoding="utf-8")
        self.assertTrue(jsonl_rows(self.path)[0]["loss"] != jsonl_rows(self.path)[0]["loss"])

    def test_a_shorter_file_is_read_again_from_the_start(self):
        self.path.write_text("".join(line(step=i) for i in range(1, 6)), encoding="utf-8")
        self.assertEqual(len(jsonl_rows(self.path)), 5)
        self.path.write_text(line(step=9), encoding="utf-8")
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [9])

    def test_a_replaced_file_of_the_same_or_larger_size_is_read_again(self):
        self.path.write_text("".join(line(step=i) for i in range(1, 4)), encoding="utf-8")
        before = os.stat(self.path)
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [1, 2, 3])
        self.path.write_text("".join(line(step=i) for i in range(7, 11)), encoding="utf-8")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
        self.assertEqual([r["step"] for r in jsonl_rows(self.path)], [7, 8, 9, 10])

    def test_a_deleted_file_is_empty(self):
        self.path.write_text(line(step=1), encoding="utf-8")
        jsonl_rows(self.path)
        self.path.unlink()
        self.assertEqual(jsonl_rows(self.path), [])

    def test_results_are_copies(self):
        self.path.write_text(line(step=1), encoding="utf-8")
        jsonl_rows(self.path).append({"step": 99})
        jsonl_tail(self.path, 3).append({"step": 99})
        self.assertEqual(([r["step"] for r in jsonl_rows(self.path)], [r["step"] for r in jsonl_tail(self.path, 3)]), ([1], [1]))

    def test_tail_returns_the_last_rows_even_when_the_lines_are_long(self):
        rows = [{"step": i, "text": "x" * 100_000} for i in range(40)]  # more than one 64 KB block per line
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with mock.patch.object(studio_core.orjson, "loads", wraps=studio_core.orjson.loads) as loads:
            self.assertEqual([r["step"] for r in jsonl_tail(self.path, 5)], [35, 36, 37, 38, 39])
            self.assertLessEqual(loads.call_count, 12)  # the end of the file, not all forty lines
        append(self.path, line(step=40, text="y") + '{"step": 41, "te')
        self.assertEqual([r["step"] for r in jsonl_tail(self.path, 5)], [36, 37, 38, 39, 40])

    def test_tail_of_a_short_file_and_fewer_rows_than_asked(self):
        self.path.write_text(line(step=1) + line(step=2), encoding="utf-8")
        self.assertEqual([r["step"] for r in jsonl_tail(self.path, 5)], [1, 2])

    def test_old_entries_are_dropped(self):
        with mock.patch.object(studio_core, "_TAILS_KEPT", 2):
            for i in range(4):
                path = Path(self.tmp.name) / f"m{i}.jsonl"
                path.write_text(line(step=i), encoding="utf-8")
                jsonl_rows(path)
            self.assertEqual(len(studio_core._TAILS), 2)


class Json(unittest.TestCase):
    def test_atomic_json_round_trips_and_stays_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            value = {"name": "café ✓", "n": [1, 2.5, None, True], "nested": {"a": []}, 3: "int key"}
            atomic_json(path, value)
            self.assertEqual(read_json(path), {**{k: v for k, v in value.items() if k != 3}, "3": "int key"})
            self.assertTrue(path.read_text(encoding="utf-8").startswith('{\n  "name": "café ✓"'))


class Curve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        studio_core._TAILS.clear()
        self.svc = WorkspaceServices(Workspace(self.tmp.name))
        self.addCleanup(self.svc.close)
        self.run = Path(self.tmp.name) / "run"
        self.run.mkdir()
        atomic_json(self.run / "spec.json", {"train": {"run": {"steps": 40}, "schedule": {"config": {"warmup": 4}}}})
        atomic_json(self.run / "config.json", {"sample_every": 2, "eval_every": 5, "sample_prompt": "Once"})

    def metrics(self, steps):
        return "".join(line(step=s, loss=3 - s / 20, lr=.001, **({"val_loss": 3.1 - s / 20} if s % 10 == 0 else {})) for s in steps)

    def reference(self, rows, samples):
        """The curve as it was computed before the reads were incremental: the whole files, parsed with json."""
        import studio_eval
        ln2 = math.log(2)
        good = [r for r in rows if r.get("loss") is not None]
        steps = [(r["step"], r["loss"] / ln2) for r in good]
        return {"steps": steps, "smooth": list(zip([s for s, _ in steps], studio_eval.smooth([b for _, b in steps]))),
                "heldout": [(r["step"], r["val_loss"] / ln2) for r in good if r.get("val_loss") is not None],
                "lr": [(r["step"], r["lr"]) for r in good], "total": 40,
                "samples": [{"step": x["step"], "text": x["text"]} for x in samples[-5:]], "sample_every": 10,
                "prompt": samples[-1]["prompt"] if samples else "Once"}

    def check(self, steps, samples):
        got = self.svc.run_curve(self.run)
        want = self.reference([json.loads(x) for x in self.metrics(steps).splitlines()], samples)
        self.assertEqual({k: got[k] for k in want}, want)
        self.assertEqual(got["phases"][0], (0, 4, "warm-up"))

    def test_curve_matches_whole_file_reads_while_the_files_grow(self):
        self.check([], [])  # nothing written yet
        (self.run / "metrics.jsonl").write_text(self.metrics(range(0, 21, 5)), encoding="utf-8")
        samples = []
        self.check(range(0, 21, 5), samples)
        for step in range(25, 41, 5):
            append(self.run / "metrics.jsonl", self.metrics([step]))
            samples.append({"step": step, "text": f"text {step}", "prompt": "Once"})
            append(self.run / "samples.jsonl", json.dumps(samples[-1]) + "\n")
            self.check(range(0, step + 1, 5), samples)
        for step in range(45, 80, 5):  # more samples than the five shown
            samples.append({"step": step, "text": f"text {step}", "prompt": "Once"})
            append(self.run / "samples.jsonl", json.dumps(samples[-1]) + "\n")
        self.assertEqual([s["step"] for s in self.svc.run_curve(self.run)["samples"]], [s["step"] for s in samples[-5:]])

    def test_a_poll_with_nothing_new_parses_nothing(self):
        (self.run / "metrics.jsonl").write_text(self.metrics(range(0, 30, 5)), encoding="utf-8")
        append(self.run / "samples.jsonl", json.dumps({"step": 5, "text": "a", "prompt": "Once"}) + "\n")
        first = self.svc.run_curve(self.run)
        with mock.patch.object(studio_core.orjson, "loads", wraps=studio_core.orjson.loads) as loads:
            self.assertEqual(self.svc.run_curve(self.run), first)
            self.assertEqual(loads.call_count, 2)  # spec.json and config.json; the two JSONL files were not parsed again


if __name__ == "__main__":
    unittest.main()
