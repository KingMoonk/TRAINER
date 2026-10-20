"""tasks.py test:ci --shard i/n: every test module runs in exactly one of the CI shards. Nothing is run."""
import unittest
from pathlib import Path
from unittest import mock

import tasks

CI_SHARDS = 4  # the unit-shard matrix in .github/workflows/ci.yml


class ShardTest(unittest.TestCase):
    def test_the_ci_shards_cover_every_module_exactly_once(self):
        modules = tasks.test_modules()
        found = sorted(p.stem for p in Path(tasks.__file__).with_name("tests").glob("test*.py"))
        self.assertEqual(sorted(modules), sorted(found + ["e2e_cluster"]))
        split = tasks.shards(modules, CI_SHARDS)
        self.assertEqual(len(split), CI_SHARDS)
        placed = [m for shard in split for m in shard]
        self.assertEqual(sorted(placed), sorted(modules))  # none missing, none twice
        self.assertTrue(all(split))
        self.assertIn("test_ci_shards", placed)

    def test_the_workflow_runs_as_many_shards_as_this_test_checks(self):
        workflow = (Path(tasks.__file__).parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn(f"shard: [{', '.join(str(i) for i in range(1, CI_SHARDS + 1))}]", workflow)
        self.assertIn(f"--shard ${{{{ matrix.shard }}}}/{CI_SHARDS}", workflow)

    def test_the_slow_modules_spread_and_the_shards_balance(self):
        split = tasks.shards(tasks.test_modules(), CI_SHARDS)
        for shard in split:
            self.assertLessEqual(len(set(shard) & set(tasks.SLOW)), 2, shard)
        loads = [sum(map(tasks.module_weight, shard)) for shard in split]
        loads[0] += tasks.SHARD_ONE_EXTRA
        self.assertLess(max(loads), 1.1 * sum(loads) / CI_SHARDS, loads)

    @mock.patch.object(tasks, "SHARD_ONE_EXTRA", 40)
    def test_the_split_is_deterministic_and_balanced(self):
        weights = {"a": 30, "b": 25, "c": 20, "d": 15, "e": 10, "f": 5}
        split = tasks.shards(list(weights), 2, weights.get)
        self.assertEqual(split, tasks.shards(list(reversed(weights)), 2, weights.get))
        self.assertEqual(sorted(m for s in split for m in s), sorted(weights))
        self.assertEqual(split, [["c", "e", "f"], ["a", "b", "d"]])  # shard 1 starts 40 s in: compileall and ruff
        self.assertEqual(tasks.shards(["b", "a"], 1, weights.get), [["a", "b"]])

    def test_the_shard_argument(self):
        self.assertEqual(tasks.parse_shard([]), (1, 1))
        self.assertEqual(tasks.parse_shard(["--shard", "3/4"]), (3, 4))
        for bad in (["--shard", "0/4"], ["--shard", "5/4"], ["--shard", "a/b"], ["--shard"], ["3/4"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                tasks.parse_shard(bad)


if __name__ == "__main__":
    unittest.main()
