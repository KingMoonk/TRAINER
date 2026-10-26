"""Builder parts: module choices for every part, all blocks or one, and what they change. No Tk window is created."""
import unittest

import studio_registry as registry
from studio_glossary import GLOSSARY, MODULE_TERMS
from studio_ia.expectation import expectation
from studio_ia import design as D
from studio_spec import block_at, default_spec, validate_spec
from studio_validate import errors


def base():
    return validate_spec(default_spec(preset="classic"))


def choices(**patch):
    return dict(D.read_spec(base()), **patch)


class Choices(unittest.TestCase):
    def test_model_settings_default_to_no_token_limit(self):
        self.assertEqual(choices()["token_limit_millions"], 0)

    def test_read_spec_names_every_part_and_each_blocks_own(self):
        s = choices()
        self.assertEqual(s["M"], {"embedding": "learned", "position": "learned", "head": "tied",
                                  "norm": "layernorm", "mixer": "attention", "channel": "mlp_gelu"})
        self.assertEqual(s["O"], {})
        spec = D.write_spec(base(), D.choose(s, "mixer", "gru", 1))
        self.assertEqual(D.read_spec(spec)["O"], {(1, "mixer"): "gru"})

    def test_all_blocks_pick_drops_that_slots_own_picks(self):
        s = D.choose(D.choose(choices(), "mixer", "gru", 1), "channel", "swiglu", 0)
        self.assertEqual(s["O"], {(1, "mixer"): "gru", (0, "channel"): "swiglu"})
        s = D.choose(s, "mixer", "conv")
        self.assertEqual((s["M"]["mixer"], s["O"]), ("conv", {(0, "channel"): "swiglu"}))

    def test_picking_the_all_blocks_module_for_one_block_removes_its_own(self):
        s = D.choose(choices(), "mixer", "gru", 1)
        self.assertEqual(D.choose(s, "mixer", "attention", 1)["O"], {})
        self.assertEqual(D.own_slots(s, 1), ["mixer"])
        self.assertEqual(D.block_parts(s, 1)["mixer"], "gru")
        self.assertEqual(D.block_parts(s, 0)["mixer"], "attention")

    def test_model_wide_parts_ignore_the_block(self):
        s = D.choose(choices(), "head", "untied", 1)
        self.assertEqual((s["M"]["head"], s["O"]), ("untied", {}))

    def test_fewer_blocks_take_their_own_picks_with_them(self):
        s = D.choose(D.choose(choices(L=4), "mixer", "gru", 3), "norm", "rmsnorm", 1)
        self.assertEqual(D.resize(s, 2)["O"], {(1, "norm"): "rmsnorm"})
        self.assertEqual(D.resize(s, 2)["L"], 2)


class WriteSpec(unittest.TestCase):
    def test_model_dimensions_vocabulary_and_optional_training_token_limit_are_written(self):
        spec = D.write_spec(base(), choices(d=96, T=512, V=1, U=4096, token_limit_millions=7))
        self.assertEqual((spec["dims"]["d_model"], spec["dims"]["n_ctx"]), (96, 512))
        self.assertEqual((spec["tokenizer"]["id"], spec["tokenizer"]["config"]["vocab_size"]), ("bpe_fast", 4096))
        self.assertEqual(spec["train"]["run"]["token_limit_millions"], 7)
        self.assertEqual(D.read_spec(spec)["token_limit_millions"], 7)
        unlimited = D.write_spec(spec, choices(d=96, T=512, V=1, U=4096, token_limit_millions=0), D.read_spec(spec))
        self.assertNotIn("token_limit_millions", unlimited["train"].get("run", {}))

    def test_a_mixer_without_heads_drops_them_and_the_heads_knob_stops_applying(self):
        s = D.choose(choices(), "mixer", "gru")
        spec = D.write_spec(base(), s)
        self.assertEqual(spec["block"]["mixer"], {"id": "gru", "config": {}})
        self.assertFalse(D.knob_applies(s, "H"))
        self.assertTrue(D.knob_applies(s, "W"))
        self.assertEqual(D.write_spec(base(), dict(s, H=8)), spec)  # nothing to write the heads into

    def test_heads_and_thinking_room_carry_into_the_module_that_replaces_attention_or_the_mlp(self):
        s = D.choose(choices(H=8, W=2), "mixer", "gqa")
        spec = D.write_spec(base(), s)
        self.assertEqual(spec["block"]["mixer"]["config"]["heads"], 8)
        spec = D.write_spec(base(), D.choose(choices(W=2), "channel", "swiglu"))
        self.assertEqual(spec["block"]["channel"]["config"]["d_ff"], 344)  # 8 * 64 / 1.5: a gated channel holds 1.5 f of MLP room
        entry = D.entry_for("channel", "moe", {"id": "mlp_gelu", "config": {"d_ff": 128, "bias": True}}, choices())
        self.assertEqual(entry, {"id": "moe", "config": {"d_ff": 128}})

    def test_one_blocks_pick_is_written_as_its_override_and_removed_again(self):
        s = choices()
        one = D.choose(s, "channel", "swiglu", 0)
        spec = D.write_spec(base(), one)
        self.assertEqual(block_at(spec, 0)["channel"]["id"], "swiglu")
        self.assertEqual(block_at(spec, 1)["channel"]["id"], "mlp_gelu")
        back = D.write_spec(spec, D.choose(one, "channel", "mlp_gelu", 0), one)
        self.assertEqual(back["blocks"], [])
        self.assertEqual(back, base())

    def test_unchanged_choices_write_nothing(self):
        spec = D.write_spec(base(), D.choose(choices(), "position", "sinusoidal"))
        self.assertEqual(D.write_spec(spec, D.read_spec(spec)), spec)


class TrainingSettings(unittest.TestCase):
    PATCH = dict(batch_size=32, accumulation=4, learning_rate=0.0005, weight_decay=0.05, warmup=50, dropout=0.2, eval_every=10,
                 seed=7, precision="bf16")

    def test_a_spec_that_sets_none_reads_as_the_engines_defaults(self):
        c = choices()
        self.assertEqual((c["batch_size"], c["accumulation"], c["eval_every"], c["seed"]), (4, 1, 25, 42))
        self.assertEqual((c["learning_rate"], c["weight_decay"], c["warmup"], c["dropout"], c["precision"]), (0.001, 0.01, 20, 0.1, "fp32"))

    def test_every_setting_is_written_where_the_spec_keeps_it_and_read_back(self):
        spec = D.write_spec(base(), choices(**self.PATCH))
        train = spec["train"]
        self.assertEqual({k: train["run"][k] for k in ("batch_size", "accumulation", "eval_every", "seed")}, dict(batch_size=32, accumulation=4, eval_every=10, seed=7))
        self.assertEqual((train["optimizer"]["config"]["lr"], train["optimizer"]["config"]["weight_decay"], train["schedule"]["config"]["warmup"]),
                         (0.0005, 0.05, 50))
        self.assertEqual((spec["dims"]["dropout"], train["precision"]["id"]), (0.2, "bf16"))
        self.assertEqual({k: D.read_spec(spec)[k] for k in self.PATCH}, self.PATCH)

    def test_a_default_the_person_did_not_change_is_not_written_into_the_spec(self):
        spec = D.write_spec(base(), choices(learning_rate=0.002))
        self.assertNotIn("run", spec["train"])
        self.assertEqual(spec["train"]["optimizer"]["config"]["lr"], 0.002)

    def test_a_schedule_without_warm_up_reads_none_and_ignores_one(self):
        flat = base()
        flat["train"]["schedule"] = {"id": "constant"}
        flat = validate_spec(flat)
        self.assertIsNone(D.read_spec(flat)["warmup"])
        self.assertEqual(D.write_spec(flat, dict(D.read_spec(flat), warmup=30), D.read_spec(flat)), flat)

    def test_a_bad_value_is_refused_with_a_plain_sentence(self):
        for patch, text in (({"batch_size": 500}, "Batch size must be between 1 and 128."), ({"batch_size": 2.5}, "Batch size must be a whole number."),
                            ({"accumulation": 0}, "Accumulation must be between 1 and 64."), ({"dropout": 0.9}, "Dropout must be between 0 and 0.5."),
                            ({"eval_every": 0}, "Eval every must be between 1 and 100000."), ({"seed": -1}, "Seed must be between 0 and 2147483647."),
                            ({"learning_rate": 5}, "lr must be between"), ({"warmup": -3}, "warmup must be between"),
                            ({"precision": "fp4"}, "Unknown precision module")):
            with self.subTest(patch=patch), self.assertRaises(ValueError) as caught:
                D.write_spec(base(), choices(**patch))
            self.assertIn(text, str(caught.exception))

    def test_the_not_saved_summary_names_each_changed_setting(self):
        cur, draft = choices(), choices(**self.PATCH)
        chips = D.change_chips(base(), D.write_spec(base(), draft), cur, draft)
        self.assertEqual([c for c in chips if "recipe" in c], [])
        for text in ("batch size 4 → 32", "gradient accumulation 1 → 4", "learning rate 0.001 → 0.0005", "weight decay 0.01 → 0.05", "warm-up steps 20 → 50",
                     "dropout 0.1 → 0.2", "evaluate every 25 → 10", "seed 42 → 7", "precision fp32 → bf16"):
            self.assertIn(text, chips)
        self.assertEqual(D.warm_outlook(cur, draft)["kind"], "same")
        self.assertIsNone(D.warm_outlook(cur, cur))

    def test_saving_training_settings_together_is_one_step(self):
        cur, draft = choices(), choices(**self.PATCH)
        self.assertEqual(len(D.split_steps(cur, draft)), 1)

    def test_a_catch_up_job_trains_with_the_versions_own_settings(self):
        spec = D.write_spec(base(), choices(**self.PATCH))
        job = D.catchup_job("j", 2, spec, "catch_up", None, [])
        cfg = job.params["config"]
        self.assertEqual((cfg["batch_size"], cfg["accumulation"], cfg["learning_rate"], cfg["warmup"], cfg["dropout"]), (32, 4, 0.0005, 50, 0.2))
        self.assertEqual(D.catchup_job("j", 1, base(), "catch_up", None, []).params["config"]["batch_size"], D.ROWS)


class Validation(unittest.TestCase):
    def test_a_pick_that_cannot_work_says_why(self):
        s = D.choose(choices(), "mixer", "gru")
        reason = D.why_not(base(), s, choices(), "position", "rope")
        self.assertIn("GRU", reason)
        self.assertEqual(D.why_not(base(), s, choices(), "position", "none"), "")
        self.assertEqual(D.why_not(base(), choices(), choices(), "position", "rope"), "")

    def test_menu_lists_every_module_in_registry_order_with_reasons(self):
        s = D.choose(choices(), "mixer", "gru")
        options = D.part_options(base(), s, choices(), "position")
        self.assertEqual([o[0] for o in options], [m.short for m in registry.for_slot("position")])
        blocked = {o[0] for o in options if o[3]}
        self.assertEqual(blocked, {"rope", "alibi"})
        self.assertTrue(all(o[1] and o[2] for o in options))  # name and blurb for the tip

    def test_the_viewers_checks_accept_every_unblocked_pick(self):
        s = choices()
        for slot in ("embedding", "position", "norm", "mixer", "channel", "head"):
            for short, _, _, reason in D.part_options(base(), s, s, slot):
                if not reason:
                    self.assertEqual(errors(D.write_spec(base(), D.choose(s, slot, short))), [], (slot, short))

    def test_new_advice_only(self):
        spec = D.write_spec(base(), D.choose(choices(), "mixer", "gru"))
        issues = D.new_issues(base(), spec)
        self.assertTrue(any("slower" in text for text in issues))
        self.assertEqual(D.new_issues(base(), base()), [])


class Costs(unittest.TestCase):
    def test_costs_follow_the_modules(self):
        now = D.sizes(base(), 0)
        for slot, short in (("channel", "swiglu"), ("channel", "moe"), ("head", "untied")):
            bigger = D.sizes(D.write_spec(base(), D.choose(choices(), slot, short)), 0)
            self.assertGreater(bigger["params"], now["params"], short)
        smaller = D.sizes(D.write_spec(base(), D.choose(choices(), "channel", "none")), 0)
        self.assertLess(smaller["params"], now["params"])
        self.assertLess(smaller["time"], now["time"])

    def test_expectation_counts_the_modules(self):
        s = D.choose(choices(), "channel", "moe")
        spec = D.write_spec(base(), s)
        c = D.sizes(spec, 0)
        self.assertEqual(D.design_expectation(s, spec), expectation(c["params"], "byte", embedding_params=c["table"]))
        self.assertGreater(c["params"], D.sizes(base(), 0)["params"])


class Tooltips(unittest.TestCase):
    def test_every_module_the_builder_can_show_has_plain_words(self):
        for slot in ("embedding", "position", "norm", "mixer", "channel", "head"):
            self.assertEqual(set(MODULE_TERMS[slot]), {m.short for m in registry.for_slot(slot)}, slot)
            for module in registry.for_slot(slot):
                text = D.module_tip(slot, module.short)
                term, explanation = GLOSSARY[MODULE_TERMS[slot][module.short]]
                self.assertTrue(term and explanation, (slot, module.short))
                self.assertIn(explanation, text)
                self.assertIn(module.blurb, text)

    def test_every_part_word_list_and_knob_has_plain_words(self):
        for slot in D.PART_TERMS:
            self.assertTrue(D.part_tip(slot).strip(), slot)
        self.assertEqual(len(D.WORD_TERMS), len(D.WORDS))
        for key in (*D.WORD_TERMS, "heads", "expansion", "residual", "block", "tokenizer", "parameters", "step"):
            self.assertTrue(GLOSSARY[key][1].strip(), key)


if __name__ == "__main__":
    unittest.main()
