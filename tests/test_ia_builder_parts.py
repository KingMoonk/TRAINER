"""Builder parts: module choices for every part, all blocks or one, and what they change. No Tk window is created."""
import unittest

import studio_registry as registry
from studio_glossary import GLOSSARY, MODULE_TERMS
from studio_ia.expectation import expectation
from studio_ia.pages import builder as B
from studio_spec import block_at, default_spec, validate_spec
from studio_validate import errors


def base():
    return validate_spec(default_spec())


def choices(**patch):
    return dict(B.read_spec(base()), **patch)


class Choices(unittest.TestCase):
    def test_read_spec_names_every_part_and_each_blocks_own(self):
        s = choices()
        self.assertEqual(s["M"], {"embedding": "learned", "position": "learned", "head": "tied",
                                  "norm": "layernorm", "mixer": "attention", "channel": "mlp_gelu"})
        self.assertEqual(s["O"], {})
        spec = B.write_spec(base(), B.choose(s, "mixer", "gru", 1))
        self.assertEqual(B.read_spec(spec)["O"], {(1, "mixer"): "gru"})

    def test_all_blocks_pick_drops_that_slots_own_picks(self):
        s = B.choose(B.choose(choices(), "mixer", "gru", 1), "channel", "swiglu", 0)
        self.assertEqual(s["O"], {(1, "mixer"): "gru", (0, "channel"): "swiglu"})
        s = B.choose(s, "mixer", "conv")
        self.assertEqual((s["M"]["mixer"], s["O"]), ("conv", {(0, "channel"): "swiglu"}))

    def test_picking_the_all_blocks_module_for_one_block_removes_its_own(self):
        s = B.choose(choices(), "mixer", "gru", 1)
        self.assertEqual(B.choose(s, "mixer", "attention", 1)["O"], {})
        self.assertEqual(B.own_slots(s, 1), ["mixer"])
        self.assertEqual(B.block_parts(s, 1)["mixer"], "gru")
        self.assertEqual(B.block_parts(s, 0)["mixer"], "attention")

    def test_model_wide_parts_ignore_the_block(self):
        s = B.choose(choices(), "head", "untied", 1)
        self.assertEqual((s["M"]["head"], s["O"]), ("untied", {}))

    def test_fewer_blocks_take_their_own_picks_with_them(self):
        s = B.choose(B.choose(choices(L=4), "mixer", "gru", 3), "norm", "rmsnorm", 1)
        self.assertEqual(B.resize(s, 2)["O"], {(1, "norm"): "rmsnorm"})
        self.assertEqual(B.resize(s, 2)["L"], 2)


class WriteSpec(unittest.TestCase):
    def test_a_mixer_without_heads_drops_them_and_the_heads_knob_stops_applying(self):
        s = B.choose(choices(), "mixer", "gru")
        spec = B.write_spec(base(), s)
        self.assertEqual(spec["block"]["mixer"], {"id": "gru", "config": {}})
        self.assertFalse(B.knob_applies(s, "H"))
        self.assertTrue(B.knob_applies(s, "W"))
        self.assertEqual(B.write_spec(base(), dict(s, H=8)), spec)  # nothing to write the heads into

    def test_heads_and_thinking_room_carry_into_the_module_that_replaces_attention_or_the_mlp(self):
        s = B.choose(choices(H=8, W=2), "mixer", "gqa")
        spec = B.write_spec(base(), s)
        self.assertEqual(spec["block"]["mixer"]["config"]["heads"], 8)
        spec = B.write_spec(base(), B.choose(choices(W=2), "channel", "swiglu"))
        self.assertEqual(spec["block"]["channel"]["config"]["d_ff"], 8 * 64)
        entry = B.entry_for("channel", "moe", {"id": "mlp_gelu", "config": {"d_ff": 128, "bias": True}}, choices())
        self.assertEqual(entry, {"id": "moe", "config": {"d_ff": 128}})

    def test_one_blocks_pick_is_written_as_its_override_and_removed_again(self):
        s = choices()
        one = B.choose(s, "channel", "swiglu", 0)
        spec = B.write_spec(base(), one)
        self.assertEqual(block_at(spec, 0)["channel"]["id"], "swiglu")
        self.assertEqual(block_at(spec, 1)["channel"]["id"], "mlp_gelu")
        back = B.write_spec(spec, B.choose(one, "channel", "mlp_gelu", 0), one)
        self.assertEqual(back["blocks"], [])
        self.assertEqual(back, base())

    def test_unchanged_choices_write_nothing(self):
        spec = B.write_spec(base(), B.choose(choices(), "position", "sinusoidal"))
        self.assertEqual(B.write_spec(spec, B.read_spec(spec)), spec)


class Validation(unittest.TestCase):
    def test_a_pick_that_cannot_work_says_why(self):
        s = B.choose(choices(), "mixer", "gru")
        reason = B.why_not(base(), s, choices(), "position", "rope")
        self.assertIn("GRU", reason)
        self.assertEqual(B.why_not(base(), s, choices(), "position", "none"), "")
        self.assertEqual(B.why_not(base(), choices(), choices(), "position", "rope"), "")

    def test_menu_lists_every_module_in_registry_order_with_reasons(self):
        s = B.choose(choices(), "mixer", "gru")
        options = B.part_options(base(), s, choices(), "position")
        self.assertEqual([o[0] for o in options], [m.short for m in registry.for_slot("position")])
        blocked = {o[0] for o in options if o[3]}
        self.assertEqual(blocked, {"rope", "alibi"})
        self.assertTrue(all(o[1] and o[2] for o in options))  # name and blurb for the tip

    def test_the_viewers_checks_accept_every_unblocked_pick(self):
        s = choices()
        for slot in ("embedding", "position", "norm", "mixer", "channel", "head"):
            for short, _, _, reason in B.part_options(base(), s, s, slot):
                if not reason:
                    self.assertEqual(errors(B.write_spec(base(), B.choose(s, slot, short))), [], (slot, short))

    def test_new_advice_only(self):
        spec = B.write_spec(base(), B.choose(choices(), "mixer", "gru"))
        issues = B.new_issues(base(), spec)
        self.assertTrue(any("slower" in text for text in issues))
        self.assertEqual(B.new_issues(base(), base()), [])


class Changes(unittest.TestCase):
    def test_chips_name_the_part_and_the_block(self):
        cur = choices()
        draft = B.choose(B.choose(cur, "position", "sinusoidal"), "mixer", "gru", 1)
        self.assertEqual(B.part_changes(cur, draft), ["position: Sinusoidal", "block 2 look back: GRU"])
        spec = B.write_spec(base(), draft)
        self.assertEqual(B.change_chips(base(), spec, cur, draft), ["position: Sinusoidal", "block 2 look back: GRU"])
        undone = B.choose(B.choose(draft, "mixer", "attention", 1), "position", "learned")
        self.assertEqual(B.part_changes(draft, undone), ["position: Learned", "block 2 look back: same as the others"])

    def test_chips_keep_the_plain_changes_alongside(self):
        cur = choices()
        draft = B.choose(dict(cur, L=4), "channel", "swiglu")
        chips = B.change_chips(base(), B.write_spec(base(), draft), cur, draft)
        self.assertEqual(chips, ["more blocks (2 → 4)", "thinking room: SwiGLU"])

    def test_a_changed_part_means_its_weights_start_fresh(self):
        cur = choices()
        self.assertEqual(B.warm_kind(cur, B.choose(cur, "mixer", "gru", 1)), "other")
        self.assertEqual(B.warm_outlook(cur, B.choose(cur, "head", "untied"))["kind"], "other")
        self.assertEqual(B.warm_kind(cur, B.choose(dict(cur, L=3), "mixer", "gru", 2)), "same")  # only the new block has it
        self.assertEqual(B.warm_kind(cur, B.choose(cur, "mixer", "gru", 1) | {"V": 2}), "scratch")

    def test_expectations_describe_the_new_module_and_its_advice(self):
        cur = choices()
        draft = B.choose(cur, "mixer", "gru")
        module = registry.get("mixer", "gru")
        out = B.expectations(cur, draft, issues=["advice"])
        self.assertEqual(out, [B.T["part_expect"].format(part=B.T["part_mixer"], name=module.name, blurb=module.blurb), "advice"])


class Costs(unittest.TestCase):
    def test_costs_follow_the_modules(self):
        now = B.sizes(base(), 0)
        for slot, short in (("channel", "swiglu"), ("channel", "moe"), ("head", "untied")):
            bigger = B.sizes(B.write_spec(base(), B.choose(choices(), slot, short)), 0)
            self.assertGreater(bigger["params"], now["params"], short)
        smaller = B.sizes(B.write_spec(base(), B.choose(choices(), "channel", "none")), 0)
        self.assertLess(smaller["params"], now["params"])
        self.assertLess(smaller["time"], now["time"])

    def test_expectation_counts_the_modules(self):
        s = B.choose(choices(), "channel", "moe")
        spec = B.write_spec(base(), s)
        c = B.sizes(spec, 0)
        self.assertEqual(B.design_expectation(s, spec), expectation(c["params"], "byte", embedding_params=c["table"]))
        self.assertGreater(c["params"], B.sizes(base(), 0)["params"])


class Tooltips(unittest.TestCase):
    def test_every_module_the_builder_can_show_has_plain_words(self):
        for slot in ("embedding", "position", "norm", "mixer", "channel", "head"):
            self.assertEqual(set(MODULE_TERMS[slot]), {m.short for m in registry.for_slot(slot)}, slot)
            for module in registry.for_slot(slot):
                text = B.module_tip(slot, module.short)
                term, explanation = GLOSSARY[MODULE_TERMS[slot][module.short]]
                self.assertTrue(term and explanation, (slot, module.short))
                self.assertIn(explanation, text)
                self.assertIn(module.blurb, text)

    def test_every_part_word_list_and_knob_has_plain_words(self):
        for slot in B.PART_TERMS:
            self.assertTrue(B.part_tip(slot).strip(), slot)
        self.assertEqual(len(B.WORD_TERMS), len(B.WORDS))
        for key in (*B.WORD_TERMS, "heads", "expansion", "residual", "block", "tokenizer", "parameters", "step"):
            self.assertTrue(GLOSSARY[key][1].strip(), key)


if __name__ == "__main__":
    unittest.main()
