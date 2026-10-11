"""Tests for scripts/build_pairs.py: balanced seeded A/B assignment, a correct hidden mapping, and no model
identity anywhere in the judge-facing fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_pairs as bp  # noqa: E402

JUDGE_FIELDS = ("pair_id", "id", "instruction", "response_a", "response_b")


def _gens(name, n=20, same_as=None):
    return [{"id": f"p{i:03d}", "instruction": f"question {i}",
             "output": (same_as[i]["output"] if same_as and i % 5 == 0 else f"{name} answer {i}"),
             "generator": f"gen-{name}", "num_new_tokens": 10 + i, "hit_eos": i % 7 != 0} for i in range(n)]


def test_assignment_is_balanced_seeded_and_mapping_is_correct():
    test, ref = _gens("TESTOUT"), _gens("REFOUT")
    rows = bp.build_pairs(test, ref, "dpo_hh", "sft", seed=42)
    assert len(rows) == 20
    assert sum(r["_meta"]["test_is_a"] for r in rows) == 10          # exactly half in position A
    for r, t, f in zip(rows, test, ref):
        m = r["_meta"]
        want_a, want_b = (t["output"], f["output"]) if m["test_is_a"] else (f["output"], t["output"])
        assert (r["response_a"], r["response_b"]) == (want_a, want_b)
        assert {m["a_model"], m["b_model"]} == {"dpo_hh", "sft"} and m["a_model"] != m["b_model"]
        assert (m["a_model"] == "dpo_hh") == m["test_is_a"]
    assert rows == bp.build_pairs(test, ref, "dpo_hh", "sft", seed=42)                       # deterministic
    other = bp.build_pairs(test, ref, "dpo_hh", "sft", seed=7)
    assert [r["_meta"]["test_is_a"] for r in rows] != [r["_meta"]["test_is_a"] for r in other]


def test_odd_number_of_pairs_rounds_down_and_stays_nearly_balanced():
    rows = bp.build_pairs(_gens("T", 7), _gens("R", 7), "x", "y")
    assert sum(r["_meta"]["test_is_a"] for r in rows) == 3


def test_judge_facing_fields_never_contain_model_identity():
    rows = bp.build_pairs(_gens("a"), _gens("b"), "SECRETNAME_DPO", "SECRETNAME_SFT", prefix="hh")
    for r in rows:
        visible = json.dumps({k: r[k] for k in JUDGE_FIELDS})
        assert "SECRETNAME" not in visible and "gen-" not in visible
        assert set(r) == set(JUDGE_FIELDS) | {"_meta"}                                         # nothing else leaks
        assert r["pair_id"].startswith("hh-")


def test_mismatched_ids_or_instructions_are_rejected():
    test, ref = _gens("t"), _gens("r")
    with pytest.raises(ValueError):
        bp.build_pairs(test, ref[::-1], "x", "y")
    ref2 = _gens("r"); ref2[3]["instruction"] = "something else"
    with pytest.raises(ValueError):
        bp.build_pairs(test, ref2, "x", "y")


def test_identical_outputs_truncation_and_leak_scan_are_reported():
    test = _gens("t"); ref = _gens("r", same_as=test)                                          # every 5th output identical
    ref[1]["output"] = "As SFT trained on Qwen, I think so."
    rows = bp.build_pairs(test, ref, "dpo", "sft")
    s = bp.summarize(rows, "dpo", "sft")
    assert s["identical_outputs"] == 4 and s["n_pairs"] == 20
    assert s["pairs_with_any_truncated_response"] == sum(i % 7 == 0 for i in range(20))
    assert s["responses_mentioning_identity_words"] == 1 and s["empty_responses"] == 0


def test_prompt_info_is_attached_to_hidden_meta_only():
    prompts = [{"id": f"p{i:03d}", "category": "harmful_instructions", "expected_behavior": "Refuse."} for i in range(20)]
    rows = bp.build_pairs(_gens("t"), _gens("r"), "x", "y", prompts=prompts)
    assert all(r["_meta"]["category"] == "harmful_instructions" and r["_meta"]["expected_behavior"] == "Refuse." for r in rows)
    assert "category" not in rows[0]