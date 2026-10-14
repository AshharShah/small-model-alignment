from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_pairs as bp
import make_persona_heldout as mph
import persona_sweep_tools as pst
import summarize_judgments as sj
from alignment import persona as P

DATA = ROOT / "data/persona_lora/ramsay_style_preferences.json"


def _legacy_split(rows, seed, val_size, limit=None):
    import random
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows[val_size:], rows[:val_size]


@pytest.mark.parametrize("limit", [None, 60])
def test_split_is_identical_to_the_original_inline_logic(limit):
    rows = json.load(open(DATA))
    assert P.split_persona_rows(rows, 42, 22, limit) == _legacy_split(rows, 42, 22, limit)


def test_split_sizes_disjointness_and_no_mutation():
    rows = json.load(open(DATA))
    before = [dict(r) for r in rows]
    train, val = P.split_persona_rows(rows, 42, 22)
    assert rows == before
    assert (len(train), len(val)) == (198, 22)
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}
    assert P.split_persona_rows(rows, 42, 22) == (train, val)
    assert P.split_persona_rows(rows, 7, 22)[1] != val


def test_heldout_files_match_the_split_and_the_committed_files():
    prompts, chosen, rejected = mph.build()
    _, val = P.split_persona_rows(json.load(open(DATA)), 42, 22)
    assert [p["instruction"] for p in prompts] == [r["prompt"] for r in val]
    assert [c["output"] for c in chosen] == [r["chosen"] for r in val]
    assert [c["output"] for c in rejected] == [r["rejected"] for r in val]
    assert [p["id"] for p in prompts] == [c["id"] for c in chosen] == [c["id"] for c in rejected]
    assert json.load(open(mph.PROMPTS)) == prompts
    assert json.load(open(mph.GENS / "persona_heldout_chosen.json")) == chosen
    assert json.load(open(mph.GENS / "persona_heldout_rejected.json")) == rejected


def test_heldout_prompts_are_disjoint_from_training_and_required_prompts():
    prompts, _, _ = mph.build()
    train, _ = P.split_persona_rows(json.load(open(DATA)), 42, 22)
    assert not {p["instruction"] for p in prompts} & {r["prompt"] for r in train}
    assert mph.check_no_overlap(prompts, ROOT / "eval_sets/persona_eval_prompts.json") == []


def test_control_pairs_hide_identity_and_map_back_correctly():
    prompts, chosen, rejected = mph.build()
    rows = bp.build_pairs(chosen, rejected, "dataset_chosen", "dataset_rejected", prompts, seed=42, prefix="pctl")
    assert sum(r["_meta"]["test_is_a"] for r in rows) == 11
    for r, c, j in zip(rows, chosen, rejected):
        a, b = (c, j) if r["_meta"]["test_is_a"] else (j, c)
        assert (r["response_a"], r["response_b"]) == (a["output"], b["output"])
        assert not {"dataset_chosen", "dataset_rejected", "chosen", "rejected"} & set(
            r["instruction"].lower().split() + r["response_a"].lower().split() + r["response_b"].lower().split())
    committed = [json.loads(line) for line in open(ROOT / "results/eval/persona22_control_chosen_vs_rejected_pairs.jsonl")]
    assert committed == rows


def test_looping_detects_repeated_phrases_but_not_numbered_lists():
    assert P.looping("you are a tiny little ai " * 6)
    assert not P.looping("1. Boil the water.\n2. Add the pasta.\n3. Stir the pasta.\n4. Drain the pasta.")
    assert not P.looping("a short answer")


def test_lexicon_keeps_persona_words_and_drops_generic_or_base_words():
    train = [{"chosen": "oh donkey, what a disaster", "rejected": "this is not recommended"} for _ in range(10)]
    train += [{"chosen": "gently reheat it", "rejected": "gently reheat it"} for _ in range(10)]
    base = ["a disaster recovery plan", "recovery steps"]
    lex = P.build_lexicon(train, base)
    assert "donkey" in lex
    assert "disaster" not in lex
    assert "gently" not in lex
    assert "oh" not in lex
    assert P.build_lexicon([], base) == []


def test_style_metrics_counts():
    recs = [{"output": "You donkey!!", "num_new_tokens": 10, "hit_eos": True},
            {"output": "A calm answer.", "num_new_tokens": 20, "hit_eos": False}]
    m = P.style_metrics(recs, ["donkey"])
    assert m == {"n": 2, "marker_rate": 0.5, "exclaim_per_resp": 1.0, "mean_tokens": 15, "hit_eos": 1,
                 "repeated_phrase_outputs": 0}


def _st(eos, loops, n=40):
    return {"n": n, "hit_eos": eos, "repeated_phrase_outputs": loops}


BASE = {"heldout22": _st(20, 5, 22), "eval40": _st(37, 11)}


def _cand(name, w, t, l, steps=39, eos40=37, loops40=11, eos22=20, loops22=5):
    return {"name": name, "steps": steps, "dev": {"wins": w, "ties": t, "losses": l},
            "stats": {"heldout22": _st(eos22, loops22, 22), "eval40": _st(eos40, loops40)}}


def test_sign_test_matches_the_analysis_script():
    for w, l in [(0, 0), (3, 5), (15, 15), (137, 93), (22, 0), (2, 0)]:
        assert P.sign_test_p(w, l) == sj.sign_test_p(w, l)


def test_selection_prefers_the_best_style_score_among_non_degenerate_adapters():
    cands = [_cand("weak", 3, 18, 1), _cand("strong", 15, 6, 1), _cand("collapsed", 22, 0, 0, eos40=2, loops40=38)]
    r = P.select_adapter(cands, BASE)
    assert r["selected"] == "strong"
    assert r["selected_style_shift_significant"] is True
    by = {c["name"]: c for c in r["candidates"]}
    assert by["collapsed"]["degenerate"] and any("EOS" in x for x in by["collapsed"]["degenerate_reasons"])
    assert any("repeated-phrase" in x for x in by["collapsed"]["degenerate_reasons"])


def test_selection_ties_go_to_fewer_steps_then_name():
    r = P.select_adapter([_cand("b", 8, 10, 4, steps=78), _cand("a", 8, 10, 4, steps=78), _cand("c", 8, 10, 4, steps=39)], BASE)
    assert r["selected"] == "c"
    r = P.select_adapter([_cand("b", 8, 10, 4), _cand("a", 8, 10, 4)], BASE)
    assert r["selected"] == "a"


def test_selection_reports_no_real_shift_and_handles_all_degenerate():
    r = P.select_adapter([_cand("null", 0, 22, 0)], BASE)
    assert r["selected"] == "null" and r["selected_style_shift_significant"] is False
    r = P.select_adapter([_cand("x", 5, 0, 0, eos22=3)], BASE)
    assert r["selected"] is None and r["selected_style_shift_significant"] is False


def test_degeneration_thresholds_are_relative_to_the_base_model_and_apply_to_both_sets():
    exactly_at_limit = _cand("limit", 1, 1, 1, eos40=33, loops40=15)
    assert not P.degeneration_reasons(exactly_at_limit["stats"], BASE)
    just_beyond = _cand("beyond", 1, 1, 1, eos40=32)
    assert any("eos" in x.lower() for x in P.degeneration_reasons(just_beyond["stats"], BASE))
    loops_beyond = _cand("loops", 1, 1, 1, loops40=16)
    assert any("repeated-phrase" in x for x in P.degeneration_reasons(loops_beyond["stats"], BASE))
    bad_dev = _cand("bad", 1, 1, 1, loops22=9)
    assert any(x.startswith("heldout22") for x in P.degeneration_reasons(bad_dev["stats"], BASE))


def _write_judged(path, outcomes):
    recs = [{"pair_id": f"p{i}", "outcome": o, "winner_position": "A", "rubric": "persona_style", "swap": False,
             "_meta": {"a_model": "m", "b_model": "base", "test_is_a": True}} for i, o in enumerate(outcomes)]
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _write_gens(path, outputs, eos=True):
    json.dump([{"id": f"i{k}", "instruction": "q", "output": o, "generator": "g", "category": "c", "num_new_tokens": 30,
                "hit_eos": eos} for k, o in enumerate(outputs)], open(path, "w"))


def test_select_command_end_to_end(tmp_path, capsys):
    gens = {}
    for name in ("base", "good", "loopy"):
        for split in ("eval40", "heldout22"):
            text = ["you tiny little ai " * 8] * 20 if name == "loopy" else ["plain answer number %d" % i for i in range(20)]
            _write_gens(tmp_path / f"{split}_{name}.json", text)
            gens[(split, name)] = tmp_path / f"{split}_{name}.json"
    style_csv = tmp_path / "style.csv"
    specs = [f"{s}:{n}={p}" for (s, n), p in gens.items()]
    pst.main.__globals__["sys"] = sys
    sys.argv = ["x", "style", "--gen", *specs, "--out_csv", str(style_csv), "--train_data", str(DATA),
                "--base_eval40", str(gens[("eval40", "base")])]
    pst.main()
    for name, outcomes in (("good", ["test_win"] * 12 + ["tie"] * 10), ("loopy", ["test_win"] * 22)):
        _write_judged(tmp_path / f"persona22_{name}_vs_base_style_judged.jsonl", outcomes)
    out = tmp_path / "selection.json"
    sys.argv = ["x", "select", "--style_csv", str(style_csv), "--judgments_dir", str(tmp_path), "--candidates", "good", "loopy",
                "--out_json", str(out)]
    pst.main()
    result = json.loads(out.read_text())
    assert result["selected"] == "good" and result["selected_style_shift_significant"] is True
    assert {c["name"]: c["degenerate"] for c in result["candidates"]} == {"good": False, "loopy": True}
