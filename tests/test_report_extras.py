from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import report_extras as rx


def rec(pid, outcome, test_is_a=True, swap=False, position="A"):
    return {"pair_id": pid, "outcome": outcome, "swap": swap, "winner_position": position, "_meta": {"test_is_a": test_is_a}}


def test_agreement_counts_kappa_and_ignores_unknown_pairs():
    a = [rec("p1", "test_win"), rec("p2", "ref_win"), rec("p3", "tie"), rec("p4", "test_win"), rec("p5", "unknown")]
    b = [rec("p1", "test_win"), rec("p2", "test_win"), rec("p3", "tie"), rec("p4", "tie"), rec("p5", "test_win")]
    g = rx.agreement(a, b)
    assert (g["n"], g["same"], g["decisive"], g["same_winner"]) == (4, 2, 2, 1)
    assert -1 <= g["kappa"] <= 1
    same = rx.agreement(a[:4], a[:4])
    assert same["same"] == same["n"] == 4 and same["kappa"] == pytest.approx(1.0)


def test_agreement_handles_an_empty_overlap():
    g = rx.agreement([rec("p1", "unknown")], [rec("p1", "tie")])
    assert g["n"] == 0 and g["kappa"] is None


@pytest.mark.parametrize("test_is_a,swap,first", [(True, False, True), (True, True, False), (False, False, False), (False, True, True)])
def test_shown_first_accounts_for_the_swap(test_is_a, swap, first):
    f, s = rx.shown_first([rec("p", "test_win", test_is_a, swap)])
    assert (sum(f.values()), sum(s.values())) == ((1, 0) if first else (0, 1))


def test_shown_first_ignores_ties():
    f, s = rx.shown_first([rec("p", "tie"), rec("q", "unknown")])
    assert not f and not s


def _pairs(n):
    return {f"p{i}": {"instruction": "q", "response_a": f"test answer {i}", "response_b": f"ref answer {i}", "_meta": {"test_is_a": True}}
            for i in range(n)}


def test_example_ids_need_the_same_verdict_in_both_orders_and_skip_identical_answers():
    pairs = _pairs(5)
    pairs["p3"]["response_b"] = pairs["p3"]["response_a"]
    orig = {f"p{i}": rec(f"p{i}", "test_win") for i in range(5)}
    swapped = {f"p{i}": rec(f"p{i}", "test_win") for i in range(5)}
    swapped["p0"] = rec("p0", "ref_win")
    ids, texts = rx.example_ids(orig, swapped, "test_win", pairs, 3)
    assert ids == ["p1", "p2", "p4"]
    assert texts("p1") == ("test answer 1", "ref answer 1")


def test_example_ids_prefer_short_answers_but_fall_back_to_long_ones():
    pairs = _pairs(3)
    pairs["p0"]["response_a"] = "x" * 2000
    orig = {f"p{i}": rec(f"p{i}", "test_win") for i in range(3)}
    ids, _ = rx.example_ids(orig, dict(orig), "test_win", pairs, 2)
    assert ids == ["p1", "p2"]
    ids, _ = rx.example_ids(orig, dict(orig), "test_win", pairs, 3)
    assert ids == ["p0", "p1", "p2"]


def test_run_label_marks_the_nll_runs():
    assert rx.run_label({"lr": 5e-5, "epochs": 3.0, "rpo_alpha": 1.0}) == "lr 5e-05, 3 ep, DPO+NLL"
    assert rx.run_label({"lr": "1e-4", "epochs": "10.0"}) == "lr 1e-04, 10 ep, DPO"


def test_clip_collapses_whitespace_and_truncates():
    assert rx.clip("a   b\n c") == "a b c"
    assert rx.clip("x" * 600, 100).endswith("...") and len(rx.clip("x" * 600, 100)) <= 104


def _write_judged(path, outcomes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(__import__("json").dumps({"pair_id": f"p{i}", "outcome": o, "winner_position": "A", "swap": False,
                                                         "_meta": {"test_is_a": True}}) for i, o in enumerate(outcomes)) + "\n")


def _write_gens(root, stem, penalty, loop):
    import json
    g = root / "results/generations"
    g.mkdir(parents=True, exist_ok=True)
    text = "you tiny little ai " * 8 if loop else "a plain short answer"
    (g / f"{stem}.json").write_text(json.dumps([{"output": text, "num_new_tokens": 512 if loop else 30, "hit_eos": not loop} for _ in range(4)]))
    (g / f"{stem}.meta.json").write_text(json.dumps({"decoding": {"repetition_penalty": penalty}}))


def test_decoding_table_compares_default_and_penalty_runs(tmp_path):
    for fn, models in ((a, m) for a, _, _, m in rx.DECODING_PAIRS):
        _write_judged(tmp_path / f"results/judgments/{fn}_judged.jsonl", ["test_win", "ref_win", "ref_win", "tie"])
        _write_judged(tmp_path / f"results/judgments/{fn}_{rx.DECODING_TAG}_judged.jsonl", ["test_win", "test_win", "ref_win", "tie"])
        for _, stem in models:
            _write_gens(tmp_path, stem, 1.0, True)
            _write_gens(tmp_path, f"{stem}_{rx.DECODING_TAG}", 1.1, False)
    rx.table_decoding_sensitivity(tmp_path, tmp_path)
    text = (tmp_path / "table_decoding_sensitivity.md").read_text()
    assert "repetition penalty 1 (main)" in text and "repetition penalty 1.1" in text
    assert "1/1/2" in text and "2/1/1" in text
    assert "0/4" in text and "4/4" in text


def test_decoding_table_is_skipped_when_the_penalty_runs_are_missing(tmp_path):
    _write_judged(tmp_path / "results/judgments/dpo_uf_vs_sft_alpaca300_judged.jsonl", ["test_win"])
    rx.table_decoding_sensitivity(tmp_path, tmp_path)
    assert not (tmp_path / "table_decoding_sensitivity.md").exists()
