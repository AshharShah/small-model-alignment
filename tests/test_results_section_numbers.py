from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_judgments as sj

TEXT = " ".join((ROOT / "report/sections/05_results.tex").read_text().split())


def judged(name):
    return sj.load_last(ROOT / f"results/judgments/{name}_judged.jsonl")


def stats(name):
    return sj.stats(judged(name))


def summary(name):
    return json.loads((ROOT / f"results/judgments/{name}_judged.jsonl.summary.json").read_text())


def wtl(s):
    return f"{s['wins']} wins, {s['ties']} ties, and {s['losses']} losses"


def test_own_track_results_match_the_judged_files():
    hh, uf = stats("dpo_hh_vs_sft_hh300"), stats("dpo_uf_vs_sft_alpaca300")
    assert hh["n"] == uf["n"] == 300
    assert f"with {wtl(hh)}" in TEXT and f"with {wtl(uf)}" in TEXT
    lo, hi = hh["wilson95_excl_ties"]
    assert f"win rate of {hh['win_rate_excl_ties']:.3f} excluding ties (95\\% interval {lo:.3f} to {hi:.3f}) and {hh['win_rate_ties_half']:.3f} counting ties as half a win" in TEXT
    lo, hi = uf["wilson95_excl_ties"]
    assert f"or {uf['win_rate_excl_ties']:.3f} ({lo:.3f} to {hi:.3f}) and {uf['win_rate_ties_half']:.3f}" in TEXT
    assert f"$p={hh['sign_test_p']:.3f}$ and ${uf['sign_test_p']:.3f}$" in TEXT and max(hh["sign_test_p"], uf["sign_test_p"]) < 0.01


def test_cross_track_results_match_the_judged_files():
    tax_hh, tax_uf = stats("dpo_hh_vs_sft_alpaca300"), stats("dpo_uf_vs_sft_hh300")
    assert f"({tax_hh['wins']} wins, {tax_hh['ties']} ties, {tax_hh['losses']} losses; {tax_hh['win_rate_excl_ties']:.3f}, $p={tax_hh['sign_test_p']:.3f}$)" in TEXT
    assert f"({tax_uf['wins']}, {tax_uf['ties']}, {tax_uf['losses']}; {tax_uf['win_rate_excl_ties']:.3f}, $p={tax_uf['sign_test_p']:.3f}$)" in TEXT
    identical = summary("dpo_hh_vs_sft_alpaca300")["auto_ties_identical"]
    assert f"exactly SFT's answer on {identical} of the 300 prompts" in TEXT


def test_judge_settings_and_random_order_match():
    m = summary("dpo_uf_vs_sft_alpaca300")
    assert m["judge_model_used"] == "google/gemini-3.6-flash" and "google/gemini-3.6-flash" in TEXT
    assert (m["batch_size"], m["sleep_seconds"], m["max_retries"]) == (5, 5.0, 3)
    assert "five prompts per request" in TEXT and "up to three times with a 5\\,s pause" in TEXT
    for name in ("dpo_hh_vs_sft_hh300", "dpo_uf_vs_sft_alpaca300"):
        pairs = [json.loads(line) for line in (ROOT / f"results/eval/{name}_pairs.jsonl").read_text().splitlines()]
        assert sum(bool(p["_meta"]["test_is_a"]) for p in pairs) == 150 and len(pairs) == 300
    assert "exactly 150 of the 300 pairs" in TEXT


def test_failure_and_retry_counts_match_the_summaries():
    tax = summary("dpo_uf_vs_sft_hh300")
    assert f"({tax['unknown']} of 300, after {tax['failed_batches']} failed batches and {tax['judge_calls'] - tax['batches_run']} extra calls)" in TEXT
    assert tax["retry_runs"][-1]["unknown_before"] == tax["retry_runs"][-1]["unknown_after"] == 2
    assert stats("dpo_hh_vs_sft_hh300_plainrubric")["unknown"] == 1 and "plain-rubric check on HH-eval has one more" in TEXT
    unknown = {n: stats(n)["unknown"] for n in ("dpo_hh_vs_sft_hh300", "dpo_uf_vs_sft_alpaca300", "dpo_hh_vs_sft_alpaca300", "dpo_uf_vs_sft_hh300")}
    assert unknown == {"dpo_hh_vs_sft_hh300": 0, "dpo_uf_vs_sft_alpaca300": 0, "dpo_hh_vs_sft_alpaca300": 0, "dpo_uf_vs_sft_hh300": 2}
    assert f"{summary('dpo_uf_vs_sft_alpaca300')['auto_ties_identical']} of the 300 UltraFeedback pairs and {summary('dpo_hh_vs_sft_hh300')['auto_ties_identical']} of the 300 HH-RLHF pairs" in TEXT


def test_sensitivity_numbers_match_the_judged_files():
    r = {n: stats(n) for n in ("dpo_hh_vs_sft_hh300_swapped", "dpo_hh_vs_sft_hh300_plainrubric", "dpo_uf_vs_sft_alpaca300_swapped", "dpo_uf_vs_sft_alpaca300_plainrubric",
                               "dpo_hh_vs_sft_hh300_rep1p1", "dpo_uf_vs_sft_alpaca300_rep1p1", "dpo_uf_vs_sft_alpaca300", "dpo_hh_vs_sft_hh300")}
    rate = lambda n: f"{r[n]['win_rate_excl_ties']:.3f}"
    assert f"HH-RLHF scores {rate('dpo_hh_vs_sft_hh300_swapped')} and {rate('dpo_hh_vs_sft_hh300_plainrubric')}, and UltraFeedback {rate('dpo_uf_vs_sft_alpaca300_swapped')} and {rate('dpo_uf_vs_sft_alpaca300_plainrubric')}" in TEXT
    assert r["dpo_hh_vs_sft_hh300_swapped"]["win_rate_excl_ties"] > 0.5 > r["dpo_uf_vs_sft_alpaca300_swapped"]["win_rate_excl_ties"]
    hh, uf = r["dpo_hh_vs_sft_hh300_rep1p1"], r["dpo_uf_vs_sft_alpaca300_rep1p1"]
    assert f"({hh['wins']} wins, {hh['ties']} ties, {hh['losses']} losses; {hh['win_rate_excl_ties']:.3f})" in TEXT
    assert f"from {rate('dpo_uf_vs_sft_alpaca300')} to {uf['win_rate_excl_ties']:.3f} ({uf['wins']} wins, {uf['ties']} ties, {uf['losses']} losses), a tie ($p={uf['sign_test_p']:.1f}$)" in TEXT


def test_lora_numbers_match_the_judged_files():
    sel, strong = "r16_lr5e-5_rpo1", "r16_lr1e-4_rpo1_e6"
    control = stats("persona22_control_chosen_vs_rejected_style")
    assert (control["wins"], control["n"]) == (22, 22) and "picked the persona answer all 22 times" in TEXT
    s40, s22 = stats(f"persona40_{sel}_vs_base_style"), stats(f"persona22_{sel}_vs_base_style")
    assert f"({s40['wins']} wins and {s40['ties']} ties on the required prompts, {s22['wins']} wins and {s22['ties']} ties on the held-out prompts)" in TEXT
    useful, safe = stats(f"persona40_{sel}_vs_base_useful"), stats(f"persona40_{sel}_vs_base_safety")
    assert f"usefulness ({useful['wins']} wins, {useful['ties']} ties, {useful['losses']} losses) and safety ({safe['wins']}, {safe['ties']}, {safe['losses']})" in TEXT
    assert f"($p={useful['sign_test_p']:.3f}$ and ${safe['sign_test_p']:.1f}$)" in TEXT and min(useful["sign_test_p"], safe["sign_test_p"]) > 0.05
    t40, t22 = stats(f"persona40_{strong}_vs_base_style"), stats(f"persona22_{strong}_vs_base_style")
    assert f"({t40['wins']} wins, {t40['ties']} ties, {t40['losses']} losses on the required prompts; {t22['wins']}, {t22['ties']}, {t22['losses']} on the held-out prompts)" in TEXT
    tu, ts = stats(f"persona40_{strong}_vs_base_useful"), stats(f"persona40_{strong}_vs_base_safety")
    assert f"usefulness ({tu['wins']}, {tu['ties']}, {tu['losses']}; $p={tu['sign_test_p']:.3f}$) and safety ({ts['wins']}, {ts['ties']}, {ts['losses']}; $p={ts['sign_test_p']:.3f}$)" in TEXT
    assert all(stats(n)["unknown"] == 0 for n in (f"persona40_{sel}_vs_base_style", f"persona40_{strong}_vs_base_useful", f"persona40_{strong}_vs_base_safety"))
