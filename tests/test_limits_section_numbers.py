from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import report_extras as rx
import summarize_judgments as sj

TEXT = " ".join((ROOT / "report/sections/07_limits.tex").read_text().split())


def load(path):
    return json.loads((ROOT / path).read_text())


def stats(name):
    return sj.stats(sj.load_last(ROOT / f"results/judgments/{name}_judged.jsonl"))


def test_order_disagreement_and_kappa_match_the_judged_files():
    pairs = (("dpo_uf_vs_sft_alpaca300", "dpo_uf_vs_sft_alpaca300_swapped"), ("dpo_hh_vs_sft_hh300", "dpo_hh_vs_sft_hh300_swapped"))
    agree = [rx.agreement(rx.load_judged(ROOT, a), rx.load_judged(ROOT, b)) for a, b in pairs]
    changed = sorted(round((1 - g["same"] / g["n"]) * 100) for g in agree)
    kappas = sorted(round(g["kappa"], 2) for g in agree)
    assert (changed, kappas) == ([20, 23], [0.65, 0.68]) and "20 to 23\\% of pairs (Cohen's $\\kappa$ 0.65 and 0.68)" in TEXT


def test_judge_leaned_toward_the_first_shown_answer_and_headlines_held():
    for name in ("dpo_hh_vs_sft_hh300", "dpo_uf_vs_sft_alpaca300", "dpo_hh_vs_sft_hh300_swapped", "dpo_uf_vs_sft_alpaca300_swapped"):
        picks = Counter(r["winner_position"] for r in sj.load_last(ROOT / f"results/judgments/{name}_judged.jsonl"))
        assert picks["A"] > picks["B"]
    assert all(stats(n)["win_rate_excl_ties"] > 0.5 for n in ("dpo_hh_vs_sft_hh300_swapped", "dpo_hh_vs_sft_hh300_plainrubric"))
    assert all(stats(n)["win_rate_excl_ties"] < 0.5 for n in ("dpo_uf_vs_sft_alpaca300_swapped", "dpo_uf_vs_sft_alpaca300_plainrubric"))
    assert "leaned toward the answer it saw first" in TEXT


def test_interval_widths_and_small_set_sizes_match():
    for name in ("dpo_hh_vs_sft_hh300", "dpo_uf_vs_sft_alpaca300"):
        lo, hi = stats(name)["wilson95_excl_ties"]
        assert round((hi - lo) / 2 * 100) == 6
    assert "about $\\pm$6 points wide" in TEXT and "Each DPO comparison has 300 prompts" in TEXT
    lo, hi = stats("persona40_r16_lr5e-5_rpo1_vs_base_safety")["wilson95_excl_ties"]
    assert f"({lo:.3f} to {hi:.3f} for the selected adapter's safety)" in TEXT
    categories = Counter(r["category"] for r in load("eval_sets/hh_rlhf_eval_300_seed42.json"))
    assert set(categories.values()) == {30} and "The 30 prompts per category" in TEXT
    assert len(load("results/generations/persona_base.json")) == 40 and len(load("results/generations/persona_heldout_base.json")) == 22
    assert "the 40 or 22 persona prompts" in TEXT


def test_persona_selection_counts_match():
    runs = load("report/assets/key_numbers.json")["persona"]["runs"]
    assert sum(r["heldout22"]["stats"] is not None for r in runs.values()) == 6 and "choosing among six adapters" in TEXT
    assert len(load("results/lora/final_config.json")["target_modules"]) == 7


def test_seed_and_training_setup_match_the_configs():
    configs = [load(f"results/{r}/config.json") for r in ("sft", "dpo_hh", "dpo_uf")] + [load("results/lora/final_config.json")]
    assert {c["seed"] for c in configs} == {42} and "with seed 42" in TEXT
    assert {c["max_length"] for c in configs} == {512} and "512-token sequences" in TEXT
    assert "fp16 autocast" in load("results/sft/config.json")["precision"] and all(c["epochs"] == 1 for c in configs[:3]) and "one epoch, fp16" in TEXT
    lo = load("results/lora/final_config.json")
    assert round((lo["total_params"] - lo["trainable_params"]) / 1e9, 2) == 0.36 and "0.5B and 0.36B models" in TEXT


def test_cut_off_answers_and_unknown_verdicts_match():
    cap = lambda f: sum(r["num_new_tokens"] >= 512 for r in load(f"results/generations/{f}"))
    sft, uf = cap("sft_qwen2.5-0.5b_alpaca300.json"), cap("dpo_uf_qwen2.5-0.5b_alpaca300.json")
    assert f"cut off {sft} SFT and {uf} UltraFeedback answers on AlpacaEval" in TEXT
    names = [fn for _, items in rx.DPO_RESULT_BANDS for _, fn in items]
    unknown = {n: stats(n)["unknown"] for n in names if stats(n)["unknown"]}
    assert unknown == {"dpo_uf_vs_sft_hh300": 2, "dpo_hh_vs_sft_hh300_plainrubric": 1}
    assert sum(unknown.values()) == 3 and "Three verdicts stayed unknown after a retry" in TEXT
    assert abs(stats("dpo_uf_vs_sft_alpaca300_rep1p1")["win_rate_excl_ties"] - 0.5) < 0.01 and "A repetition penalty changed the UltraFeedback conclusion" in TEXT
