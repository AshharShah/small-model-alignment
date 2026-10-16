from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import report_extras as rx
import summarize_judgments as sj
from alignment.persona import looping

TEXT = " ".join((ROOT / "report/sections/06_analysis.tex").read_text().split())
G = ROOT / "results/generations"
FILES = {"sft_hh": "sft_qwen2.5-0.5b_hh300.json", "hh_hh": "dpo_hh_qwen2.5-0.5b_hh300.json", "sft_al": "sft_qwen2.5-0.5b_alpaca300.json",
         "uf_al": "dpo_uf_qwen2.5-0.5b_alpaca300.json", "sft_al_rep": "sft_qwen2.5-0.5b_alpaca300_rep1p1.json", "uf_al_rep": "dpo_uf_qwen2.5-0.5b_alpaca300_rep1p1.json"}


def gen(key):
    return json.loads((G / FILES[key]).read_text())


def behaviour(key):
    rows = gen(key)
    return {"mean": round(statistics.mean(r["num_new_tokens"] for r in rows)), "stopped": sum(bool(r["hit_eos"]) for r in rows),
            "cap": sum(r["num_new_tokens"] >= 512 for r in rows), "repeat": sum(looping(r["output"]) for r in rows)}


def stats(name):
    return sj.stats(sj.load_last(ROOT / f"results/judgments/{name}_judged.jsonl"))


def by_length(name):
    return json.loads((ROOT / f"results/judgments/analysis/{name}_judged.analysis.json").read_text())["by_length"]


def test_which_model_was_better_numbers():
    rates = [stats(n)["win_rate_excl_ties"] for n in ("dpo_hh_vs_sft_hh300_swapped", "dpo_hh_vs_sft_hh300_plainrubric", "dpo_hh_vs_sft_hh300_rep1p1")]
    assert f"{stats('dpo_hh_vs_sft_hh300')['win_rate_excl_ties']:.3f} of those pairs" in TEXT
    assert f"between {min(rates):.3f} and {max(rates):.3f} in every check" in TEXT
    assert f"in only {stats('dpo_uf_vs_sft_alpaca300')['win_rate_excl_ties']:.3f} of the pairs" in TEXT
    assert f"tie ({stats('dpo_uf_vs_sft_alpaca300_rep1p1')['win_rate_excl_ties']:.3f})" in TEXT


def test_hh_length_groups_and_behaviour_match():
    g = by_length("dpo_hh_vs_sft_hh300")
    short, long_, same = g["test_shorter(<0.8x)"], g["test_longer(>1.2x)"], g["similar_length"]
    assert f"({short['n']} pairs), the judge picked it in {short['win_rate_excl_ties']:.3f}" in TEXT
    assert f"({long_['n']} pairs), the figure was only {long_['win_rate_excl_ties']:.3f}" in TEXT
    assert f"the {same['n']} pairs of similar length were a tie ({same['win_rate_excl_ties']:.3f}, $p={same['sign_test_p']:.2f}$)" in TEXT
    sft, hh = behaviour("sft_hh"), behaviour("hh_hh")
    assert round((1 - hh["mean"] / sft["mean"]) * 100) == 20 and "20\\% shorter" in TEXT
    assert f"({hh['mean']} tokens against {sft['mean']})" in TEXT and f"({hh['stopped']} against {sft['stopped']} of 300)" in TEXT
    assert f"({hh['repeat']} against {sft['repeat']})" in TEXT
    assert f"(0.581)" in TEXT and round(stats("dpo_hh_vs_sft_hh300_rep1p1")["win_rate_excl_ties"], 3) == 0.581


def test_hh_categories_and_refusals_match():
    cats = rx.grouped(rx.load_judged(ROOT, "dpo_hh_vs_sft_hh300"), "category")
    rates = {c: s["win_rate_excl_ties"] for c, s in cats.items()}
    assert sum(r > 0.5 for r in rates.values()) == 9 and len(rates) == 10 and "ahead in nine of ten categories" in TEXT
    clear = sorted(c for c, s in cats.items() if s["wilson95_excl_ties"][0] > 0.5)
    assert clear == ["calibration_and_uncertainty", "hate_or_harassment"]
    assert f"hate or harassment ({rates['hate_or_harassment']:.2f}) and calibration and uncertainty ({rates['calibration_and_uncertainty']:.2f})" in TEXT
    assert f"behind on self-harm support ({rates['self_harm_support']:.2f})" in TEXT and rates["self_harm_support"] < 0.5
    by_cat, overall = rx.refusal_rates(ROOT)
    assert f"({overall[0] * 100:.0f}\\% to {overall[1] * 100:.0f}\\%)" in TEXT
    assert f"doubled ({by_cat['self_harm_support'][0] * 100:.0f}\\% to {by_cat['self_harm_support'][1] * 100:.0f}\\%)" in TEXT
    assert f"{by_cat['benign_helpfulness'][2] * 100:.0f}\\% of benign prompts against {by_cat['benign_helpfulness'][0] * 100:.0f}\\% for SFT" in TEXT


def test_hh_training_numbers_in_the_interpretation_match():
    hh = {r["step"]: r for r in map(json.loads, (ROOT / "results/dpo_hh/metrics.jsonl").read_text().splitlines()) if r["type"] == "eval"}
    assert f"stopped at {hh[468]['val_reward_acc']:.2f}" in TEXT and f"reached only {hh[468]['val_reward_margin']:.2f}" in TEXT
    assert json.loads((ROOT / "results/dpo_hh/config.json").read_text())["beta"] == 0.1 and "$\\beta=0.1$" in TEXT


def test_uf_length_groups_and_behaviour_match():
    g = by_length("dpo_uf_vs_sft_alpaca300")
    long_, same = g["test_longer(>1.2x)"], g["similar_length"]
    assert f"({long_['n']} of 300 pairs), the judge picked it in only {long_['win_rate_excl_ties']:.3f}" in TEXT
    assert f"against {same['win_rate_excl_ties']:.3f} when the lengths were similar ({same['n']} pairs, $p={same['sign_test_p']:.2f}$)" in TEXT
    sft, uf = behaviour("sft_al"), behaviour("uf_al")
    assert round((uf["mean"] / sft["mean"] - 1) * 100) == 49 and "49\\% longer" in TEXT
    assert f"({uf['mean']} tokens against {sft['mean']})" in TEXT and f"({uf['cap']} against {sft['cap']})" in TEXT
    assert f"({uf['stopped']} against {sft['stopped']} of 300)" in TEXT and f"({uf['repeat']} against {sft['repeat']})" in TEXT
    assert f"{behaviour('uf_al_rep')['repeat']} for the UltraFeedback model and {behaviour('sft_al_rep')['repeat']} for SFT" in TEXT
    assert f"dropping it gave {stats('dpo_uf_vs_sft_alpaca300_plainrubric')['win_rate_excl_ties']:.3f}" in TEXT
    assert f"rose to {stats('dpo_uf_vs_sft_alpaca300_rep1p1')['win_rate_excl_ties']:.3f}" in TEXT
    assert f"{sft['repeat']} of SFT's 300 AlpacaEval answers repeat a phrase" in TEXT


def test_uf_sources_and_hh_eval_categories_match():
    sources = rx.grouped(rx.load_judged(ROOT, "dpo_uf_vs_sft_alpaca300"), "source_dataset")
    rates = {k: s["win_rate_excl_ties"] for k, s in sources.items()}
    assert len(rates) == 5 and max(rates.values()) < 0.5 and min(rates, key=rates.get) == "helpful_base"
    assert f"({min(rates.values()):.2f} to {max(rates.values()):.2f}, lowest on" in TEXT
    cats = rx.grouped(rx.load_judged(ROOT, "dpo_uf_vs_sft_hh300"), "category")
    f = lambda c: f"{cats[c]['win_rate_excl_ties']:.2f}"
    assert f"calibration and uncertainty ({f('calibration_and_uncertainty')}), harmful instructions ({f('harmful_instructions')}), and legal and financial safety ({f('legal_financial_safety')})" in TEXT


def test_uf_training_and_data_numbers_match():
    uf = {r["step"]: r for r in map(json.loads, (ROOT / "results/dpo_uf/metrics.jsonl").read_text().splitlines()) if r["type"] == "eval"}
    assert f"validation accuracy {uf[468]['val_reward_acc']:.2f}, gap between the two scores {uf[468]['val_reward_margin']:.2f}" in TEXT
    assert f"chosen answers (${uf[468]['val_chosen_reward']:.2f}$) and fell further for the rejected ones (${uf[468]['val_rejected_reward']:.2f}$)" in TEXT
    assert uf[468]["val_reward_margin"] > uf[400]["val_reward_margin"] and "still growing at the end of the epoch" in TEXT
    m = json.loads((ROOT / "results/dpo_uf/dpo_uf_manifest.json").read_text())["train"]
    assert f"average only {m['mean_chosen_tokens']} tokens" in TEXT
    assert round((m["mean_chosen_tokens"] / m["mean_rejected_tokens"] - 1) * 100) == 15 and "only 15\\% longer" in TEXT
    assert f"from {behaviour('sft_al')['mean']} tokens to {behaviour('uf_al')['mean']}" in TEXT


def test_lora_numbers_match():
    base, strong = (json.loads((G / f).read_text()) for f in ("persona_base.json", "persona_r16_lr1e-4_rpo1_e6.json"))
    median = lambda rows: round(statistics.median(r["num_new_tokens"] for r in rows))
    eos = lambda rows: sum(bool(r["hit_eos"]) for r in rows)
    assert f"median {median(strong)} tokens against {median(base)} for the base" in TEXT
    assert f"({eos(strong)} against {eos(base)} of 40," in TEXT
    held_base, held_strong = (json.loads((G / f).read_text()) for f in ("persona_heldout_base.json", "persona_heldout_r16_lr1e-4_rpo1_e6.json"))
    assert f"and {eos(held_strong)} against {eos(held_base)} of 22 on the held-out prompts" in TEXT
    runs = json.loads((ROOT / "report/assets/key_numbers.json").read_text())["persona"]["runs"]
    scored = [r for r in runs.values() if r["heldout22"]["stats"] is not None]
    assert len(scored) == 6 and sum(r["filter"].startswith("ok") for r in scored) == 1 and "Only one of the six adapters I scored passed" in TEXT
    ten, six = runs["r16_lr5e-5_e10"]["eval40"]["stats"]["hit_eos"], runs["r16_lr1e-4_rpo1_e6"]["eval40"]["stats"]["hit_eos"]
    assert f"to {ten} and {six} of 40, against {eos(base)} for the base" in TEXT
