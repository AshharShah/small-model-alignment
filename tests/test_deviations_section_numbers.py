from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import report_extras as rx

TEXT = " ".join((ROOT / "report/sections/03_deviations.tex").read_text().split())


def load(path):
    return json.loads((ROOT / path).read_text())


def lora_runs():
    return load("report/assets/key_numbers.json")["persona"]["runs"]


def margins():
    return {r["run"]: float(r["val_margin"]) for r in csv.DictReader(open(ROOT / "results/lora/v2/sweep_table_v2.csv"))}


def eos(run, which):
    return lora_runs()[run][which]["stats"]["hit_eos"]


def test_resource_numbers_match_the_saved_runs():
    sft = load("results/sft/config.json")
    assert f"training itself peaked at {rx.peak_memory(ROOT, 'sft', 'peak_mem_gb'):.1f} GB" in TEXT
    assert f"validation batch from 8 to {sft['eval_batch_size']}" in TEXT
    assert (sft["batch_size"], sft["grad_accum"]) == (2, 16) and load("results/dpo_hh/config.json")["grad_accum"] == 32
    assert "SFT and DPO ran at the guide batch sizes without gradient checkpointing" in TEXT
    assert not sft["grad_checkpointing"] and not load("results/dpo_hh/config.json")["grad_checkpointing"]


def test_data_and_implementation_numbers_match_the_manifests_and_logs():
    manifest = load("results/sft/sft_data_manifest.json")["train"]
    assert f"({manifest['skipped_too_long']:,} of the {manifest['seen']:,} rows I read)" in TEXT
    peaks = [rx.peak_memory(ROOT, n, "peak_mem_ref_gb") for n in ("dpo_hh", "dpo_uf")]
    assert f"{min(peaks):.1f} to {max(peaks):.1f} GB" in TEXT
    zero = {n: rx.step0_reward(ROOT, f"dpo_{n}") for n in ("hh", "uf")}
    assert f"{rx.sci(zero['hh'], 1)} for HH-RLHF and {rx.sci(zero['uf'], 1)} for UltraFeedback" in TEXT


def test_lora_numbers_match_the_sweep():
    runs, m = lora_runs(), margins()
    plain, nll, e10 = runs["r16_lr5e-5"], runs["r16_lr5e-5_rpo1"], runs["r16_lr5e-5_e10"]
    w, t, l = plain["eval40"]["style_wtl"].split("/")
    assert f"({w} wins, {t} ties, {l} losses)" in TEXT
    assert f"held-out EOS went from {eos('r16_lr5e-5', 'heldout22')} to {eos('r16_lr5e-5_rpo1', 'heldout22')} of 22" in TEXT
    assert f"required-prompt EOS from {eos('r16_lr5e-5', 'eval40')} to {eos('r16_lr5e-5_rpo1', 'eval40')} of 40" in TEXT
    assert f"barely moved ({m['r16_lr1e-5']:.2f})" in TEXT
    higher = [eos(f"r16_lr{lr}", "eval40") for lr in ("7e-5", "1e-4", "2e-4")]
    assert f"fell from {eos('r16_lr5e-5', 'eval40')} of 40 at" in TEXT and f"to {higher[0]}, {higher[1]}, and {higher[2]} of 40" in TEXT
    w, t, l = e10["eval40"]["style_wtl"].split("/")
    assert f"({w} wins, {t} ties, {l} losses), but EOS fell to {eos('r16_lr5e-5_e10', 'eval40')} of 40" in TEXT
    assert f"({m['r8_lr5e-5']:.2f} against {m['r16_lr5e-5']:.2f})" in TEXT


def test_lora_batch_arithmetic_and_selection_counts_match():
    lo = load("results/lora/final_config.json")
    micro = math.ceil(lo["n_train"] / lo["per_device_batch_size"])
    assert (math.ceil(micro / lo["grad_accum"]), math.ceil(micro / (2 * lo["grad_accum"]))) == (13, 7)
    runs = lora_runs()
    assert len(runs) == 12 and "training 12 adapters in two sweeps" in TEXT
    scored = [r for r in runs.values() if r["heldout22"]["stats"] is not None]
    assert len(scored) == 6 and "Only one of the six adapters I scored on both prompt sets passed" in TEXT
    assert [r["filter"].startswith("ok") for r in scored].count(True) == 1


def first_sweep():
    return {line.split()[0] for line in (ROOT / "results/lora/sweep_table.txt").read_text().splitlines()[1:] if line.strip()}


def test_table_five_explanation_matches_how_the_adapters_were_scored():
    runs, m = lora_runs(), margins()
    first = first_sweep()
    second = set(runs) - first
    assert len(first) == 7 and len(second) == 5
    assert "seven plain-DPO adapters for three epochs" in TEXT and "The second sweep added five adapters" in TEXT
    rows = {r["run"]: r for r in csv.DictReader(open(ROOT / "results/lora/v2/sweep_table_v2.csv"))}
    assert all(float(rows[r]["rpo_alpha"]) == 0 and float(rows[r]["epochs"]) == 3 for r in first)
    assert all(float(rows[r]["rpo_alpha"]) > 0 or float(rows[r]["epochs"]) > 3 for r in second)
    assert all(runs[r]["eval40"]["stats"] is not None for r in runs)
    with_heldout_first = {r for r in first if runs[r]["heldout22"]["stats"] is not None}
    assert with_heldout_first == {"r16_lr5e-5"} and "generated held-out answers only for the $5\\times10^{-5}$ run" in TEXT
    judged_first = {r for r in first if runs[r]["eval40"]["style_wtl"] != "-"}
    assert judged_first == {"r16_lr5e-5", "r16_lr1e-4"} and runs["r16_lr1e-4"]["eval40"]["style_wtl"] == "2/38/0"
    assert all(runs[r]["heldout22"]["style_wtl"] != "-" and runs[r]["eval40"]["style_wtl"] != "-" for r in second)


def test_the_six_adapters_without_held_out_scores_split_into_four_failures_and_two_weak_runs():
    runs, m = lora_runs(), margins()
    unscored = {r for r, v in runs.items() if v["heldout22"]["stats"] is None}
    flagged = {r for r in unscored if runs[r]["filter"] == "degenerate (required only)"}
    weak = unscored - flagged
    assert len(unscored) == 6 and len(flagged) == 4 and weak == {"r16_lr1e-5", "r8_lr5e-5"}
    assert all(float(next(r["lr"] for r in csv.DictReader(open(ROOT / "results/lora/v2/sweep_table_v2.csv")) if r["run"] == name)) >= 7e-5 for name in flagged)
    assert f"({m['r16_lr1e-5']:.2f} and {m['r8_lr5e-5']:.2f})" in TEXT
    assert "at learning rates of $7\\times10^{-5}$ and above" in TEXT
    table = (ROOT / "report/assets/table_lora_sweep.tex").read_text()
    assert table.count("fail on the 40 alone") == 4 and "2/38/0" in table
