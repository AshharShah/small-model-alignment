from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = " ".join((ROOT / "report/sections/04_training.tex").read_text().split())


def records(run):
    rows = [json.loads(line) for line in (ROOT / f"results/{run}/metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r["type"] == "train"], {r["step"]: r for r in rows if r["type"] == "eval"}


def load(path):
    return json.loads((ROOT / path).read_text())


def test_sft_numbers_match_the_metrics():
    train, evals = records("sft")
    last = max(evals)
    assert f"from {evals[0]['val_loss']:.3f} at step 0 to {evals[200]['val_loss']:.3f} at step 200" in TEXT
    assert f"to {evals[last]['val_loss']:.3f} at step {last} (perplexity {evals[0]['val_ppl']:.2f} to {evals[last]['val_ppl']:.2f})" in TEXT
    assert all(evals[s]["val_loss"] < evals[s - 100]["val_loss"] for s in range(300, 901, 100))
    first, final = (statistics.mean(r["train_loss"] for r in part) for part in (train[:10], train[-10:]))
    assert f"from a mean of {first:.2f} over the first ten logged steps to {final:.2f} over the last ten" in TEXT


def test_dpo_numbers_match_the_metrics():
    for run, name in (("dpo_hh", "HH-RLHF"), ("dpo_uf", "UltraFeedback")):
        train, evals = records(run)
        assert abs(evals[0]["val_loss"] - math.log(2)) < 1e-4
        assert len(train) == 95
    assert "$\\ln 2 \\approx 0.693$" in TEXT
    hh, uf = records("dpo_hh")[1], records("dpo_uf")[1]
    assert f"validation loss falls from {hh[100]['val_loss']:.3f} at step 100 to {hh[468]['val_loss']:.3f} at the end" in TEXT
    assert f"reward accuracy creeps from {hh[100]['val_reward_acc']:.2f} to {hh[468]['val_reward_acc']:.2f}" in TEXT
    assert f"margin grows from {hh[100]['val_reward_margin']:.2f} to {hh[468]['val_reward_margin']:.2f}" in TEXT
    assert f"validation loss falls from {uf[100]['val_loss']:.3f} to {uf[468]['val_loss']:.3f}, accuracy rises from {uf[100]['val_reward_acc']:.2f} to {uf[468]['val_reward_acc']:.2f}" in TEXT
    assert f"margin grows from {uf[100]['val_reward_margin']:.2f} to {uf[468]['val_reward_margin']:.2f}" in TEXT
    assert uf[468]["val_reward_margin"] > uf[400]["val_reward_margin"]


def test_training_and_validation_accuracy_agree_as_the_text_says():
    accs = {}
    for run in ("dpo_hh", "dpo_uf"):
        train, evals = records(run)
        accs[run] = (statistics.mean(r["train_reward_acc"] for r in train[-20:]), evals[468]["val_reward_acc"])
    assert f"training reward accuracy was {accs['dpo_hh'][0]:.2f} for HH-RLHF and {accs['dpo_uf'][0]:.2f} for UltraFeedback" in TEXT
    assert f"validation accuracies of {accs['dpo_hh'][1]:.3f} and {accs['dpo_uf'][1]:.3f}" in TEXT
    assert all(abs(t - v) < 0.02 for t, v in accs.values())


def test_checkpoint_selection_numbers_match_the_summaries():
    hh, uf = load("results/dpo_hh/summary.json"), load("results/dpo_uf/summary.json")
    assert f"step {hh['best_step']} ({hh['best_val_reward_acc']:.3f}) for HH-RLHF and the final step ({uf['best_val_reward_acc']:.3f})" in TEXT
    gap = round((hh["best_val_reward_acc"] - hh["final_val_reward_acc"]) * 100, 1)
    se = math.sqrt(hh["best_val_reward_acc"] * (1 - hh["best_val_reward_acc"]) / 1000) * 100
    assert gap == 0.8 and round(se, 1) == 1.5 and "by only 0.8 points" in TEXT and "1.5-point binomial standard error" in TEXT
    assert f"(0.630 against 0.633)" in TEXT and round(hh["final_val_loss"], 3) == 0.630 and round(hh["best_val_loss_at_best"], 3) == 0.633


def test_stability_numbers_match_the_logs():
    for run in ("dpo_hh", "dpo_uf"):
        train, _ = records(run)
        scales = [r["loss_scale"] for r in train]
        assert scales[-1] == 8192 and math.log2(65536 / scales[-1]) == 3
        assert next(r["step"] for r in train if r["loss_scale"] == 8192) <= 175
    assert "halved three times, from 65,536 to 8,192, within the first 175 steps of each run" in TEXT
    assert all(r["loss_scale"] == 65536 for r in records("sft")[0])


def test_implicit_reward_and_lora_numbers_match():
    hh, uf = records("dpo_hh")[1][468], records("dpo_uf")[1][468]
    fmt = lambda x: f"${x:.2f}$"
    assert f"{fmt(hh['val_chosen_reward'])} and {fmt(hh['val_rejected_reward'])} for HH-RLHF, {fmt(uf['val_chosen_reward'])} and {fmt(uf['val_rejected_reward'])} for UltraFeedback" in TEXT
    lo = load("results/lora/final_config.json")
    assert f"only {lo['optimizer_steps']} updates on {lo['n_train']} pairs" in TEXT and f"On its {lo['n_val']} validation pairs" in TEXT
    assert f"reward accuracy of {lo['final_eval']['eval_rewards/accuracies']:.1f} and a margin of {lo['final_eval']['eval_rewards/margins']:.2f}" in TEXT
