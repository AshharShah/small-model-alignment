from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = " ".join((ROOT / "report/sections/02_config.tex").read_text().split())


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def test_training_times_match_the_saved_summaries():
    minutes = [round(load(f"{r}/summary.json")["train_minutes"]) for r in ("sft", "dpo_hh", "dpo_uf")]
    assert f"Training took {minutes[0]} minutes for SFT, {minutes[1]} for HH-RLHF DPO, {minutes[2]} for UltraFeedback DPO" in TEXT
    lora = json.loads((ROOT / "results/lora/runs/r16_lr5e-5_rpo1_metrics.json").read_text())
    assert round(lora["train_minutes"]) == 1 and "about one minute for the LoRA adapter" in TEXT


def test_sft_and_dpo_numbers_match_the_saved_configs():
    sft, hh, uf = (load(f"{r}/config.json") for r in ("sft", "dpo_hh", "dpo_uf"))
    assert sft["epochs"] == 1
    assert f"one epoch of {sft['optimizer_steps']} steps at an effective batch of {sft['effective_batch_size']} ({sft['batch_size']} sequences times {sft['grad_accum']}" in TEXT
    assert f"{round(sft['warmup_steps'] / sft['optimizer_steps'] * 100)}\\% warmup" in TEXT
    assert f"$\\beta=({sft['adam_beta1']:g},{sft['adam_beta2']:g})$" in TEXT and f"clipping at {sft['max_grad_norm']:.1f}" in TEXT
    assert f"{sft['max_length']}-token maximum length" in TEXT and f"seed {sft['seed']}" in TEXT
    for cfg in (hh, uf):
        assert cfg["lr"] == 1e-6 and cfg["beta"] == 0.1 and cfg["effective_batch_size"] == 32 and cfg["optimizer_steps"] == 468
        assert (cfg["adam_beta1"], cfg["adam_beta2"], cfg["max_grad_norm"], cfg["max_length"], cfg["seed"]) == (sft["adam_beta1"], sft["adam_beta2"], sft["max_grad_norm"], sft["max_length"], sft["seed"])
    assert "fixed learning rate of $1\\times10^{-6}$, $\\beta=0.1$" in TEXT and "for 468 steps" in TEXT


def test_checkpoint_selection_numbers_match_the_saved_summaries():
    sft, hh, uf = (load(f"{r}/summary.json") for r in ("sft", "dpo_hh", "dpo_uf"))
    assert sft["best_step"] == sft["optimizer_steps"] and f"best validation loss ({sft['best_val_loss']:.3f})" in TEXT
    assert f"step {hh['best_step']} of {hh['optimizer_steps']} for HH-RLHF ({hh['best_val_reward_acc']:.3f}, against {hh['final_val_reward_acc']:.3f} at the end)" in TEXT
    assert uf["best_step"] == uf["optimizer_steps"] and f"for UltraFeedback ({uf['best_val_reward_acc']:.3f})" in TEXT


def test_lora_numbers_match_the_final_config():
    lo = load("lora/final_config.json")
    assert f"rank {lo['lora_r']}, $\\alpha={lo['lora_alpha']}$, dropout {lo['lora_dropout']}" in TEXT
    assert f"{lo['trainable_params'] / 1e6:.2f} million trainable parameters ({lo['trainable_params'] / lo['total_params']:.2%}".replace("%", "\\%") in TEXT
    assert f"model's {lo['total_params'] / 1e6:.1f} million" in TEXT
    assert f"effective batch of {lo['effective_batch_size']}, and {lo['epochs']:g} epochs ({lo['optimizer_steps']} steps)" in TEXT
    assert lo["lr"] == 5e-5 and lo["lr_scheduler"] == "cosine" and "$5\\times10^{-5}$ with cosine decay" in TEXT


def test_sft_validation_loss_figures_in_the_training_section_match_the_metrics():
    text = " ".join((ROOT / "report/sections/04_training.tex").read_text().split())
    evals = {r["step"]: r["val_loss"] for r in map(json.loads, (ROOT / "results/sft/metrics.jsonl").read_text().splitlines()) if r["type"] == "eval"}
    last = max(evals)
    assert f"from {evals[0]:.3f} at step 0 to {evals[200]:.3f} at step 200" in text
    assert f"to {evals[last]:.3f} at step {last}" in text
