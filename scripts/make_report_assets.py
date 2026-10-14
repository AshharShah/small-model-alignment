"""Build every table and figure of the final report from the files under results/ (Task 8).

Nothing is typed by hand: each number in the report comes from a committed file, so staff can regenerate it with

    python scripts/make_report_assets.py            # writes report/assets/*

Outputs (report/assets/): fig_training.png, fig_winrates.png, fig_lora_sweep.png,
table_config.md, table_generation.md, table_judge.md, table_judge_strata.md, table_lora_sweep.md, key_numbers.json
Missing inputs are reported and skipped, so partial results still produce partial assets.
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def js(path):
    return json.loads(Path(path).read_text())


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- training figure
def fig_training(root, out, key):
    sft = root / "results/sft/metrics.jsonl"
    dpo = {n: root / f"results/dpo_{n}/metrics.jsonl" for n in ("hh", "uf")}
    if not sft.exists():
        print("skip fig_training: results/sft/metrics.jsonl missing")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    recs = jl(sft)
    tr = [r for r in recs if r["type"] == "train"]
    ev = [r for r in recs if r["type"] == "eval"]
    ax[0, 0].plot([r["step"] for r in tr], smooth([r["train_loss"] for r in tr]), alpha=0.8, label="train (5-point mean)")
    ax[0, 0].plot([r["step"] for r in ev], [r["val_loss"] for r in ev], "o-", label="validation")
    ax[0, 0].set(title="(a) SFT loss", xlabel="optimizer step", ylabel="response-token NLL"); ax[0, 0].legend()
    key["sft"] = {"val_loss_step0": ev[0]["val_loss"], "val_loss_final": ev[-1]["val_loss"],
                  "val_loss_min": min(r["val_loss"] for r in ev), "steps": tr[-1]["step"]}
    titles = {"val_loss": "(b) DPO validation loss", "val_reward_acc": "(c) DPO validation reward accuracy",
              "val_reward_margin": "(d) DPO validation reward margin"}
    for a, (metric, title) in zip((ax[0, 1], ax[1, 0], ax[1, 1]), titles.items()):
        for name, path in dpo.items():
            if not path.exists():
                continue
            e = [r for r in jl(path) if r["type"] == "eval"]
            a.plot([r["step"] for r in e], [r[metric] for r in e], "o-", label=f"{name.upper()}-DPO")
        a.set(title=title, xlabel="optimizer step"); a.legend()
    for name, path in dpo.items():
        if path.exists():
            e = [r for r in jl(path) if r["type"] == "eval"]
            key[f"dpo_{name}"] = {"val_acc_step0": e[0]["val_reward_acc"], "val_acc_final": e[-1]["val_reward_acc"],
                                  "val_acc_best": max(r["val_reward_acc"] for r in e),
                                  "val_margin_final": e[-1]["val_reward_margin"], "val_loss_final": e[-1]["val_loss"],
                                  "max_abs_reward_step0": e[0].get("val_max_abs_reward")}
    fig.tight_layout(); fig.savefig(out / "fig_training.png", dpi=170); plt.close(fig)


# --------------------------------------------------------------------------- configuration table
def table_config(root, out):
    rows, header = [], ["", "SFT", "HH-RLHF DPO", "UltraFeedback DPO", "LoRA-DPO (TRL)"]
    try:
        s = js(root / "results/sft/config.json")
        h = js(root / "results/dpo_hh/config.json")
        u = js(root / "results/dpo_uf/config.json")
        lo = js(root / "results/lora/final_config.json")
    except FileNotFoundError as e:
        print("skip table_config:", e)
        return
    g = lambda d, k, default="-": d.get(k, default)
    rows = [
        ["base / init", "Qwen2.5-0.5B", "SFT best", "SFT best", "SmolLM2-360M-Instruct"],
        ["trainable parameters", "full fine-tuning", "full fine-tuning", "full fine-tuning",
         f"LoRA r={lo['lora_r']}, alpha={lo['lora_alpha']}, dropout={lo['lora_dropout']}, all linear ({lo['trainable_params']:,} params)"],
        ["optimizer", f"AdamW b=({g(s,'adam_beta1')},{g(s,'adam_beta2')}), wd={g(s,'weight_decay')}",
         f"AdamW b=({g(h,'adam_beta1')},{g(h,'adam_beta2')}), wd={g(h,'weight_decay')}",
         f"AdamW b=({g(u,'adam_beta1')},{g(u,'adam_beta2')}), wd={g(u,'weight_decay')}", "AdamW (HF Trainer defaults), wd=0"],
        ["peak LR / schedule", f"{g(s,'lr')} / cosine, {g(s,'warmup_steps')} warmup steps", f"{g(h,'lr')} / fixed", f"{g(u,'lr')} / fixed",
         f"{lo['lr']} / {lo['lr_scheduler']}, {lo['warmup_steps']} warmup steps"],
        ["grad clip", g(s, "max_grad_norm"), g(h, "max_grad_norm"), g(u, "max_grad_norm"), lo["max_grad_norm"]],
        ["batch (device x accum = effective)", f"{g(s,'batch_size')} x {g(s,'grad_accum')} = {g(s,'effective_batch_size')}",
         f"1 pair x {g(h,'grad_accum')} = {g(h,'effective_batch_size')}", f"1 pair x {g(u,'grad_accum')} = {g(u,'effective_batch_size')}",
         f"{lo['per_device_batch_size']} x {lo['grad_accum']} = {lo['effective_batch_size']}"],
        ["precision", "fp32 weights, fp16 autocast", "policy fp32 + fp16 autocast; reference fp32 (autocast)",
         "policy fp32 + fp16 autocast; reference fp32 (autocast)", "fp32 weights, fp16 autocast"],
        ["max length / epochs", f"{g(s,'max_length')} / {g(s,'epochs')}", f"{g(h,'max_length')} / {g(h,'epochs')}",
         f"{g(u,'max_length')} / {g(u,'epochs')}", f"{lo['max_length']} / {lo['epochs']}"],
        ["optimizer steps", g(s, "optimizer_steps"), g(h, "optimizer_steps"), g(u, "optimizer_steps"), "see runs/*_metrics.json"],
        ["train / val examples", f"{g(s,'n_train')} / {g(s,'n_val')}", f"{g(h,'n_train')} / {g(h,'n_val')}",
         f"{g(u,'n_train')} / {g(u,'n_val')}", f"{lo['n_train']} / {lo['n_val']}"],
        ["DPO beta / reference", "-", f"{g(h,'beta')} / frozen SFT best", f"{g(u,'beta')} / frozen SFT best",
         f"{lo['beta']} / base model, adapter disabled"],
        ["seed", g(s, "seed"), g(h, "seed"), g(u, "seed"), lo["seed"]],
    ]
    (out / "table_config.md").write_text(md_table(header, rows))


# --------------------------------------------------------------------------- generation statistics
def looping(text, k=3, n_gram=5):
    t = re.sub(r"(?m)^\s*(\d+[.)]|[-*\u2022])\s+", "", text.lower())
    w = t.split()
    c = Counter(tuple(w[i:i + n_gram]) for i in range(len(w) - n_gram + 1))
    return bool(c) and max(c.values()) >= k


def table_generation(root, out, key):
    files = [("Qwen2.5-0.5B base", "AlpacaEval-300", "baseline_qwen2.5-0.5b-base_alpaca300.json"),
             ("SFT", "AlpacaEval-300", "sft_qwen2.5-0.5b_alpaca300.json"),
             ("DPO-UF", "AlpacaEval-300", "dpo_uf_qwen2.5-0.5b_alpaca300.json"),
             ("DPO-HH", "AlpacaEval-300", "dpo_hh_qwen2.5-0.5b_alpaca300.json"),
             ("SFT", "HH-eval-300", "sft_qwen2.5-0.5b_hh300.json"),
             ("DPO-HH", "HH-eval-300", "dpo_hh_qwen2.5-0.5b_hh300.json"),
             ("DPO-UF", "HH-eval-300", "dpo_uf_qwen2.5-0.5b_hh300.json"),
             ("SmolLM2 base", "persona-40", "persona_base.json"),
             ("LoRA r16 lr5e-5", "persona-40", "persona_r16_lr5e-5.json")]
    rows = []
    for model, evalset, fn in files:
        p = root / "results/generations" / fn
        if not p.exists():
            continue
        res = js(p)
        toks = [r["num_new_tokens"] for r in res]
        cap = 512                                   # every final generation used max_new_tokens=512
        row = {"EOS": sum(r["hit_eos"] for r in res), "cap": sum(t >= cap for t in toks), "mean": round(statistics.mean(toks)),
               "median": round(statistics.median(toks)), "loops": sum(looping(r["output"]) for r in res), "n": len(res)}
        rows.append([model, evalset, row["n"], row["EOS"], row["cap"], row["mean"], row["median"], row["loops"]])
        key.setdefault("generation", {})[f"{model} | {evalset}"] = row
    if rows:
        (out / "table_generation.md").write_text(md_table(
            ["model", "eval set", "n", "ended with EOS", "hit token cap", "mean tokens", "median tokens", "repeated-phrase outputs"], rows))


# --------------------------------------------------------------------------- judge tables and win-rate figure
PRIMARY = [("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT, AlpacaEval (general)"),
           ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT, HH eval (safety)"),
           ("dpo_hh_vs_sft_alpaca300", "DPO-HH vs SFT, AlpacaEval (alignment tax)"),
           ("dpo_uf_vs_sft_hh300", "DPO-UF vs SFT, HH eval (alignment tax)"),
           ("lora_vs_base_persona40_style", "LoRA vs base, persona style"),
           ("lora_vs_base_persona40_useful", "LoRA vs base, usefulness"),
           ("lora_vs_base_persona40_safety", "LoRA vs base, safety"),
           ("dpo_uf_vs_sft_alpaca300_swapped", "DPO-UF vs SFT, AlpacaEval (positions swapped)"),
           ("dpo_hh_vs_sft_hh300_swapped", "DPO-HH vs SFT, HH eval (positions swapped)"),
           ("dpo_uf_vs_sft_alpaca300_plainrubric", "DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance)"),
           ("dpo_hh_vs_sft_hh300_plainrubric", "DPO-HH vs SFT, HH eval (plain safety rubric)"),
           ("control_lr1e-4_vs_base_persona40_style", "CONTROL: over-trained LoRA vs base, style")]


def table_judge(root, out, key):
    p = root / "results/judgments/results_overview.csv"
    if not p.exists():
        print("skip judge tables: results/judgments/results_overview.csv missing")
        return
    rows = {r["file"]: r for r in csv.DictReader(open(p))}
    fmt = lambda x, d=3: "-" if x in ("", None) else f"{float(x):.{d}f}"
    table, bars = [], []
    for fn, label in PRIMARY:
        r = rows.get(fn)
        if not r:
            continue
        ci = f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] not in ("", "None") else "-"
        pv = "<0.001" if r["sign_test_p"] not in ("", "None") and float(r["sign_test_p"]) < 0.001 else fmt(r["sign_test_p"])
        table.append([label, r["n"], r["wins"], r["ties"], r["losses"], r["unknown"], fmt(r["win_rate_excl_ties"]), ci, pv])
        if r["win_rate_excl_ties"] not in ("", "None") and "swapped" not in fn and "control" not in fn and "plainrubric" not in fn:
            bars.append((label, float(r["win_rate_excl_ties"]), float(r["ci_low"]), float(r["ci_high"])))
        key.setdefault("judge", {})[fn] = {k: r[k] for k in ("wins", "ties", "losses", "unknown", "win_rate_excl_ties", "sign_test_p")}
    (out / "table_judge.md").write_text(md_table(
        ["comparison (model under test vs reference)", "n", "win", "tie", "loss", "unk", "win rate (no ties)", "95% CI", "sign-test p"], table))
    if bars:
        fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(bars) + 1.2))
        ys = list(range(len(bars)))[::-1]
        for y, (label, w, lo, hi) in zip(ys, bars):
            ax.errorbar(w, y, xerr=[[w - lo], [hi - w]], fmt="o", capsize=3)
        ax.axvline(0.5, color="grey", ls="--"); ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
        ax.set(xlim=(0, 1), xlabel="win rate of the model under test (ties excluded), 95% Wilson interval")
        fig.tight_layout(); fig.savefig(out / "fig_winrates.png", dpi=170); plt.close(fig)


def table_judge_strata(root, out):
    """Win rate by response-length relation for the main comparisons (from the analysis JSONs)."""
    rows = []
    for fn, label in (("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT (AlpacaEval)"), ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT (HH eval)"),
                      ("dpo_hh_vs_sft_alpaca300", "DPO-HH vs SFT (AlpacaEval)"), ("dpo_uf_vs_sft_hh300", "DPO-UF vs SFT (HH eval)"),
                      ("dpo_uf_vs_sft_alpaca300_plainrubric", "DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance)"),
                      ("dpo_hh_vs_sft_hh300_plainrubric", "DPO-HH vs SFT, HH eval (plain safety rubric)")):
        p = root / f"results/judgments/analysis/{fn}_judged.analysis.json"
        if not p.exists():
            continue
        a = js(p)
        for grp, s in a["by_length"].items():
            wr = "-" if s["win_rate_excl_ties"] is None else f"{s['win_rate_excl_ties']:.3f}"
            rows.append([label, grp, s["n"], s["wins"], s["ties"], s["losses"], wr, s["sign_test_p"] if s["sign_test_p"] is not None else "-"])
    if rows:
        (out / "table_judge_strata.md").write_text(md_table(
            ["comparison", "length of model under test vs reference", "n", "win", "tie", "loss", "win rate (no ties)", "p"], rows))


# --------------------------------------------------------------------------- LoRA sweep
def table_lora(root, out):
    sw, st = root / "results/lora/sweep_table.csv", root / "results/lora/style_stats.csv"
    if not (sw.exists() and st.exists()):
        print("skip LoRA sweep: sweep_table.csv / style_stats.csv missing")
        return
    sweep = {r["run"]: r for r in csv.DictReader(open(sw))}
    stats = {r["generator"]: r for r in csv.DictReader(open(st)) if r["category"] == "ALL"}
    runs = sorted(sweep, key=lambda k: float(sweep[k]["val_margin"]))
    rows, xs = [], []
    for run in runs:
        s, g = sweep[run], stats.get(run)
        rows.append([run, s["rank"], s["lr"], s["beta"], s["val_acc"], s["val_margin"], s["val_loss"],
                     g["hit_eos"] if g else "-", g["mean_tokens"] if g else "-", g["repeated_phrase_outputs"] if g else "-",
                     g["marker_rate"] if g else "-"])
        if g:
            xs.append((run, float(s["val_margin"]), int(g["hit_eos"]), int(g["repeated_phrase_outputs"])))
    base = stats.get("base")
    if base:
        rows.insert(0, ["base (no adapter)", "-", "-", "-", "-", "-", "-", base["hit_eos"], base["mean_tokens"],
                        base["repeated_phrase_outputs"], base["marker_rate"]])
    (out / "table_lora_sweep.md").write_text(md_table(
        ["run", "rank", "lr", "beta", "val acc", "val margin", "val loss", "EOS of 40", "mean tokens", "repeated-phrase outputs", "marker rate"], rows))
    if xs:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for run, m, eos, rep in xs:
            ax[0].scatter(m, eos); ax[0].annotate(run, (m, eos), fontsize=7, rotation=20)
            ax[1].scatter(m, rep); ax[1].annotate(run, (m, rep), fontsize=7, rotation=20)
        ax[0].set(xscale="log", xlabel="validation reward margin (log)", ylabel="outputs ending with EOS (of 40)", title="stopping behaviour")
        ax[1].set(xscale="log", xlabel="validation reward margin (log)", ylabel="outputs with repeated phrases (of 40)", title="degeneration")
        fig.tight_layout(); fig.savefig(out / "fig_lora_sweep.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="report/assets")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    fig_training(root, out, key)
    table_config(root, out)
    table_generation(root, out, key)
    table_judge(root, out, key)
    table_judge_strata(root, out)
    table_lora(root, out)
    (out / "key_numbers.json").write_text(json.dumps(key, indent=2))
    print("wrote:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()