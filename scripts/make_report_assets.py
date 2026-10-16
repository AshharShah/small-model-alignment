import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import report_extras
import summarize_judgments as sj
from alignment.persona import build_lexicon, degeneration_reasons, looping, split_persona_rows, style_metrics
from report_extras import jl, js, md_table, run_label


def short(run):
    return run.replace("r16_", "")


LEGACY_STYLE_FILES = {"r16_lr1e-4": "control_lr1e-4_vs_base_persona40_style"}


def persona_context(root):
    p = root / "results/lora/v2/persona_v2_selection.json"
    if not p.exists():
        return None
    sel = js(p)
    strongest = max(sel["candidates"], key=lambda c: (c["dev_style_score"], -c["steps"]))["name"]
    return {"selected": sel["selected"], "strongest": strongest, "candidates": sel["candidates"],
            "significant": sel["selected_style_shift_significant"]}


def judged_row(root, fn):
    p = root / f"results/judgments/{fn}_judged.jsonl"
    if not p.exists():
        return None
    s = sj.stats(sj.load_last(p))
    lo, hi = s["wilson95_excl_ties"]
    blank = lambda x: "" if x is None else x
    return {"n": s["n"], "wins": s["wins"], "ties": s["ties"], "losses": s["losses"], "unknown": s["unknown"],
            "win_rate_excl_ties": blank(s["win_rate_excl_ties"]), "win_rate_ties_half": blank(s["win_rate_ties_half"]),
            "ci_low": blank(lo), "ci_high": blank(hi), "sign_test_p": blank(s["sign_test_p"])}


def decoding_text(root, generation_file, template):
    p = root / "results/generations" / generation_file.replace(".json", ".meta.json")
    if not p.exists():
        return "-"
    d = js(p)["decoding"]
    return (f"greedy (do_sample={d['do_sample']}; temperature and top-p unused), max_new_tokens={d['max_new_tokens']}, "
            f"repetition_penalty={d['repetition_penalty']}, length_penalty=1.0, no_repeat_ngram_size=0; {template}")


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def fig_training(root, out, key):
    sft = root / "results/sft/metrics.jsonl"
    dpo = {n: root / f"results/dpo_{n}/metrics.jsonl" for n in ("hh", "uf")}
    if not sft.exists():
        print("skip fig_training: results/sft/metrics.jsonl missing")
        return
    colors = {"sft": "tab:green", "hh": "tab:blue", "uf": "tab:orange"}
    rc = {"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
          "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.9))
        recs = jl(sft)
        tr = [r for r in recs if r["type"] == "train"]
        ev = [r for r in recs if r["type"] == "eval"]
        best_sft = js(root / "results/sft/summary.json")
        a = ax[0, 0]
        a.plot([r["step"] for r in tr], smooth([r["train_loss"] for r in tr]), color=colors["sft"], alpha=0.35, lw=1.4)
        a.plot([r["step"] for r in ev if r["step"] > 0], [r["val_loss"] for r in ev if r["step"] > 0], "o-", color=colors["sft"])
        a.plot([ev[0]["step"]], [ev[0]["val_loss"]], "o", mfc="white", color=colors["sft"])
        kept = next(r for r in ev if r["step"] == best_sft["best_step"])
        a.plot([kept["step"]], [kept["val_loss"]], "*", ms=11, color=colors["sft"], mec="black", zorder=5)
        a.set(title="(a) SFT loss", xlabel="optimizer step", ylabel="response-token NLL")
        key["sft"] = {"val_loss_step0": ev[0]["val_loss"], "val_loss_final": ev[-1]["val_loss"],
                      "val_loss_min": min(r["val_loss"] for r in ev), "steps": tr[-1]["step"], "train_minutes": best_sft["train_minutes"]}
        panels = (("loss", "val_loss", "train_loss", "(b) DPO loss", "DPO loss"),
                  ("acc", "val_reward_acc", "train_reward_acc", "(c) DPO reward accuracy", "reward accuracy"),
                  ("margin", "val_reward_margin", "train_reward_margin", "(d) DPO reward margin", "reward margin"))
        summaries = {n: js(root / f"results/dpo_{n}/summary.json") for n, p in dpo.items() if p.exists()}
        for a, (kind, vkey, tkey, title, ylabel) in zip((ax[0, 1], ax[1, 0], ax[1, 1]), panels):
            for name, path in dpo.items():
                if not path.exists():
                    continue
                rows = jl(path)
                t = [r for r in rows if r["type"] == "train"]
                e = [r for r in rows if r["type"] == "eval"]
                a.plot([r["step"] for r in t], smooth([r[tkey] for r in t]), color=colors[name], alpha=0.35, lw=1.4)
                a.plot([r["step"] for r in e if r["step"] > 0], [r[vkey] for r in e if r["step"] > 0], "o-", color=colors[name])
                first = e[0]
                a.plot([first["step"]], [first[vkey]], "o", mfc="white", color=colors[name])
                best = next(r for r in e if r["step"] == summaries[name]["best_step"])
                a.plot([best["step"]], [best[vkey]], "*", ms=11, color=colors[name], mec="black", zorder=5)
            a.set(title=title, xlabel="optimizer step", ylabel=ylabel)
            if kind == "acc":
                a.annotate("step 0: policy = reference,\nthe strict comparison fires on float noise", (0, 0.12), fontsize=6.5,
                           xytext=(22, 18), textcoords="offset points", arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))
        for a in ax.flat:
            a.grid(alpha=0.2, lw=0.5)
        handles = [plt.Line2D([], [], color=colors[n], lw=2, label=lab) for n, lab in (("sft", "SFT"), ("hh", "HH-RLHF DPO"), ("uf", "UltraFeedback DPO"))]
        handles += [plt.Line2D([], [], color="grey", alpha=0.5, lw=1.6, label="training (5-point moving average)"),
                    plt.Line2D([], [], color="grey", marker="o", lw=1, label="validation"),
                    plt.Line2D([], [], color="grey", marker="o", mfc="white", ls="", label="step 0 (before training)"),
                    plt.Line2D([], [], color="grey", marker="*", ls="", ms=9, mec="black", label="checkpoint kept")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, columnspacing=1.4, handletextpad=0.5)
        fig.tight_layout(rect=(0, 0.095, 1, 1), h_pad=0.8, w_pad=1.0)
        fig.savefig(out / "fig_training.png", dpi=300)
        plt.close(fig)
    for name, path in dpo.items():
        if path.exists():
            e = [r for r in jl(path) if r["type"] == "eval"]
            key[f"dpo_{name}"] = {"val_acc_step0": e[0]["val_reward_acc"], "val_acc_final": e[-1]["val_reward_acc"],
                                  "val_acc_best": max(r["val_reward_acc"] for r in e), "best_step": summaries[name]["best_step"],
                                  "val_margin_final": e[-1]["val_reward_margin"], "val_loss_final": e[-1]["val_loss"],
                                  "max_abs_reward_step0": e[0].get("val_max_abs_reward"), "train_minutes": summaries[name]["train_minutes"]}


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
        ["optimizer steps", g(s, "optimizer_steps"), g(h, "optimizer_steps"), g(u, "optimizer_steps"), g(lo, "optimizer_steps")],
        ["train / val examples", f"{g(s,'n_train')} / {g(s,'n_val')}", f"{g(h,'n_train')} / {g(h,'n_val')}",
         f"{g(u,'n_train')} / {g(u,'n_val')}", f"{lo['n_train']} / {lo['n_val']}"],
        ["DPO beta / reference", "-", f"{g(h,'beta')} / frozen SFT best", f"{g(u,'beta')} / frozen SFT best",
         f"{lo['beta']} / base model, adapter disabled"
         + (f"; plus NLL term on the chosen answer, rpo_alpha={lo['rpo_alpha']}" if lo.get("rpo_alpha") else "")],
        ["seed", g(s, "seed"), g(h, "seed"), g(u, "seed"), lo["seed"]],
        ["generation settings (evaluation)",
         decoding_text(root, "sft_qwen2.5-0.5b_alpaca300.json", "prompt template '### Instruction / ### Response', no chat template"),
         decoding_text(root, "dpo_hh_qwen2.5-0.5b_hh300.json", "prompt template '### Instruction / ### Response', no chat template"),
         decoding_text(root, "dpo_uf_qwen2.5-0.5b_alpaca300.json", "prompt template '### Instruction / ### Response', no chat template"),
         decoding_text(root, f"persona_{lo['run']}.json", "SmolLM2 chat template")],
    ]
    (out / "table_config.md").write_text(md_table(header, rows))


def table_generation(root, out, key, ctx):
    files = [("Qwen2.5-0.5B base", "AlpacaEval-300", "baseline_qwen2.5-0.5b-base_alpaca300.json"),
             ("SFT", "AlpacaEval-300", "sft_qwen2.5-0.5b_alpaca300.json"),
             ("DPO-UF", "AlpacaEval-300", "dpo_uf_qwen2.5-0.5b_alpaca300.json"),
             ("DPO-HH", "AlpacaEval-300", "dpo_hh_qwen2.5-0.5b_alpaca300.json"),
             ("SFT", "HH-eval-300", "sft_qwen2.5-0.5b_hh300.json"),
             ("DPO-HH", "HH-eval-300", "dpo_hh_qwen2.5-0.5b_hh300.json"),
             ("DPO-UF", "HH-eval-300", "dpo_uf_qwen2.5-0.5b_hh300.json"),
             ("SmolLM2 base", "persona-40", "persona_base.json")]
    if ctx:
        picks = [("selected", ctx["selected"])] + ([("strongest", ctx["strongest"])] if ctx["strongest"] != ctx["selected"] else [])
        files += [(f"LoRA {short(run)} ({tag})", "persona-40", f"persona_{run}.json") for tag, run in picks]
        files += [("SmolLM2 base", "held-out-22", "persona_heldout_base.json")]
        files += [(f"LoRA {short(run)} ({tag})", "held-out-22", f"persona_heldout_{run}.json") for tag, run in picks]
    rows = []
    for model, evalset, fn in files:
        p = root / "results/generations" / fn
        if not p.exists():
            continue
        res = js(p)
        toks = [r["num_new_tokens"] for r in res]
        cap = 512
        row = {"EOS": sum(r["hit_eos"] for r in res), "cap": sum(t >= cap for t in toks), "mean": round(statistics.mean(toks)),
               "median": round(statistics.median(toks)), "loops": sum(looping(r["output"]) for r in res), "n": len(res)}
        rows.append([model, evalset, row["n"], row["EOS"], row["cap"], row["mean"], row["median"], row["loops"]])
        key.setdefault("generation", {})[f"{model} | {evalset}"] = row
    if rows:
        (out / "table_generation.md").write_text(md_table(
            ["model", "eval set", "n", "ended with EOS", "hit token cap", "mean tokens", "median tokens", "repeated-phrase outputs"], rows))


DPO_ROWS = [("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT, AlpacaEval (general)"),
            ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT, HH eval (safety)"),
            ("dpo_hh_vs_sft_alpaca300", "DPO-HH vs SFT, AlpacaEval (alignment tax)"),
            ("dpo_uf_vs_sft_hh300", "DPO-UF vs SFT, HH eval (alignment tax)")]

ROBUSTNESS_ROWS = [("dpo_uf_vs_sft_alpaca300_swapped", "DPO-UF vs SFT, AlpacaEval (positions swapped)"),
                   ("dpo_hh_vs_sft_hh300_swapped", "DPO-HH vs SFT, HH eval (positions swapped)"),
                   ("dpo_uf_vs_sft_alpaca300_plainrubric", "DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance)"),
                   ("dpo_hh_vs_sft_hh300_plainrubric", "DPO-HH vs SFT, HH eval (plain safety rubric)")]


def persona_rows(ctx):
    if not ctx:
        return []
    picks = [("selected", ctx["selected"])] + ([("strongest", ctx["strongest"])] if ctx["strongest"] != ctx["selected"] else [])
    rows = []
    for tag, run in picks:
        for kind, label in (("style", "persona style"), ("useful", "usefulness"), ("safety", "safety")):
            rows.append((f"persona40_{run}_vs_base_{kind}", f"LoRA {short(run)} ({tag}) vs base, {label}, 40 required prompts"))
    for tag, run in picks:
        rows.append((f"persona22_{run}_vs_base_style", f"LoRA {short(run)} ({tag}) vs base, persona style, 22 held-out prompts"))
    rows.append(("persona22_control_chosen_vs_rejected_style",
                 "POSITIVE CONTROL: dataset persona vs neutral answers, persona style, 22 held-out prompts"))
    return rows


def judge_note(root):
    p = root / "results/judgments/dpo_uf_vs_sft_alpaca300_judged.jsonl.summary.json"
    if not p.exists():
        return ""
    m = js(p)
    return (f"\nJudge: {m['judge_model_used']} through the Kaggle model proxy, {m['batch_size']} pairs per request, {m['sleep_seconds']:g} s between attempts, "
            f"up to {m['max_retries']} attempts per batch, then the pairs of a failed batch one at a time. A verdict that stayed invalid after that is recorded "
            "as unknown, excluded from both win rates and logged in a failure file (see the reliability table). Win rate (ties excluded) = wins / (wins + losses); "
            "win rate (ties = 1/2) = (wins + ties / 2) / (wins + ties + losses). Pairs with identical outputs count as ties without being sent to the judge.\n")


def table_judge(root, out, key, ctx):
    p = root / "results/judgments/results_overview.csv"
    if not p.exists():
        print("skip judge tables: results/judgments/results_overview.csv missing")
        return
    rows = {r["file"]: r for r in csv.DictReader(open(p))}
    fmt = lambda x, d=3: "-" if x in ("", None) else f"{float(x):.{d}f}"
    table, bars = [], []
    for fn, label in DPO_ROWS + persona_rows(ctx) + ROBUSTNESS_ROWS:
        r = rows.get(fn) or judged_row(root, fn)
        if not r:
            continue
        ci = f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] not in ("", "None") else "-"
        pv = "<0.001" if r["sign_test_p"] not in ("", "None") and float(r["sign_test_p"]) < 0.001 else fmt(r["sign_test_p"])
        table.append([label, r["n"], r["wins"], r["ties"], r["losses"], r["unknown"], fmt(r["win_rate_excl_ties"]),
                      fmt(r.get("win_rate_ties_half")), ci, pv])
        if (r["win_rate_excl_ties"] not in ("", "None") and "swapped" not in fn and "control" not in fn
                and "plainrubric" not in fn and "persona22" not in fn):
            bars.append((label, float(r["win_rate_excl_ties"]), float(r["ci_low"]), float(r["ci_high"])))
        key.setdefault("judge", {})[fn] = {k: r[k] for k in ("wins", "ties", "losses", "unknown", "win_rate_excl_ties", "sign_test_p")}
    text = md_table(["comparison (model under test vs reference)", "n", "win", "tie", "loss", "unk", "win rate (ties excluded)",
                     "win rate (ties = 1/2)", "95% CI (ties excluded)", "sign-test p"], table)
    text += judge_note(root)
    (out / "table_judge.md").write_text(text)
    if bars:
        bars = [(l.replace(" vs base,", ":").replace(", 40 required prompts", " (n=40)"), w, lo, hi) for l, w, lo, hi in bars]
        fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(bars) + 1.2))
        ys = list(range(len(bars)))[::-1]
        for y, (label, w, lo, hi) in zip(ys, bars):
            ax.errorbar(w, y, xerr=[[w - lo], [hi - w]], fmt="o", capsize=3)
        ax.axvline(0.5, color="grey", ls="--"); ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
        ax.set(xlim=(0, 1), xlabel="win rate of the model under test, ties excluded (95% Wilson interval)")
        fig.tight_layout(); fig.savefig(out / "fig_winrates.png", dpi=170); plt.close(fig)


def table_judge_strata(root, out):
    rows = []
    for fn, label in (("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT (AlpacaEval)"), ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT (HH eval)"),
                      ("dpo_hh_vs_sft_alpaca300", "DPO-HH vs SFT (AlpacaEval)"), ("dpo_uf_vs_sft_hh300", "DPO-UF vs SFT (HH eval)"),
                      ("dpo_uf_vs_sft_alpaca300_plainrubric", "DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance)"),
                      ("dpo_hh_vs_sft_hh300_plainrubric", "DPO-HH vs SFT, HH eval (plain safety rubric)"),
                      ("dpo_uf_vs_sft_alpaca300_rep1p1", "DPO-UF vs SFT, AlpacaEval (repetition penalty 1.1)"),
                      ("dpo_hh_vs_sft_hh300_rep1p1", "DPO-HH vs SFT, HH eval (repetition penalty 1.1)")):
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


def table_lora(root, out, key, ctx):
    sweep_csv = root / "results/lora/v2/sweep_table_v2.csv"
    gens = root / "results/generations"
    if not (ctx and sweep_csv.exists() and (gens / "persona_heldout_base.json").exists()):
        print("skip LoRA sweep: v2 sweep table, selection file or held-out generations missing")
        return
    sweep = sorted(csv.DictReader(open(sweep_csv)), key=lambda r: (float(r["rpo_alpha"]), float(r["epochs"]), float(r["lr"]), int(r["rank"])))
    base40, base22 = js(gens / "persona_base.json"), js(gens / "persona_heldout_base.json")
    train_rows, _ = split_persona_rows(js(root / "data/persona_lora/ramsay_style_preferences.json"))
    lexicon = build_lexicon(train_rows, [r["output"] for r in base40])
    stats_of = lambda fn: style_metrics(js(gens / fn), lexicon) if (gens / fn).exists() else None
    base_stats = {"eval40": style_metrics(base40, lexicon), "heldout22": style_metrics(base22, lexicon)}
    eos = lambda s: f"{s['hit_eos']}/{s['n']}" if s else "-"
    loops = lambda s: s["repeated_phrase_outputs"] if s else "-"

    def verdict(fn):
        r = judged_row(root, fn)
        if not r:
            return "-", None
        return f"{r['wins']}/{r['ties']}/{r['losses']}", (r["wins"] + 0.5 * r["ties"]) / (r["wins"] + r["ties"] + r["losses"])

    rows = [["base (no adapter)", "-", "-", "-", "-", "-", eos(base_stats["heldout22"]), loops(base_stats["heldout22"]), "-",
             eos(base_stats["eval40"]), loops(base_stats["eval40"]), "-", "-"]]
    points, summary, tex_runs = [], {}, []
    for r in sweep:
        run = r["run"]
        s40, s22 = stats_of(f"persona_{run}.json"), stats_of(f"persona_heldout_{run}.json")
        wtl22, score22 = verdict(f"persona22_{run}_vs_base_style")
        wtl40, score40 = verdict(f"persona40_{run}_vs_base_style")
        if score40 is None and run in LEGACY_STYLE_FILES:
            wtl40, score40 = verdict(LEGACY_STYLE_FILES[run])
        state = "-"
        if s40 and s22:
            state = "degenerate" if degeneration_reasons({"eval40": s40, "heldout22": s22}, base_stats) else "ok"
            if run == ctx["selected"]:
                state += " (selected)"
        elif s40 and degeneration_reasons({"eval40": s40, "heldout22": base_stats["heldout22"]}, base_stats):
            state = "degenerate (required only)"
        rows.append([run, r["rpo_alpha"], r["epochs"], r["lr"], r["optimizer_steps"], r["val_margin"],
                     eos(s22), loops(s22), wtl22, eos(s40), loops(s40), wtl40, state])
        summary[run] = {"heldout22": {"stats": s22, "style_wtl": wtl22}, "eval40": {"stats": s40, "style_wtl": wtl40}, "filter": state}
        tex_runs.append({"run": run, "rank": int(r["rank"]), "lr": float(r["lr"]), "epochs": float(r["epochs"]), "beta": float(r["beta"]),
                         "rpo_alpha": float(r["rpo_alpha"]), "margin": float(r["val_margin"]), "eos22": eos(s22), "wtl22": wtl22,
                         "eos40": eos(s40), "loops40": loops(s40), "loops22": loops(s22), "wtl40": wtl40, "state": state, "selected": run == ctx["selected"]})
        if s40 and s22 and score22 is not None and score40 is not None:
            name = run_label(r)
            points.append((run, score22, s22["hit_eos"] / s22["n"], score40, s40["hit_eos"] / s40["n"], name))
    (out / "table_lora_sweep.md").write_text(md_table(
        ["run", "rpo alpha", "epochs", "lr", "steps", "val margin", "held-out: EOS of 22", "held-out: repeated-phrase outputs",
         "held-out: style W/T/L", "required: EOS of 40", "required: repeated-phrase outputs", "required: style W/T/L", "degeneration filter"], rows))
    report_extras.tex_lora_sweep(out, {"eos40": eos(base_stats["eval40"]), "loops40": loops(base_stats["eval40"]),
                                          "eos22": eos(base_stats["heldout22"]), "loops22": loops(base_stats["heldout22"])}, tex_runs)
    report_extras.tex_deviations(root, out, tex_runs)
    key["persona"] = {"selected": ctx["selected"], "strongest": ctx["strongest"], "style_shift_significant": ctx["significant"],
                      "runs": summary}
    if points:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.2))
        colors = {pt[0]: plt.cm.tab10(i) for i, pt in enumerate(points)}
        for a, title, xi, yi, base in ((ax[0], "(a) held-out prompts (dev, n=22)", 1, 2, base_stats["heldout22"]),
                                       (ax[1], "(b) required prompts (test, n=40)", 3, 4, base_stats["eval40"])):
            base_rate = base["hit_eos"] / base["n"]
            a.axhspan(0, base_rate - 0.10, color="tab:red", alpha=0.10)
            a.axhline(base_rate, color="grey", ls="--", lw=1)
            a.text(0.455, base_rate + 0.012, "base model", fontsize=7, color="grey")
            a.text(0.455, 0.02, "filtered out as degenerate (shaded)", fontsize=7, color="tab:red")
            for pt in points:
                chosen = pt[0] == ctx["selected"]
                a.scatter(pt[xi], pt[yi], s=190 if chosen else 55, marker="*" if chosen else "o", color=colors[pt[0]],
                          edgecolor="black", linewidth=0.6, zorder=3)
            a.set(title=title, xlim=(0.45, 1.0), ylim=(0, 1.08), xlabel="persona-style score vs base model (wins + 0.5 ties) / pairs",
                  ylabel="answers that end on their own (fraction)")
        handles = [plt.Line2D([], [], marker="*" if pt[0] == ctx["selected"] else "o", ls="", color=colors[pt[0]], markeredgecolor="black",
                              markersize=13 if pt[0] == ctx["selected"] else 7) for pt in points]
        labels = [pt[5] + (" (selected)" if pt[0] == ctx["selected"] else "") for pt in points]
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(out / "fig_lora_sweep.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="report/assets")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    ctx = persona_context(root)
    fig_training(root, out, key)
    table_config(root, out)
    table_generation(root, out, key, ctx)
    table_judge(root, out, key, ctx)
    table_judge_strata(root, out)
    table_lora(root, out, key, ctx)
    report_extras.build_all(root, out, ctx, DPO_ROWS + persona_rows(ctx) + ROBUSTNESS_ROWS)
    (out / "key_numbers.json").write_text(json.dumps(key, indent=2))
    report_extras.write_index(out)
    print("wrote:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
