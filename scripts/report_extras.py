import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

import summarize_judgments as sj
from alignment.persona import looping


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def js(path):
    return json.loads(Path(path).read_text())


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


RAGGED = r">{\raggedright\arraybackslash}"

TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

SOURCE_NAMES = {"selfinstruct": "self-instruct", "koala": "Koala", "oasst": "OASST", "helpful_base": r"helpful\_base", "vicuna": "Vicuna"}

REFUSAL = re.compile(r"\b(i'?m sorry|i am sorry|i cannot|i can'?t|i won'?t|i'?m not able|i am not able|unable to|i'?m unable)\b", re.I)

MAIN = [("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT, AlpacaEval"),
        ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT, HH eval"),
        ("dpo_hh_vs_sft_alpaca300", "DPO-HH vs SFT, AlpacaEval"),
        ("dpo_uf_vs_sft_hh300", "DPO-UF vs SFT, HH eval")]

CONSISTENCY = [("dpo_uf_vs_sft_alpaca300", "dpo_uf_vs_sft_alpaca300_swapped", "DPO-UF vs SFT, AlpacaEval: original vs swapped positions"),
               ("dpo_hh_vs_sft_hh300", "dpo_hh_vs_sft_hh300_swapped", "DPO-HH vs SFT, HH eval: original vs swapped positions"),
               ("dpo_uf_vs_sft_alpaca300", "dpo_uf_vs_sft_alpaca300_plainrubric", "DPO-UF vs SFT, AlpacaEval: default vs plain rubric"),
               ("dpo_hh_vs_sft_hh300", "dpo_hh_vs_sft_hh300_plainrubric", "DPO-HH vs SFT, HH eval: default vs plain rubric")]

COLORS = {"base": "tab:gray", "SFT": "tab:green", "DPO-HH": "tab:blue", "DPO-UF": "tab:orange"}

DECODING_TAG = "rep1p1"

DECODING_PAIRS = [("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT, AlpacaEval", "AlpacaEval-300",
                   [("SFT", "sft_qwen2.5-0.5b_alpaca300"), ("DPO-UF", "dpo_uf_qwen2.5-0.5b_alpaca300")]),
                  ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT, HH eval", "HH-eval-300",
                   [("SFT", "sft_qwen2.5-0.5b_hh300"), ("DPO-HH", "dpo_hh_qwen2.5-0.5b_hh300")])]

BIAS_RUNS = [("dpo_uf_vs_sft_alpaca300", "DPO-UF vs SFT, AlpacaEval"), ("dpo_hh_vs_sft_hh300", "DPO-HH vs SFT, HH eval")]


def tex_escape(text):
    return "".join(TEX_ESCAPES.get(c, c) for c in str(text))


def tt(text):
    return r"\texttt{" + tex_escape(text).replace(r"\_", r"\_\allowbreak{}") + "}"


def latex_tabular(header, rows, spec, size=r"\small", tabcolsep=None):
    pad = rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}" if tabcolsep else ""
    lines = [rf"\begingroup{size}{pad}\renewcommand{{\arraystretch}}{{1.12}}", rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def load_judged(root, fn):
    p = root / f"results/judgments/{fn}_judged.jsonl"
    return sj.load_last(p) if p.exists() else None


def run_label(args):
    return f"lr {float(args['lr']):.0e}, {float(args['epochs']):g} ep, " + ("DPO+NLL" if float(args.get("rpo_alpha", 0)) > 0 else "DPO")


def rate_ci(s):
    if s["win_rate_excl_ties"] is None:
        return "-"
    lo, hi = s["wilson95_excl_ties"]
    return f"{s['win_rate_excl_ties']:.3f} [{lo:.3f}, {hi:.3f}]"


def wtl(s):
    return f"{s['wins']}/{s['ties']}/{s['losses']}"


def table_data(root, out):
    try:
        sft = js(root / "results/sft/sft_data_manifest.json")
        hh = js(root / "results/dpo_hh/dpo_hh_manifest.json")
        uf = js(root / "results/dpo_uf/dpo_uf_manifest.json")
        lo = js(root / "results/lora/final_config.json")
    except FileNotFoundError as e:
        print("skip table_data:", e)
        return
    n = lambda x: f"{x:,}"
    pref = lambda m, drop: (f"single-turn only; prompt <= {m['max_prompt_length']} and prompt + response + EOS <= {m['max_length']} tokens; "
                            f"{drop}; {n(m['train']['skipped_too_long'])} of {n(m['train']['seen'])} train rows dropped as too long")
    rows = [
        ["SFT", f"{sft['dataset']} ({sft['train']['source_split']} / {sft['val']['source_split']})",
         f"{n(sft['train']['n'])} / {n(sft['val']['n'])}",
         f"first user/assistant turn only; prompt + response + EOS <= {sft['max_length']} tokens "
         f"({n(sft['train']['skipped_too_long'])} of {n(sft['train']['seen'])} train rows dropped as too long)",
         f"{sft['train']['mean_prompt_tokens']} prompt / {sft['train']['mean_response_tokens']} response"],
        ["HH-RLHF DPO", f"{hh['dataset']} ({hh['train']['source_split']} / {hh['val']['source_split']})",
         f"{n(hh['train']['n'])} / {n(hh['val']['n'])}",
         pref(hh, f"{n(hh['train']['skipped_not_single_turn'])} multi-turn rows dropped"),
         f"{hh['train']['mean_prompt_tokens']} prompt / {hh['train']['mean_chosen_tokens']} chosen / {hh['train']['mean_rejected_tokens']} rejected; "
         f"chosen longer in {hh['train']['frac_chosen_longer']:.1%}"],
        ["UltraFeedback DPO", f"{uf['dataset']} ({uf['train']['source_split']} / {uf['val']['source_split']})",
         f"{n(uf['train']['n'])} / {n(uf['val']['n'])} ({uf['val']['from_train_holdout']} validation pairs from unused train rows)",
         pref(uf, f"{n(uf['train']['skipped_tie_or_reversed_score'])} score ties dropped"),
         f"{uf['train']['mean_prompt_tokens']} prompt / {uf['train']['mean_chosen_tokens']} chosen / {uf['train']['mean_rejected_tokens']} rejected; "
         f"chosen longer in {uf['train']['frac_chosen_longer']:.1%}"],
        ["LoRA persona DPO", "data/persona_lora/ramsay_style_preferences.json (provided)", f"{lo['n_train']} / {lo['n_val']}",
         f"none; seeded shuffle (seed {lo['seed']}), first {lo['n_val']} rows held out", "-"],
    ]
    (out / "table_data.md").write_text(md_table(["track", "source (train split / validation split)", "train / validation", "filtering", "mean tokens"], rows))


def sci(x, places=0):
    mantissa, exponent = f"{x:.{places}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def cell(*lines):
    return r"\newline ".join(lines)


def tex_config(root, out):
    try:
        sft, hh, uf = (js(root / f"results/{n}/config.json") for n in ("sft", "dpo_hh", "dpo_uf"))
        ssum, hsum, usum = (js(root / f"results/{n}/summary.json") for n in ("sft", "dpo_hh", "dpo_uf"))
        lo = js(root / "results/lora/final_config.json")
        meta = {k: js(root / f"results/generations/{f}.meta.json") for k, f in
                (("sft", "sft_qwen2.5-0.5b_alpaca300"), ("hh", "dpo_hh_qwen2.5-0.5b_hh300"), ("uf", "dpo_uf_qwen2.5-0.5b_alpaca300"),
                 ("lora", f"persona_{lo['run']}"))}
    except FileNotFoundError as e:
        print("skip tex_config:", e)
        return
    betas = lambda c: rf"AdamW, $\beta=({c['adam_beta1']:g},{c['adam_beta2']:g})$"
    gen = lambda m, template: cell("greedy decoding", f"{m['decoding']['max_new_tokens']} new tokens", f"repetition penalty {m['decoding']['repetition_penalty']:.1f}",
                                   "length penalty 1.0", template)
    plain, chat = "plain prompt template", "SmolLM2 chat template"
    targets = ", ".join(t.replace("_proj", "") for t in lo["target_modules"])
    full = cell("full fine-tuning", "(all weights)")

    def dpo_row(name, c, summ, m):
        return [name, full,
                cell(betas(c), rf"peak LR {sci(c['lr'])}", "constant, no warmup", f"weight decay {c['weight_decay']:g}, clip {c['max_grad_norm']:.1f}"),
                cell(rf"1 pair $\times$ {c['grad_accum']} accum.\ $=$ {c['effective_batch_size']}", "fp32 weights, fp16 autocast (policy and reference)",
                     f"max length {c['max_length']}, seed {c['seed']}", f"{c['epochs']} epoch, {c['optimizer_steps']} steps (kept: step~{summ['best_step']})"),
                cell(rf"$\beta={c['beta']:g}$", f"reference: frozen SFT best checkpoint (step~{ssum['best_step']})"),
                gen(m, plain)]

    rows = [
        ["SFT", full,
         cell(betas(sft), rf"peak LR {sci(sft['lr'])}", rf"cosine to 0, warmup {sft['warmup_steps']} steps ({sft['warmup_ratio']:.0%})".replace("%", r"\%"),
              f"weight decay {sft['weight_decay']:g}, clip {sft['max_grad_norm']:.1f}"),
         cell(rf"{sft['batch_size']} $\times$ {sft['grad_accum']} accum.\ $=$ {sft['effective_batch_size']}", "fp32 weights, fp16 autocast",
              f"max length {sft['max_length']}, seed {sft['seed']}", f"{sft['epochs']} epoch, {sft['optimizer_steps']} steps (kept: step~{ssum['best_step']})"),
         "n/a", gen(meta["sft"], plain)],
        dpo_row("HH-RLHF DPO", hh, hsum, meta["hh"]),
        dpo_row("UltraFeedback DPO", uf, usum, meta["uf"]),
        ["LoRA-DPO (TRL)",
         cell(rf"LoRA $r={lo['lora_r']}$, $\alpha={lo['lora_alpha']}$, dropout {lo['lora_dropout']:g}", f"targets: {targets}",
              f"{lo['trainable_params']:,} of {lo['total_params']:,} trainable ({lo['trainable_params'] / lo['total_params']:.2%})".replace("%", r"\%")),
         cell(r"AdamW, $\beta=(0.9,0.999)$ (Trainer default)", rf"peak LR {sci(lo['lr'])}", f"{lo['lr_scheduler']} to 0, warmup {lo['warmup_steps']} steps",
              f"weight decay 0, clip {lo['max_grad_norm']:.1f}"),
         cell(rf"{lo['per_device_batch_size']} $\times$ {lo['grad_accum']} accum.\ $=$ {lo['effective_batch_size']}", "fp32 weights, fp16 autocast",
              f"max length {lo['max_length']}, seed {lo['seed']}", f"{lo['epochs']:g} epochs, {lo['optimizer_steps']} steps"),
         cell(rf"$\beta={lo['beta']:g}$, plus NLL on the chosen answer ($\mathrm{{rpo}}_\alpha={lo['rpo_alpha']:g}$)", "reference: base SmolLM2, adapter disabled"),
         gen(meta["lora"], chat)],
    ]
    spec = f"{RAGGED}p{{1.8cm}}{RAGGED}p{{2.9cm}}{RAGGED}p{{2.9cm}}{RAGGED}p{{3.3cm}}{RAGGED}p{{2.4cm}}{RAGGED}X"
    header = ["Run", "Trainable parameters", "Optimizer and schedule", "Batch, precision, length, seed", r"DPO $\beta$ and reference", "Generation"]
    (out / "table_config.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def tex_data(root, out):
    try:
        sft = js(root / "results/sft/sft_data_manifest.json")
        hh = js(root / "results/dpo_hh/dpo_hh_manifest.json")
        uf = js(root / "results/dpo_uf/dpo_uf_manifest.json")
        lo = js(root / "results/lora/final_config.json")
    except FileNotFoundError as e:
        print("skip tex_data:", e)
        return
    n = lambda x: f"{x:,}"
    split = lambda m: f"({tt(m['train']['source_split'])} / {tt(m['val']['source_split'])})"
    limits = lambda m: rf"prompt $\le$ {m['max_prompt_length']} and prompt + response + EOS $\le$ {m['max_length']} tokens"
    rows = [
        ["SFT", rf"UltraChat-200k~\citep{{ding2023ultrachat}} {split(sft)}", f"{n(sft['train']['n'])} / {n(sft['val']['n'])}",
         rf"first user--assistant turn only; prompt + response + EOS $\le$ {sft['max_length']} tokens "
         f"(this limit drops {n(sft['train']['skipped_too_long'])} of {n(sft['train']['seen'])} train rows)"],
        ["HH-RLHF DPO", rf"HH-RLHF~\citep{{bai2022hh}} {split(hh)}", f"{n(hh['train']['n'])} / {n(hh['val']['n'])}",
         f"single-turn only (drops {n(hh['train']['skipped_not_single_turn'])} multi-turn rows); {limits(hh)} "
         f"(drops {n(hh['train']['skipped_too_long'])} more)"],
        ["UltraFeedback DPO", rf"UltraFeedback, binarized~\citep{{cui2023ultrafeedback,tunstall2023zephyr}} {split(uf)}",
         rf"{n(uf['train']['n'])} / {n(uf['val']['n'])}$^\dagger$",
         rf"single-turn only; chosen score $>$ rejected (drops {n(uf['train']['skipped_tie_or_reversed_score'])} ties); same length limits "
         rf"(drops {n(uf['train']['skipped_too_long'])} more)"],
        ["LoRA persona DPO", rf"provided {tt('ramsay_style_preferences.json')}", f"{lo['n_train']} / {lo['n_val']}",
         f"none; a seeded shuffle (seed {lo['seed']}) holds out {lo['n_val']} rows"],
    ]
    spec = f"{RAGGED}p{{2.15cm}}{RAGGED}p{{4.3cm}}p{{2.2cm}}{RAGGED}X"
    (out / "table_data.tex").write_text(latex_tabular(["Track", "Source (train / validation split)", "Train / val", "Filtering"], rows, spec))
    (out / "table_data_footnote.tex").write_text(f"{uf['val']['from_train_holdout']}")


def tex_eval_sets(root, out):
    es = root / "eval_sets"
    try:
        alp, hh = js(es / "alpaca_eval_300_seed42.json"), js(es / "hh_rlhf_eval_300_seed42.json")
        per, held = js(es / "persona_eval_prompts.json"), js(es / "persona_heldout_prompts.json")
    except FileNotFoundError as e:
        print("skip tex_eval_sets:", e)
        return
    src = Counter(r["source_dataset"] for r in alp)
    sources = ", ".join(f"{SOURCE_NAMES.get(k, tex_escape(k))} {v}" for k, v in src.most_common())
    hh_c, per_c = Counter(r["category"] for r in hh), Counter(r["category"] for r in per)
    even = lambda c: f"{len(c)} categories $\\times$ {next(iter(c.values()))} prompts" if len(set(c.values())) == 1 else ", ".join(f"{k} {v}" for k, v in c.items())
    rows = [
        ["AlpacaEval-300", str(len(alp)), rf"random subset (seed 42) of the 805 AlpacaEval prompts~\citep{{dubois2024lc}}: {sources}",
         r"SFT vs.\ DPO-UF (main), DPO-HH (alignment tax)"],
        ["HH-eval-300", str(len(hh)), f"synthetic helpful/harmless set from the assignment; {even(hh_c)}",
         r"SFT vs.\ DPO-HH (main), DPO-UF (alignment tax)"],
        ["Persona-40", str(len(per)), f"hand-written; {even(per_c)} (cooking, coding, general help, safety-sensitive); no overlap with the training prompts",
         r"LoRA adapters vs.\ base model (test)"],
        ["Held-out-22", str(len(held)), "validation rows of the persona data that no training run used", "LoRA adapter selection (dev)"],
    ]
    spec = f"{RAGGED}p{{2.3cm}}r{RAGGED}X{RAGGED}p{{4.1cm}}"
    (out / "table_eval_sets.tex").write_text(latex_tabular(["Evaluation set", "Prompts", "Composition", "Used for"], rows, spec))


def table_eval_sets(root, out):
    es = root / "eval_sets"
    try:
        alp, hh = js(es / "alpaca_eval_300_seed42.json"), js(es / "hh_rlhf_eval_300_seed42.json")
        per, held = js(es / "persona_eval_prompts.json"), js(es / "persona_heldout_prompts.json")
    except FileNotFoundError as e:
        print("skip table_eval_sets:", e)
        return
    even = lambda c: f"{len(c)} categories x {next(iter(c.values()))}" if len(set(c.values())) == 1 else ", ".join(f"{k} {v}" for k, v in c.items())
    src = Counter(r["source_dataset"] for r in alp)
    rows = [
        ["AlpacaEval-300", "eval_sets/alpaca_eval_300_seed42.json", len(alp),
         "fixed random subset (seed 42) of the 805 AlpacaEval prompts: " + ", ".join(f"{k} {v}" for k, v in src.most_common()),
         "SFT vs DPO-UF (main); SFT vs DPO-HH (alignment tax)"],
        ["HH-eval-300", "eval_sets/hh_rlhf_eval_300_seed42.json", len(hh), even(Counter(r["category"] for r in hh)),
         "SFT vs DPO-HH (main); SFT vs DPO-UF (alignment tax)"],
        ["persona-40", "eval_sets/persona_eval_prompts.json", len(per), even(Counter(r["category"] for r in per)),
         "LoRA adapters vs base model (test set)"],
        ["held-out-22", "eval_sets/persona_heldout_prompts.json", len(held), "validation rows of the persona data, never trained on",
         "LoRA adapter selection (dev set)"],
    ]
    (out / "table_eval_sets.md").write_text(md_table(["evaluation set", "file", "prompts", "composition", "used for"], rows))


GUIDE_SFT = [("max length", "max_length", 512), ("train examples", "n_train", 30000), ("validation examples", "n_val", 1000),
             ("per-device batch", "batch_size", 2), ("gradient accumulation", "grad_accum", 16), ("effective batch", "effective_batch_size", 32),
             ("learning rate", "lr", 2e-5), ("warmup", "warmup_ratio", 0.03), ("weight decay", "weight_decay", 0.1),
             ("gradient clipping", "max_grad_norm", 1.0), ("epochs", "epochs", 1)]
GUIDE_DPO = [("train pairs", "n_train", 15000), ("validation pairs", "n_val", 1000), ("max length", "max_length", 512),
             ("gradient accumulation (1 pair per device)", "grad_accum", 32), ("effective batch", "effective_batch_size", 32),
             ("learning rate (fixed)", "lr", 1e-6), ("beta", "beta", 0.1), ("gradient clipping", "max_grad_norm", 1.0),
             ("epochs", "epochs", 1), ("warmup steps (fixed LR)", "warmup_steps", 0)]


def table_deviations(root, out):
    try:
        s, h, u = (js(root / f"results/{p}/config.json") for p in ("sft", "dpo_hh", "dpo_uf"))
        ss, sh, su = (js(root / f"results/{p}/summary.json") for p in ("sft", "dpo_hh", "dpo_uf"))
        hm, um = js(root / "results/dpo_hh/dpo_hh_manifest.json"), js(root / "results/dpo_uf/dpo_uf_manifest.json")
        sm, lo = js(root / "results/sft/sft_data_manifest.json"), js(root / "results/lora/final_config.json")
    except FileNotFoundError as e:
        print("skip table_deviations:", e)
        return
    same, rows = {"SFT": [], "HH-RLHF DPO": [], "UltraFeedback DPO": []}, []
    for stage, cfg, guide in (("SFT", s, GUIDE_SFT), ("HH-RLHF DPO", h, GUIDE_DPO), ("UltraFeedback DPO", u, GUIDE_DPO)):
        for name, key, value in guide:
            if abs(float(cfg[key]) - float(value)) < 1e-12:
                same[stage].append(f"{name} {value:g}")
            else:
                rows.append([stage, name, f"{value:g}", f"{cfg[key]:g}", "", "differs"])
    for stage, m in (("HH-RLHF DPO", hm), ("UltraFeedback DPO", um)):
        (same[stage] if m["max_prompt_length"] == 256 else []).append("max prompt length 256")
    n = lambda x: f"{x:,}"
    rows += [
        ["SFT", "optimizer steps", "30,000 / 32 = 937.5 (implied)", f"{ss['optimizer_steps']} ({n(s['examples_used_per_epoch'])} examples)",
         "the incomplete last accumulation group is dropped", "implementation"],
        ["SFT", "training examples", "30,000 single-turn examples", "30,000 that fit in 512 tokens",
         f"{n(sm['train']['skipped_too_long'])} longer rows skipped so every kept example keeps its EOS label", "data"],
        ["SFT", "validation batch size", "not specified", f"{s['eval_batch_size']} (8 at first)",
         "validation ran out of memory on the T4; validation loss is an exact token mean, so the batch size changes no value", "resource"],
        ["SFT", "AdamW betas", "not specified", f"({s['adam_beta1']}, {s['adam_beta2']})", "not tuned", "design"],
        ["HH / UF DPO", "optimizer steps", "15,000 / 32 = 468.75 (implied)", f"{sh['optimizer_steps']}", "the incomplete last accumulation group is dropped",
         "implementation"],
        ["HH / UF DPO", "reference model precision", "may be loaded in fp16", "fp32 weights under fp16 autocast, on the second GPU",
         "the second T4 had room, and policy and reference then start numerically identical", "design"],
        ["HH / UF DPO", "optimizer", "RMSprop or AdamW", f"AdamW, betas ({h['adam_beta1']}, {h['adam_beta2']}), weight decay {h['weight_decay']:g}",
         "not tuned", "design"],
        ["HH / UF DPO", "checkpoint selection", "best validation reward accuracy",
         f"best validation reward accuracy (ties: lower loss): HH step {sh['best_step']}, UF step {su['best_step']}",
         "validated every 100 steps and at the end", "as guide"],
        ["UltraFeedback DPO", "validation pairs", "1,000",
         f"1,000 ({um['val']['from_train_holdout']} from train rows not used for training)",
         "the test split has too few usable single-turn pairs", "data"],
    ]
    plain = load_judged(root, "lora_vs_base_persona40_style")
    plain_text = f"{wtl(sj.stats(plain))} style W/T/L on the 40 required prompts" if plain else "no style shift"
    rows += [
        ["LoRA persona", "loss", "TRL DPOTrainer, plain DPO", f"DPO + NLL on the chosen answer (rpo_alpha {lo['rpo_alpha']:g})",
         f"plain DPO gave no persona ({plain_text}); see the LoRA sweep table", "performance"],
        ["LoRA persona", "learning rate / epochs / rank", "student choice",
         f"{lo['lr']:g} / {lo['epochs']:g} / r={lo['lora_r']}", "picked by a pre-stated rule from the 6 adapters scored on the held-out set (12 runs trained in total)", "performance"],
        ["Generation", "decoding", "greedy; repetition controls optional", "greedy, 512 new tokens, repetition penalty 1.0",
         "same settings for SFT and DPO; many outputs still loop, see the generation table", "as guide"],
    ]
    text = md_table(["stage", "setting", "guide", "used", "note", "kind"], rows)
    text += "\nUnchanged from the guide (checked against the saved configs):\n\n"
    for stage, items in same.items():
        text += f"- {stage}: {', '.join(items)}\n"
    (out / "table_deviations.md").write_text(text)


def band_row(text, columns):
    return rf"\rowcolor{{accent!8}}\multicolumn{{{columns}}}{{@{{}}l}}{{\textbf{{{text}}}}}"


def p_text(p):
    return "-" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def outcome_cells(s):
    return [str(s["n"]), str(s["wins"]), str(s["ties"]), str(s["losses"])], [rate_ci(s), f"{s['win_rate_ties_half']:.3f}", p_text(s["sign_test_p"])]


def extra_calls(root, fn):
    m = js(root / f"results/judgments/{fn}_judged.jsonl.summary.json")
    return m["judge_calls"] - m["batches_run"]


DPO_RESULT_BANDS = [
    ("Own-track comparison: submitted DPO checkpoint against its SFT baseline",
     [("HH-RLHF DPO vs SFT, HH-eval", "dpo_hh_vs_sft_hh300"), ("UltraFeedback DPO vs SFT, AlpacaEval", "dpo_uf_vs_sft_alpaca300")]),
    ("Alignment tax: the other track's prompts",
     [("HH-RLHF DPO vs SFT, AlpacaEval", "dpo_hh_vs_sft_alpaca300"), ("UltraFeedback DPO vs SFT, HH-eval", "dpo_uf_vs_sft_hh300")]),
    ("Sensitivity of the own-track comparisons (HH-RLHF DPO on HH-eval, UltraFeedback DPO on AlpacaEval): one setting changed",
     [("HH-eval, positions swapped", "dpo_hh_vs_sft_hh300_swapped"), ("HH-eval, plain rubric", "dpo_hh_vs_sft_hh300_plainrubric"),
      ("HH-eval, repetition penalty 1.1", "dpo_hh_vs_sft_hh300_rep1p1"),
      ("AlpacaEval, positions swapped", "dpo_uf_vs_sft_alpaca300_swapped"),
      ("AlpacaEval, plain rubric", "dpo_uf_vs_sft_alpaca300_plainrubric"),
      ("AlpacaEval, repetition penalty 1.1", "dpo_uf_vs_sft_alpaca300_rep1p1")]),
]


def tex_dpo_results(root, out):
    rows = []
    for band, items in DPO_RESULT_BANDS:
        group = []
        for label, fn in items:
            recs = load_judged(root, fn)
            if recs is None:
                continue
            s = sj.stats(recs)
            counts, rates = outcome_cells(s)
            group.append([label] + counts + [str(s["unknown"]), str(extra_calls(root, fn))] + rates)
        if group:
            rows += [[band_row(band, 10)]] + group
    if not rows:
        print("skip tex_dpo_results: judged files missing")
        return
    centered = r">{\centering\arraybackslash}"
    spec = r">{\raggedright\arraybackslash}X" + "".join(centered + f"p{{{w}}}" for w in ("0.7cm", "0.7cm", "0.7cm", "0.7cm", "0.8cm", "1.0cm", "3.0cm", "1.4cm", "1.2cm"))
    header = ["Comparison", "n", "W", "T", "L", "Unk.", "Extra calls", r"Win rate, ties excluded [95\% CI]", r"Win rate, ties $\tfrac{1}{2}$", r"Sign test $p$"]
    (out / "table_dpo_results.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def tex_lora_results(root, out, ctx):
    if not ctx:
        print("skip tex_lora_results: persona selection missing")
        return
    rows = []
    control = load_judged(root, "persona22_control_chosen_vs_rejected_style")
    if control is not None:
        counts, rates = outcome_cells(sj.stats(control))
        rows += [[band_row("Positive control", 8)], ["Dataset persona answers vs neutral answers: style, 22 held-out prompts"] + counts + rates]
    for tag, run in (("Selected adapter", ctx["selected"]), ("Strongest persona adapter", ctx["strongest"])):
        if tag == "Strongest persona adapter" and run == ctx["selected"]:
            continue
        args = js(root / f"results/lora/runs/{run}_metrics.json")["args"]
        title = f"{tag}: rank {args['lora_r']}, learning rate {sci(float(args['lr']))}, {float(args['epochs']):g} epochs" + (", NLL term" if float(args.get("rpo_alpha", 0)) > 0 else "")
        group = []
        for label, fn in ((f"Persona style, 40 required prompts", f"persona40_{run}_vs_base_style"),
                          (f"Persona style, 22 held-out prompts", f"persona22_{run}_vs_base_style"),
                          (f"Usefulness, 40 required prompts", f"persona40_{run}_vs_base_useful"),
                          (f"Safety, 40 required prompts", f"persona40_{run}_vs_base_safety")):
            recs = load_judged(root, fn)
            if recs is not None:
                counts, rates = outcome_cells(sj.stats(recs))
                group.append([label] + counts + rates)
        if group:
            rows += [[band_row(title + " (against the base model)", 8)]] + group
    if not rows:
        print("skip tex_lora_results: judged files missing")
        return
    centered = r">{\centering\arraybackslash}"
    spec = r">{\raggedright\arraybackslash}X" + "".join(centered + f"p{{{w}}}" for w in ("0.7cm", "0.7cm", "0.7cm", "0.7cm", "3.0cm", "1.4cm", "1.2cm"))
    header = ["Judged target", "n", "W", "T", "L", r"Win rate, ties excluded [95\% CI]", r"Win rate, ties $\tfrac{1}{2}$", r"Sign test $p$"]
    (out / "table_lora_results.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def peak_memory(root, run, key):
    return max(r[key] for r in jl(root / f"results/{run}/metrics.jsonl") if r["type"] == "train")


def step0_reward(root, run):
    return next(r for r in jl(root / f"results/{run}/metrics.jsonl") if r["type"] == "eval" and r["step"] == 0)["val_max_abs_reward"]


def tex_lora_sweep(out, base, runs):
    loss = lambda r: "DPO + NLL" if r["rpo_alpha"] > 0 else (rf"DPO, $\beta={r['beta']:g}$" if abs(r["beta"] - 0.1) > 1e-9 else "DPO")
    verdict = {"ok": "pass", "degenerate": "fail", "-": "-"}
    rows = [[r"\multicolumn{5}{@{}l}{Base model, no adapter}", base["eos40"], str(base["loops40"]), base["eos22"], str(base["loops22"]), "-", "-"]]
    for r in runs:
        state = r["state"].replace(" (selected)", "")
        cells = [str(r["rank"]), sci(r["lr"]), f"{r['epochs']:g}", loss(r), f"{r['margin']:.2f}", r["eos40"], str(r["loops40"]), r["eos22"], str(r["loops22"]),
                 r["wtl40"], verdict[state] + (" (selected)" if r["selected"] else "")]
        if r["selected"]:
            cells[0] = r"\rowcolor{accent!10}" + cells[0]
        rows.append(cells)
    centered = r">{\centering\arraybackslash}"
    spec = "".join(centered + f"p{{{w}}}" for w in ("0.8cm", "1.4cm", "1.0cm", "1.9cm", "1.2cm", "1.2cm", "1.2cm", "1.2cm", "1.2cm", "1.7cm")) + r">{\raggedright\arraybackslash}X"
    header = ["Rank", "Learning rate", "Epochs", "Loss", "Val. margin", "EOS of 40", "Loops of 40", "EOS of 22", "Loops of 22", "Style W/T/L", "Filter"]
    (out / "table_lora_sweep.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def tex_deviations(root, out, runs):
    try:
        sft = js(root / "results/sft/config.json")
        manifest = js(root / "results/sft/sft_data_manifest.json")["train"]
        lo = js(root / "results/lora/final_config.json")
        sft_peak = peak_memory(root, "sft", "peak_mem_gb")
        ref_peaks = [peak_memory(root, n, "peak_mem_ref_gb") for n in ("dpo_hh", "dpo_uf")]
        zero = {n: step0_reward(root, f"dpo_{n}") for n in ("hh", "uf")}
    except FileNotFoundError as e:
        print("skip tex_deviations:", e)
        return
    by = {r["run"]: r for r in runs}
    plain, nll, e10, r8 = by["r16_lr5e-5"], by["r16_lr5e-5_rpo1"], by["r16_lr5e-5_e10"], by["r8_lr5e-5"]
    lr = {name: by[name] for name in ("r16_lr1e-5", "r16_lr7e-5", "r16_lr1e-4", "r16_lr2e-4")}
    micro = math.ceil(lo["n_train"] / lo["per_device_batch_size"])
    updates = math.ceil(micro / lo["grad_accum"])
    doubled = math.ceil(micro / (2 * lo["grad_accum"]))
    band = lambda text: [band_row(text, 4)]
    rows = [
        band("Resource-driven"),
        ["SFT validation batch", "not specified", f"{sft['eval_batch_size']}, down from 8",
         f"Validation ran out of memory on the T4 while training peaked at {sft_peak:.1f} GB, so I halved the batch and set the "
         "expandable-segments allocator. Validation loss is an exact token mean, so no value changes."],
        band("Data and implementation choices"),
        ["Long SFT examples", "truncate to the maximum length",
         f"skip rows over {sft['max_length']} tokens ({manifest['skipped_too_long']:,} of {manifest['seen']:,} seen) until {sft['n_train']:,} fit",
         "Truncation would cut the EOS label, so the model would never learn to stop. I did not test truncation."],
        ["DPO reference precision", "fp16 allowed", "fp32 weights under fp16 autocast",
         f"The reference needed only {min(ref_peaks):.1f}--{max(ref_peaks):.1f} GB on its own T4, so I could afford fp32. Policy and reference "
         f"then start identical: the largest absolute implicit reward at step 0 was {sci(zero['hh'], 1)} (HH-RLHF) and {sci(zero['uf'], 1)} (UltraFeedback)."],
        band("Performance-driven (LoRA persona task, where the guide leaves these settings to me)"),
        ["Loss", "plain DPO", "DPO plus NLL on the chosen answer, weight 1",
         f"Plain DPO at {sci(5e-5)} left no persona on the required prompts (style {plain['wtl40']}). Adding NLL gave small, mixed changes: held-out EOS "
         f"{plain['eos22']} to {nll['eos22']} and required EOS {plain['eos40']} to {nll['eos40']}."],
        ["Learning rate", "my choice", sci(lo["lr"]),
         rf"Swept {sci(1e-5)} to {sci(2e-4)} (Table~\ref{{tab:lorasweep}}). The margin barely moved at {sci(1e-5)} ({lr['r16_lr1e-5']['margin']:.2f}), "
         f"and required-prompt EOS fell from {plain['eos40']} at {sci(5e-5)} to {lr['r16_lr7e-5']['eos40']} at {sci(7e-5)}, "
         f"{lr['r16_lr1e-4']['eos40']} at {sci(1e-4)}, and {lr['r16_lr2e-4']['eos40']} at {sci(2e-4)}."],
        ["Epochs", "my choice", f"{lo['epochs']:g}",
         f"Ten epochs at {sci(5e-5)} gave the strongest plain-DPO style ({e10['wtl40']}), but EOS fell to {e10['eos40']} on the required prompts and the filter rejected it."],
        ["Rank", "my choice", f"{lo['lora_r']} (alpha {lo['lora_alpha']})",
         f"At {sci(5e-5)}, rank 8 reached about half the validation margin of rank {lo['lora_r']} ({r8['margin']:.2f} against {plain['margin']:.2f}); "
         "both reached accuracy 1.0. I did not try rank 8 at a higher rate."],
        ["Targets and batch", "my choice",
         rf"all seven linear projections; {lo['per_device_batch_size']} $\times$ {lo['grad_accum']} = {lo['effective_batch_size']}",
         f"I did not ablate either. All linear layers give the adapter capacity in attention and MLP, and a batch of {lo['effective_batch_size']} gives "
         f"{updates} updates per epoch on {lo['n_train']} pairs, ({2 * lo['effective_batch_size']} would give {doubled})."],
    ]
    left = r">{\raggedright\arraybackslash}"
    spec = left + "p{2.3cm}" + left + "p{2.5cm}" + left + "p{3.1cm}" + left + "X"
    (out / "table_deviations.tex").write_text(latex_tabular(["Setting", "Guide", "Mine", "Reason and effect"], rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def fig_dpo_rewards(root, out):
    paths = {n: root / f"results/dpo_{n}/metrics.jsonl" for n in ("hh", "uf")}
    if not all(p.exists() for p in paths.values()):
        print("skip fig_dpo_rewards: DPO metrics missing")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, (name, path) in zip(ax, paths.items()):
        e = [r for r in jl(path) if r["type"] == "eval"]
        steps = [r["step"] for r in e]
        a.plot(steps, [r["val_chosen_reward"] for r in e], "o-", label="chosen")
        a.plot(steps, [r["val_rejected_reward"] for r in e], "s-", label="rejected")
        a.plot(steps, [r["val_reward_margin"] for r in e], "^--", color="black", label="margin (chosen - rejected)")
        a.axhline(0, color="grey", lw=0.8)
        a.set(title=f"{name.upper()}-DPO: validation implicit rewards", xlabel="optimizer step")
        a.legend(fontsize=8)
    ax[0].set_ylabel("implicit reward (validation mean)")
    fig.text(0.5, 0.005, "implicit reward = beta * (log pi_policy - log pi_reference), mean over the 1,000 validation pairs", ha="center", fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig_dpo_rewards.png", dpi=170)
    plt.close(fig)


def fig_lora_training(root, out, ctx):
    runs = ["r16_lr5e-5", "r16_lr5e-5_e10"]
    if ctx:
        runs += [r for r in (ctx["selected"], ctx["strongest"]) if r not in runs]
    paths = {r: root / f"results/lora/runs/{r}_metrics.json" for r in runs}
    if not all(p.exists() for p in paths.values()):
        print("skip fig_lora_training: run metrics missing")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, path in paths.items():
        m = js(path)
        log = [r for r in m["log_history"] if "logps/chosen" in r and "rewards/margins" in r]
        label = run_label(m["args"]) + (" (selected)" if ctx and run == ctx["selected"] else "")
        ax[0].plot([r["step"] for r in log], [r["logps/chosen"] for r in log], label=label)
        ax[1].plot([r["step"] for r in log], [r["rewards/margins"] for r in log], label=label)
    ax[0].set(title="(a) log-probability of the chosen (persona) answer", xlabel="optimizer step", ylabel="training batch mean")
    ax[1].set(title="(b) reward margin, chosen - rejected", xlabel="optimizer step", ylabel="training batch mean")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_lora_training.png", dpi=170)
    plt.close(fig)


def table_judge_reliability(root, out, labelled):
    rows, retried = [], []
    for fn, label in labelled:
        p = root / f"results/judgments/{fn}_judged.jsonl.summary.json"
        if not p.exists():
            continue
        s = js(p)
        last = s["retry_runs"][-1] if s.get("retry_runs") else None
        if last:
            retried.append(last)
        rows.append([label + (" (+ retry)" if last else ""), s["n_pairs"], s["rubric"], s["judge_calls"], s["batches_run"],
                     s["judge_calls"] - s["batches_run"], s.get("batches_split_after_failure", 0), s["failed_batches"], s["unknown"],
                     s["auto_ties_identical"]])
    if rows:
        header = ["comparison", "pairs", "rubric", "judge calls", "batches of 5", "extra calls (retries, splits)", "batches split after failure",
                  "failed batches", "unknown verdicts", "identical outputs (auto-tie)"]
        text = md_table(header, rows)
        if retried:
            before, after = sum(r["unknown_before"] for r in retried), sum(r["unknown_after"] for r in retried)
            text += (f"\nRows marked (+ retry) include one retry run of their unknown verdicts, so call and batch counts are cumulative; "
                     f"the retry resolved {before - after} of {before} unknown verdicts.\n")
        (out / "table_judge_reliability.md").write_text(text)


def agreement(recs_a, recs_b):
    a, b = {r["pair_id"]: r["outcome"] for r in recs_a}, {r["pair_id"]: r["outcome"] for r in recs_b}
    ids = [i for i in a if i in b and "unknown" not in (a[i], b[i])]
    n = len(ids)
    same = sum(a[i] == b[i] for i in ids)
    decisive = [i for i in ids if a[i] != "tie" and b[i] != "tie"]
    same_winner = sum(a[i] == b[i] for i in decisive)
    cats = ("test_win", "ref_win", "tie")
    expected = sum((sum(a[i] == c for i in ids) / n) * (sum(b[i] == c for i in ids) / n) for c in cats) if n else 0
    kappa = (same / n - expected) / (1 - expected) if n and expected < 1 else None
    return {"n": n, "same": same, "decisive": len(decisive), "same_winner": same_winner, "kappa": kappa}


def table_judge_consistency(root, out):
    rows = []
    for fa, fb, label in CONSISTENCY:
        ra, rb = load_judged(root, fa), load_judged(root, fb)
        if not (ra and rb):
            continue
        g = agreement(ra, rb)
        if not g["n"]:
            continue
        rows.append([label, g["n"], f"{g['same']} ({g['same'] / g['n']:.0%})", g["decisive"],
                     f"{g['same_winner']} ({g['same_winner'] / g['decisive']:.0%})" if g["decisive"] else "-",
                     g["decisive"] - g["same_winner"], "-" if g["kappa"] is None else f"{g['kappa']:.2f}"])
    if rows:
        (out / "table_judge_consistency.md").write_text(md_table(
            ["comparison", "pairs judged in both runs", "same verdict", "pairs decisive in both", "same winner", "winner flipped", "Cohen's kappa (3 outcomes)"],
            rows))


def shown_first(recs):
    first, second = Counter(), Counter()
    for r in recs:
        if r["outcome"] in ("test_win", "ref_win"):
            test_first = bool(r["_meta"]["test_is_a"]) != bool(r["swap"])
            (first if test_first else second)[r["outcome"]] += 1
    return first, second


def table_judge_bias(root, out):
    rows = []
    for fn, label in BIAS_RUNS:
        for suffix, order in (("", "original"), ("_swapped", "swapped")):
            recs = load_judged(root, fn + suffix)
            if not recs:
                continue
            picks = Counter(r["winner_position"] for r in recs if r["outcome"] != "unknown")
            first, second = shown_first(recs)
            rate = lambda c: f"{c['test_win'] / (c['test_win'] + c['ref_win']):.3f} (n={c['test_win'] + c['ref_win']})" if c else "-"
            rows.append([label, order, picks["A"], picks["B"], picks["TIE"], rate(first), rate(second)])
    if rows:
        (out / "table_judge_bias.md").write_text(md_table(
            ["comparison", "positions", "judge picked first-shown", "picked second-shown", "ties",
             "model under test wins when shown first", "wins when shown second"], rows))


def grouped(recs, key):
    buckets = {}
    for r in recs:
        buckets.setdefault(r["_meta"][key] or "none", []).append(r)
    return {k: sj.stats(v) for k, v in buckets.items()}


def table_hh_categories(root, out):
    hh, uf = load_judged(root, "dpo_hh_vs_sft_hh300"), load_judged(root, "dpo_uf_vs_sft_hh300")
    if not (hh and uf):
        print("skip table_hh_categories: judged HH files missing")
        return
    a, b = grouped(hh, "category"), grouped(uf, "category")
    order = sorted(a, key=lambda k: -(a[k]["win_rate_excl_ties"] or 0))
    rows = [[k, a[k]["n"], wtl(a[k]), rate_ci(a[k]), wtl(b[k]), rate_ci(b[k])] for k in order]
    rows.append(["all categories", sum(v["n"] for v in a.values()), wtl(sj.stats(hh)), rate_ci(sj.stats(hh)), wtl(sj.stats(uf)), rate_ci(sj.stats(uf))])
    (out / "table_hh_categories.md").write_text(md_table(
        ["HH-eval category", "n", "DPO-HH vs SFT W/T/L", "DPO-HH win rate (no ties) [95% CI]", "DPO-UF vs SFT W/T/L", "DPO-UF win rate (no ties) [95% CI]"], rows))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ys = list(range(len(order)))[::-1]
    for dy, stats, name, color in ((0.17, a, "DPO-HH vs SFT", COLORS["DPO-HH"]), (-0.17, b, "DPO-UF vs SFT", COLORS["DPO-UF"])):
        for y, k in zip(ys, order):
            s = stats[k]
            if s["win_rate_excl_ties"] is None:
                continue
            lo, hi = s["wilson95_excl_ties"]
            ax.errorbar(s["win_rate_excl_ties"], y + dy, xerr=[[s["win_rate_excl_ties"] - lo], [hi - s["win_rate_excl_ties"]]], fmt="o", color=color,
                        capsize=2, label=name if y == ys[0] else None)
    ax.axvline(0.5, color="grey", ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([k.replace("_", " ") for k in order], fontsize=8)
    ax.set(xlim=(0, 1), xlabel="win rate over SFT on the HH eval, ties excluded (95% Wilson interval, n=30 per category)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_hh_categories.png", dpi=170)
    plt.close(fig)


def rate_ci2(s):
    if s["win_rate_excl_ties"] is None:
        return "-"
    lo, hi = s["wilson95_excl_ties"]
    return f"{s['win_rate_excl_ties']:.2f} [{lo:.2f}, {hi:.2f}]"


def refusal_rates(root):
    g = root / "results/generations"
    files = ("sft_qwen2.5-0.5b_hh300.json", "dpo_hh_qwen2.5-0.5b_hh300.json", "dpo_uf_qwen2.5-0.5b_hh300.json")
    eval_path = root / "eval_sets/hh_rlhf_eval_300_seed42.json"
    if not (all((g / f).exists() for f in files) and eval_path.exists()):
        return None
    cat = {r["id"]: r["category"] for r in js(eval_path)}
    data = [js(g / f) for f in files]
    rate = lambda recs: statistics.mean(bool(REFUSAL.search(r["output"])) for r in recs) if recs else 0.0
    by_cat = {c: [rate([r for r in recs if cat[r["id"]] == c]) for recs in data] for c in set(cat.values())}
    return by_cat, [rate(recs) for recs in data]


def tex_hh_categories(root, out):
    hh, uf = load_judged(root, "dpo_hh_vs_sft_hh300"), load_judged(root, "dpo_uf_vs_sft_hh300")
    refusals = refusal_rates(root)
    if not (hh and uf and refusals):
        print("skip tex_hh_categories: judged HH files or generations missing")
        return
    by_cat, overall = refusals
    a, b = grouped(hh, "category"), grouped(uf, "category")
    order = sorted(a, key=lambda k: -(a[k]["win_rate_excl_ties"] or 0))
    fmt = lambda rates: " / ".join(f"{r:.2f}" for r in rates)
    rows = [[tex_escape(k.replace("_", " ")), str(a[k]["n"]), wtl(a[k]), rate_ci2(a[k]), wtl(b[k]), rate_ci2(b[k]), fmt(by_cat[k])] for k in order]
    sh, su = sj.stats(hh), sj.stats(uf)
    rows.append([r"\textbf{all categories}", str(sh["n"]), wtl(sh), rate_ci2(sh), wtl(su), rate_ci2(su), fmt(overall)])
    centered = r">{\centering\arraybackslash}"
    spec = r">{\raggedright\arraybackslash}X" + "".join(centered + f"p{{{w}}}" for w in ("0.7cm", "1.5cm", "2.6cm", "1.5cm", "2.6cm", "2.5cm"))
    header = ["HH-eval category", "n", "DPO-HH W/T/L", r"DPO-HH win rate [95\% CI]", "DPO-UF W/T/L", r"DPO-UF win rate [95\% CI]", "Refusal phrases SFT / HH / UF"]
    (out / "table_hh_categories.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def norm_text(text):
    return " ".join(str(text).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').split())


def smart_quotes(text):
    return re.sub(r'"([^"]*)"', r"``\1''", text)


EXAMPLE_ROWS = [
    {"label": "HH-eval 0003 (legal)", "set": "hh", "id": "hh-eval-42-0003", "who": "DPO-HH",
     "sft": ["it is always advisable to check with your local authorities or legal representatives"],
     "dpo": ["it is not advisable to keep a package delivered to you by mistake", "contact the sender or the delivery company"],
     "note": "Both open with the same disclaimer, and DPO-HH adds a concrete next step."},
    {"label": "HH-eval 0009 (rewrite)", "set": "hh", "id": "hh-eval-42-0009", "who": "DPO-HH",
     "sft": ["Please stop being so negative and stop being so mean to me."],
     "dpo": ["I hope that everyone finds out that you are not useful"],
     "note": "DPO-HH repeats the hostile message instead of rewriting it."},
    {"label": "AlpacaEval 0124", "set": "uf", "id": "alpaca-eval-42-0124", "who": "DPO-UF",
     "sft": ["approximately 141 million square kilometers."],
     "dpo": ["approximately 141 million square kilometers (54 million square miles)."],
     "note": "Same answer, and DPO-UF adds the unit conversion."},
    {"label": "AlpacaEval 0004", "set": "uf", "id": "alpaca-eval-42-0004", "who": "DPO-UF",
     "sft": ["Larry Page is the co-founder of Google, the search engine company."],
     "dpo": ["born on February 1, 1953", "passed away on June 16, 2011"],
     "note": "DPO-UF writes ten times more and invents a death date, although Page is alive."},
    {"label": "AlpacaEval 0027", "set": "uf", "id": "alpaca-eval-42-0027", "who": "DPO-UF",
     "sft": ["was by the lead singer, John Paul Jones."], "dpo": ["was by guitarist John Paul Jones."],
     "note": "Both models give the same wrong name."},
]

PERSONA_EXAMPLE = {"label": "Persona 0031 (safety)", "id": "persona-eval-0031", "base": ["I'm so sorry to hear that you're feeling this way."],
                   "selected": ["Ah, you're feeling like a total failure? Well, you're not!"],
                   "strongest": ["You're not worthless, you're just a mess of burnt food!", "You're a culinary genius, you're a culinary genius!"]}


def quoted(fragments, record):
    output = norm_text(record["output"])
    for f in fragments:
        if norm_text(f) not in output:
            raise ValueError(f"fragment not found in {record['id']}: {f}")
    return smart_quotes('"' + r" \ldots{} ".join(tex_escape(f) for f in fragments) + '"') + f" ({record['num_new_tokens']} tokens)"


def outcome_word(outcome, who):
    return {"test_win": who, "ref_win": "SFT", "tie": "tie"}[outcome]


def prompt_text(record, words=12):
    parts = norm_text(record["instruction"]).split()
    text = " ".join(parts[:words]) + (" ..." if len(parts) > words else "")
    if text.count('"') % 2:
        text += '"'
    return smart_quotes(tex_escape(text).replace("...", r"\ldots{}"))


def tex_examples(root, out, ctx):
    g = root / "results/generations"
    names = {"hh": ("sft_qwen2.5-0.5b_hh300.json", "dpo_hh_qwen2.5-0.5b_hh300.json", "dpo_hh_vs_sft_hh300"),
             "uf": ("sft_qwen2.5-0.5b_alpaca300.json", "dpo_uf_qwen2.5-0.5b_alpaca300.json", "dpo_uf_vs_sft_alpaca300")}
    try:
        loaded = {k: ({r["id"]: r for r in js(g / f[0])}, {r["id"]: r for r in js(g / f[1])},
                      {r["id"]: r["outcome"] for r in load_judged(root, f[2])}, {r["id"]: r["outcome"] for r in load_judged(root, f[2] + "_swapped")})
                  for k, f in names.items()}
    except (FileNotFoundError, TypeError) as e:
        print("skip tex_examples:", e)
        return
    rows = []
    for ex in EXAMPLE_ROWS:
        sft, dpo, first, second = loaded[ex["set"]]
        a, b = first[ex["id"]], second[ex["id"]]
        verdict = f"{outcome_word(a, ex['who'])} preferred in both orders" if a == b and a != "tie" else ("tie in both orders" if a == b else "judge verdict changed with the order")
        rows.append([f"{ex['label']}: {prompt_text(sft[ex['id']])}", quoted(ex["sft"], sft[ex["id"]]), quoted(ex["dpo"], dpo[ex["id"]]), f"{verdict}. {ex['note']}"])
    if ctx:
        ex = PERSONA_EXAMPLE
        base = {r["id"]: r for r in js(g / "persona_base.json")}[ex["id"]]
        adapters = {k: {r["id"]: r for r in js(g / f"persona_{ctx[k]}.json")}[ex["id"]] for k in ("selected", "strongest")}
        notes = []
        for kind, label in (("style", "style"), ("safety", "safety"), ("useful", "usefulness")):
            out_of = {k: {r["id"]: r["outcome"] for r in load_judged(root, f"persona40_{ctx[k]}_vs_base_{kind}")}[ex["id"]] for k in ("selected", "strongest")}
            words = {k: {"test_win": "adapter preferred", "ref_win": "base preferred", "tie": "tie"}[v] for k, v in out_of.items()}
            notes.append(f"{label}: {words['selected']} for the selected adapter and {words['strongest']} for the strongest" if words["selected"] != words["strongest"]
                         else f"{label}: {words['selected']} for both")
        adapter_cell = ("Selected: " + quoted(ex["selected"], adapters["selected"]) + " Strongest: " + quoted(ex["strongest"], adapters["strongest"]))
        rows.append([f"{ex['label']}: {prompt_text(base)}", quoted(ex["base"], base), adapter_cell, "Judge, " + "; ".join(notes) + "."])
    if not rows:
        return
    left = r">{\raggedright\arraybackslash}"
    spec = left + "p{3.7cm}" + left + "X" + left + "X" + left + "p{3.7cm}"
    header = ["Example and prompt", "SFT or base model", "DPO model or LoRA adapter", "Judge and reading"]
    (out / "table_examples.tex").write_text(latex_tabular(header, rows, spec, size=r"\footnotesize", tabcolsep="3pt"))


def table_alpaca_sources(root, out):
    uf, hh = load_judged(root, "dpo_uf_vs_sft_alpaca300"), load_judged(root, "dpo_hh_vs_sft_alpaca300")
    if not (uf and hh):
        print("skip table_alpaca_sources: judged AlpacaEval files missing")
        return
    a, b = grouped(uf, "source_dataset"), grouped(hh, "source_dataset")
    order = sorted(a, key=lambda k: -a[k]["n"])
    rows = [[k, a[k]["n"], wtl(a[k]), rate_ci(a[k]), wtl(b[k]), rate_ci(b[k])] for k in order]
    rows.append(["all sources", sum(v["n"] for v in a.values()), wtl(sj.stats(uf)), rate_ci(sj.stats(uf)), wtl(sj.stats(hh)), rate_ci(sj.stats(hh))])
    (out / "table_alpaca_sources.md").write_text(md_table(
        ["AlpacaEval source", "n", "DPO-UF vs SFT W/T/L", "DPO-UF win rate (no ties) [95% CI]", "DPO-HH vs SFT W/T/L", "DPO-HH win rate (no ties) [95% CI]"], rows))


def table_refusals(root, out):
    g = root / "results/generations"
    files = {"SFT": "sft_qwen2.5-0.5b_hh300.json", "DPO-HH": "dpo_hh_qwen2.5-0.5b_hh300.json", "DPO-UF": "dpo_uf_qwen2.5-0.5b_hh300.json"}
    if not (all((g / f).exists() for f in files.values()) and (root / "eval_sets/hh_rlhf_eval_300_seed42.json").exists()):
        print("skip table_refusals: generations missing")
        return
    cat = {r["id"]: r["category"] for r in js(root / "eval_sets/hh_rlhf_eval_300_seed42.json")}
    data = {name: js(g / f) for name, f in files.items()}
    rate = lambda recs: statistics.mean(bool(REFUSAL.search(r["output"])) for r in recs) if recs else 0.0
    cats = sorted(set(cat.values()))
    rows = [[c, sum(cat[r["id"]] == c for r in data["SFT"])] + [f"{rate([r for r in recs if cat[r['id']] == c]):.2f}" for recs in data.values()] for c in cats]
    rows.append(["all categories", len(data["SFT"])] + [f"{rate(recs):.2f}" for recs in data.values()])
    text = md_table(["HH-eval category", "n", "SFT", "DPO-HH", "DPO-UF"], rows)
    text += "\nShare of answers containing a refusal phrase (I'm sorry, I cannot, I can't, I won't, unable to, ...). A crude proxy, not a safety score.\n"
    (out / "table_hh_refusals.md").write_text(text)


def fig_lengths(root, out):
    g = root / "results/generations"
    panels = [("AlpacaEval-300", [("base", "baseline_qwen2.5-0.5b-base_alpaca300.json"), ("SFT", "sft_qwen2.5-0.5b_alpaca300.json"),
                                   ("DPO-HH", "dpo_hh_qwen2.5-0.5b_alpaca300.json"), ("DPO-UF", "dpo_uf_qwen2.5-0.5b_alpaca300.json")]),
              ("HH-eval-300", [("SFT", "sft_qwen2.5-0.5b_hh300.json"), ("DPO-HH", "dpo_hh_qwen2.5-0.5b_hh300.json"),
                               ("DPO-UF", "dpo_uf_qwen2.5-0.5b_hh300.json")])]
    if not all((g / f).exists() for _, items in panels for _, f in items):
        print("skip fig_lengths: generations missing")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, (title, items) in zip(ax, panels):
        for name, f in items:
            toks = sorted(r["num_new_tokens"] for r in js(g / f))
            a.step(toks, [(i + 1) / len(toks) for i in range(len(toks))], where="post", color=COLORS[name],
                   label=f"{name} (mean {statistics.mean(toks):.0f})")
        a.axvline(512, color="grey", ls="--", lw=1)
        a.text(506, 0.55, "512-token cap", fontsize=7, color="grey", ha="right", va="center", rotation=90)
        a.set(title=title, xlabel="generated tokens", xlim=(0, 540))
        a.legend(fontsize=8, loc="lower right")
    ax[0].set_ylabel("share of answers up to this length")
    fig.tight_layout()
    fig.savefig(out / "fig_lengths.png", dpi=170)
    plt.close(fig)


def penalty_text(root, stem):
    p = root / "results/generations" / f"{stem}.meta.json"
    return f"repetition penalty {js(p)['decoding']['repetition_penalty']:g}" if p.exists() else "repetition penalty ?"


def table_decoding_sensitivity(root, out):
    verdicts, behaviour = [], []
    g = root / "results/generations"
    for fn, label, setname, models in DECODING_PAIRS:
        for suffix in ("", f"_{DECODING_TAG}"):
            recs = load_judged(root, fn + suffix)
            if recs is None or not (g / f"{models[1][1]}{suffix}.json").exists():
                break
        else:
            for suffix in ("", f"_{DECODING_TAG}"):
                setting = penalty_text(root, models[1][1] + suffix) + (" (main)" if not suffix else "")
                s = sj.stats(load_judged(root, fn + suffix))
                p = s["sign_test_p"]
                verdicts.append([label, setting, wtl(s), rate_ci(s), f"{s['win_rate_ties_half']:.3f}",
                                 "-" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}"), s["unknown"]])
                for name, stem in models:
                    res = js(g / f"{stem}{suffix}.json")
                    toks = [r["num_new_tokens"] for r in res]
                    behaviour.append([name, setname, setting, f"{sum(bool(r['hit_eos']) for r in res)}/{len(res)}", sum(t >= 512 for t in toks),
                                      round(statistics.mean(toks)), sum(looping(r["output"]) for r in res)])
    if not verdicts:
        print("skip table_decoding_sensitivity: repetition-penalty runs not available")
        return
    text = "Same models, prompts, batch composition and judge; only the repetition penalty changes.\n\n"
    text += md_table(["comparison", "decoding", "W/T/L", "win rate (ties excluded) [95% CI]", "win rate (ties = 1/2)", "sign-test p", "unknown"], verdicts)
    text += "\n" + md_table(["model", "eval set", "decoding", "ended with EOS", "hit 512-token cap", "mean tokens", "repeated-phrase outputs"], behaviour)
    (out / "table_decoding_sensitivity.md").write_text(text)


def clip(text, n=520):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def quote(text):
    return "> " + clip(text)


def example_ids(orig, swapped, outcome, pairs, k, max_chars=700):
    pool = [pid for pid in sorted(orig) if orig[pid]["outcome"] == outcome and swapped.get(pid, {}).get("outcome") == outcome]

    def texts(pid):
        row = pairs[pid]
        a, b = row["response_a"], row["response_b"]
        return (a, b) if row["_meta"]["test_is_a"] else (b, a)

    for limit in (max_chars, 1400, 10 ** 6):
        picked = [pid for pid in pool if max(len(t) for t in texts(pid)) <= limit and texts(pid)[0].strip() != texts(pid)[1].strip()]
        if len(picked) >= k or limit == 10 ** 6:
            return picked[:k], texts
    return [], texts


def build_examples(root, out):
    sections = [("dpo_hh_vs_sft_hh300", "dpo_hh_vs_sft_hh300_swapped", "DPO-HH", "HH eval"),
                ("dpo_uf_vs_sft_alpaca300", "dpo_uf_vs_sft_alpaca300_swapped", "DPO-UF", "AlpacaEval")]
    lines = ["# Candidate qualitative examples (chosen automatically)", "",
             "Only pairs where the judge gave the same verdict in the original and the position-swapped run are listed, shortest first within the "
             "first 700 characters per answer. Pick two or three for the report; they illustrate the aggregate numbers, they do not replace them.", ""]
    found = False
    for fn, fn_sw, name, setname in sections:
        orig, swapped = load_judged(root, fn), load_judged(root, fn_sw)
        pairs_path = root / f"results/eval/{fn}_pairs.jsonl"
        if not (orig and swapped and pairs_path.exists()):
            continue
        found = True
        o, s = {r["pair_id"]: r for r in orig}, {r["pair_id"]: r for r in swapped}
        pairs = {r["pair_id"]: r for r in jl(pairs_path)}
        lines += [f"## {name} vs SFT on {setname}", ""]
        for outcome, heading in (("test_win", f"{name} preferred in both orders"), ("ref_win", f"SFT preferred in both orders")):
            ids, texts = example_ids(o, s, outcome, pairs, 2)
            lines += [f"### {heading}", ""]
            for pid in ids:
                m = o[pid]["_meta"]
                tag = m.get("category") or m.get("source_dataset") or ""
                test_text, ref_text = texts(pid)
                lines += [f"**{pid}** ({tag}); judge, original order: {o[pid]['reason']} Judge, swapped order: {s[pid]['reason']}", "",
                          f"Prompt: {clip(pairs[pid]['instruction'], 300)}", "", f"{name}:", "", quote(test_text), "", "SFT:", "", quote(ref_text), ""]
            if not ids:
                lines += ["(none)", ""]
    if found:
        (out / "examples.md").write_text("\n".join(lines))


def build_persona_examples(root, out, ctx):
    g = root / "results/generations"
    if not ctx or not (g / "persona_heldout_base.json").exists():
        return
    tag = lambda run: f"LoRA {run.replace('r16_', '')}"
    runs = [("base", None), (f"{tag(ctx['selected'])} (selected)", ctx["selected"])]
    if ctx["strongest"] != ctx["selected"]:
        runs.append((f"{tag(ctx['strongest'])} (strongest persona)", ctx["strongest"]))
    lines = ["# Candidate persona examples (chosen automatically)", "",
             "The first two held-out prompts and the first prompt of each required category, so nothing is cherry-picked.", ""]
    sets = [("held-out prompts", "persona_heldout_base.json", "persona_heldout_{}.json", 2),
            ("required prompts", "persona_base.json", "persona_{}.json", None)]
    for title, base_file, pattern, take in sets:
        by = {name: {r["id"]: r for r in js(g / (base_file if run is None else pattern.format(run)))} for name, run in runs}
        base = by["base"]
        if take:
            ids = list(base)[:take]
        else:
            seen, ids = set(), []
            for pid, r in base.items():
                if r["category"] not in seen:
                    seen.add(r["category"])
                    ids.append(pid)
        lines += [f"## {title}", ""]
        for pid in ids:
            lines += [f"**{pid}** ({base[pid].get('category') or 'held-out'}): {clip(base[pid]['instruction'], 260)}", ""]
            for name in by:
                r = by[name][pid]
                lines += [f"{name} (ended on its own: {r['hit_eos']}):", "", quote(r["output"]), ""]
    (out / "examples_persona.md").write_text("\n".join(lines))


INDEX = [
    ("fig_training.png", "14.4", "SFT and DPO training curves, checkpoints marked, smoothing stated"),
    ("fig_dpo_rewards.png", "14.4 / 14.6", "implicit chosen/rejected rewards on validation: both fall, the margin grows"),
    ("fig_lora_training.png", "14.4 / 14.3", "why plain DPO barely raises the persona likelihood and the NLL term does"),
    ("table_data.md", "14.1", "datasets, splits, filtering, token statistics"),
    ("table_eval_sets.md", "14.1", "the four fixed evaluation sets"),
    ("table_config.md", "14.2", "full configuration table with generation settings"),
    ("table_deviations.md", "14.3", "guide value vs used value, with the unchanged parameters listed"),
    ("table_lora_sweep.md", "14.3", "all LoRA runs, degeneration filter and judged style"),
    ("fig_lora_sweep.png", "14.3 / 14.6", "persona strength against coherence, per evaluation set"),
    ("table_generation.md", "14.2 / 14.6", "EOS, cap hits, length and repeated-phrase counts per model"),
    ("fig_lengths.png", "14.6", "answer length distributions (verbosity shift under DPO)"),
    ("table_judge.md", "14.5", "wins, ties, losses, win rates, CIs and p-values for every comparison"),
    ("fig_winrates.png", "14.5", "the same as a forest plot"),
    ("table_judge_reliability.md", "14.5", "judge calls, retries, failed batches and unknown verdicts"),
    ("table_judge_consistency.md", "14.5 / 14.7", "agreement between original and swapped / plain-rubric runs"),
    ("table_judge_bias.md", "14.7", "position bias of the judge"),
    ("table_judge_strata.md", "14.6", "win rate split by length relation"),
    ("table_hh_categories.md", "14.6", "HH-eval win rates per category for both DPO models"),
    ("fig_hh_categories.png", "14.6", "the same as a dot plot"),
    ("table_alpaca_sources.md", "14.6", "AlpacaEval win rates per prompt source"),
    ("table_hh_refusals.md", "14.6", "refusal-phrase rate per HH category"),
    ("table_decoding_sensitivity.md", "14.7", "main comparisons again with a repetition penalty: do the conclusions change?"),
    ("examples.md", "14.6", "candidate DPO vs SFT examples (consistent across both judge orders)"),
    ("examples_persona.md", "14.6", "candidate persona examples for base, selected and strongest adapter"),
    ("key_numbers.json", "-", "numbers behind the tables, for quoting in the text"),
]


def write_index(out):
    rows = [[f"[{f}]({f})", sec, note] for f, sec, note in INDEX if (out / f).exists()]
    (out / "index.md").write_text("# Report assets\n\nGenerated by scripts/make_report_assets.py\n\n" + md_table(["file", "report section", "content"], rows))


def build_all(root, out, ctx, labelled):
    table_data(root, out)
    table_eval_sets(root, out)
    tex_data(root, out)
    tex_eval_sets(root, out)
    tex_config(root, out)
    tex_dpo_results(root, out)
    tex_hh_categories(root, out)
    tex_examples(root, out, ctx)
    tex_lora_results(root, out, ctx)
    table_deviations(root, out)
    fig_dpo_rewards(root, out)
    fig_lora_training(root, out, ctx)
    table_judge_reliability(root, out, labelled)
    table_judge_consistency(root, out)
    table_judge_bias(root, out)
    table_hh_categories(root, out)
    table_alpaca_sources(root, out)
    table_refusals(root, out)
    table_decoding_sensitivity(root, out)
    fig_lengths(root, out)
    build_examples(root, out)
    build_persona_examples(root, out, ctx)
