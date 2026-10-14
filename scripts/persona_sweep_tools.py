import argparse
import csv
import json
from pathlib import Path

import summarize_judgments as sj
from alignment.persona import SPLITS, build_lexicon, select_adapter, split_persona_rows, style_metrics

STYLE_FIELDS = ["generator", "set", "category", "n", "marker_rate", "exclaim_per_resp", "mean_tokens", "hit_eos",
                "repeated_phrase_outputs"]


def run_name(path):
    p = Path(path)
    return p.parent.name if p.name == "metrics.json" else p.name.replace("_metrics.json", "")


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cmd_table(args):
    rows = []
    for p in args.metrics:
        m = json.load(open(p))
        a, log, fe = m["args"], m["log_history"], m["final_eval"]
        train = [r for r in log if "loss" in r and "rewards/margins" in r]
        rows.append({
            "run": run_name(p), "rank": a["lora_r"], "alpha": a["lora_alpha"], "lr": a["lr"], "beta": a["beta"],
            "rpo_alpha": a.get("rpo_alpha", 0.0), "epochs": a["epochs"],
            "optimizer_steps": max((r.get("step", 0) for r in log), default=0), "trainable_params": m["trainable_params"],
            "val_acc": fe["eval_rewards/accuracies"], "val_margin": round(fe["eval_rewards/margins"], 4),
            "val_loss": round(fe["eval_loss"], 4), "val_nll": round(fe["eval_nll_loss"], 4) if "eval_nll_loss" in fe else "",
            "train_loss_first": round(train[0]["loss"], 4) if train else "", "train_loss_last": round(train[-1]["loss"], 4) if train else "",
            "chosen_logp_first": round(train[0]["logps/chosen"], 1) if train else "",
            "chosen_logp_last": round(train[-1]["logps/chosen"], 1) if train else "",
            "train_minutes": round(m["train_minutes"], 2)})
    rows.sort(key=lambda r: (r["rpo_alpha"], r["epochs"], r["lr"], r["rank"]))
    write_csv(args.out_csv, rows, list(rows[0]))
    print(f"{'run':24s}{'rpo':>5s}{'ep':>5s}{'lr':>8s}{'steps':>6s}{'margin':>8s}  {'chosen logp (first -> last step)':s}")
    for r in rows:
        print(f"{r['run']:24s}{r['rpo_alpha']:>5}{r['epochs']:>5}{r['lr']:>8.0e}{r['optimizer_steps']:>6}{r['val_margin']:>8}"
              f"  {r['chosen_logp_first']} -> {r['chosen_logp_last']}")


def cmd_style(args):
    train_rows, _ = split_persona_rows(json.load(open(args.train_data)), args.seed, args.val_size)
    base_outputs = [r["output"] for r in json.load(open(args.base_eval40))]
    lexicon = build_lexicon(train_rows, base_outputs)
    print("marker lexicon (train rows only):", lexicon)
    rows = []
    for spec in args.gen:
        head, path = spec.split("=", 1)
        split, name = head.split(":", 1)
        assert split in SPLITS, f"unknown set {split!r}; use one of {SPLITS}"
        recs = json.load(open(path))
        for cat in sorted({r.get("category") or "none" for r in recs}) + ["ALL"]:
            sub = [r for r in recs if cat == "ALL" or (r.get("category") or "none") == cat]
            rows.append({"generator": name, "set": split, "category": cat, **style_metrics(sub, lexicon)})
    write_csv(args.out_csv, rows, STYLE_FIELDS)
    if args.out_lexicon:
        Path(args.out_lexicon).write_text(json.dumps({"lexicon": lexicon, "built_from": "train rows only"}, indent=2))
    print(f"\n{'generator':26s}{'set':>10s}{'marker':>8s}{'!/resp':>8s}{'tokens':>8s}{'EOS':>5s}{'n':>4s}{'loops':>7s}")
    for r in rows:
        if r["category"] == "ALL":
            print(f"{r['generator']:26s}{r['set']:>10s}{r['marker_rate']:>8}{r['exclaim_per_resp']:>8}{r['mean_tokens']:>8}"
                  f"{r['hit_eos']:>5}{r['n']:>4}{r['repeated_phrase_outputs']:>7}")


def cmd_select(args):
    stats = {}
    for r in csv.DictReader(open(args.style_csv)):
        if r["category"] == "ALL":
            stats.setdefault(r["generator"], {})[r["set"]] = {
                "n": int(r["n"]), "hit_eos": int(r["hit_eos"]), "repeated_phrase_outputs": int(r["repeated_phrase_outputs"])}
    steps = {r["run"]: int(r["optimizer_steps"]) for r in csv.DictReader(open(args.sweep_csv))} if args.sweep_csv else {}
    cands = []
    for name in args.candidates:
        judged = sj.load_last(Path(args.judgments_dir) / f"persona22_{name}_vs_base_style_judged.jsonl")
        s = sj.stats(judged)
        cands.append({"name": name, "steps": steps.get(name, 0), "stats": stats[name],
                      "dev": {"wins": s["wins"], "losses": s["losses"], "ties": s["ties"]}})
    result = select_adapter(cands, stats["base"])
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print("RULE:", result["rule"], "\n")
    print(f"{'candidate':24s}{'W':>4s}{'T':>4s}{'L':>4s}{'score':>7s}{'p':>8s}  status")
    for r in result["candidates"]:
        print(f"{r['name']:24s}{r['dev_wins']:>4}{r['dev_ties']:>4}{r['dev_losses']:>4}{r['dev_style_score']:>7.2f}"
              f"{str(r['dev_sign_test_p']):>8s}  {'DEGENERATE: ' + '; '.join(r['degenerate_reasons']) if r['degenerate'] else 'ok'}")
    print("\nSELECTED:", result["selected"], "| real style shift:", result["selected_style_shift_significant"])


def cmd_overview(args):
    rows = []
    for p in args.judged:
        recs = sj.load_last(p)
        s, m = sj.stats(recs), recs[0]["_meta"]
        test = m["a_model"] if m["test_is_a"] else m["b_model"]
        ref = m["b_model"] if m["test_is_a"] else m["a_model"]
        lo, hi = s["wilson95_excl_ties"]
        rows.append({"file": Path(p).name.replace("_judged.jsonl", ""), "test": test, "ref": ref, "rubric": recs[0]["rubric"],
                     "swap": recs[0]["swap"], "judge": recs[0].get("judge_model_used"), "n": s["n"], "wins": s["wins"],
                     "ties": s["ties"], "losses": s["losses"], "unknown": s["unknown"],
                     "win_rate_excl_ties": s["win_rate_excl_ties"], "ci_low": lo, "ci_high": hi,
                     "win_rate_ties_half": s["win_rate_ties_half"], "sign_test_p": s["sign_test_p"]})
    write_csv(args.out_csv, rows, list(rows[0]))
    print(f"{'comparison':58s}{'rubric':>15s}{'n':>4s}{'W':>4s}{'T':>4s}{'L':>4s}{'unk':>4s}{'win%':>7s}{'p':>8s}")
    for r in rows:
        wr = f"{r['win_rate_excl_ties']:.2f}" if r["win_rate_excl_ties"] is not None else "-"
        print(f"{r['file']:58s}{r['rubric']:>15s}{r['n']:>4}{r['wins']:>4}{r['ties']:>4}{r['losses']:>4}{r['unknown']:>4}"
              f"{wr:>7s}{str(r['sign_test_p']):>8s}")


def cmd_final_config(args):
    m = json.load(open(args.metrics))
    a, fe = m["args"], m["final_eval"]
    sel = json.load(open(args.selection))
    run = run_name(args.metrics)
    cfg = {
        "run": run, "base_model": a["model_name"], "lora_r": a["lora_r"], "lora_alpha": a["lora_alpha"],
        "lora_dropout": a["lora_dropout"], "target_modules": a["target_modules"], "lr": a["lr"], "beta": a["beta"],
        "rpo_alpha": a.get("rpo_alpha", 0.0), "epochs": a["epochs"],
        "optimizer_steps": max((r.get("step", 0) for r in m["log_history"]), default=0),
        "per_device_batch_size": a["batch_size"], "grad_accum": a["grad_accum"],
        "effective_batch_size": a["batch_size"] * a["grad_accum"], "warmup_steps": a["warmup_steps"],
        "lr_scheduler": a["lr_scheduler"], "max_grad_norm": a["max_grad_norm"], "max_length": a["max_length"],
        "seed": a["seed"], "n_train": m["n_train"], "n_val": m["n_val"], "trainable_params": m["trainable_params"],
        "total_params": m["total_params"],
        "final_eval": {k: v for k, v in fe.items() if "rewards" in k or k in ("eval_loss", "eval_nll_loss")},
        "versions": m.get("versions", {}),
        "selected_by": {"selection_file": args.selection, "selected": sel["selected"],
                        "style_shift_significant": sel["selected_style_shift_significant"]},
    }
    assert sel["selected"] == run, f"selection picked {sel['selected']!r}, not {run!r}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfg, indent=2))
    print("wrote", args.out, "for", run)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("final_config")
    f.add_argument("--metrics", required=True)
    f.add_argument("--selection", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_final_config)
    t = sub.add_parser("table")
    t.add_argument("--metrics", nargs="+", required=True)
    t.add_argument("--out_csv", required=True)
    t.set_defaults(fn=cmd_table)
    s = sub.add_parser("style")
    s.add_argument("--gen", nargs="+", required=True, help="SET:NAME=PATH, SET in {eval40, heldout22}")
    s.add_argument("--train_data", default="data/persona_lora/ramsay_style_preferences.json")
    s.add_argument("--base_eval40", default="results/generations/persona_base.json")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--val_size", type=int, default=22)
    s.add_argument("--out_csv", required=True)
    s.add_argument("--out_lexicon", default=None)
    s.set_defaults(fn=cmd_style)
    c = sub.add_parser("select")
    c.add_argument("--style_csv", required=True)
    c.add_argument("--judgments_dir", required=True)
    c.add_argument("--candidates", nargs="+", required=True)
    c.add_argument("--sweep_csv", default=None)
    c.add_argument("--out_json", required=True)
    c.set_defaults(fn=cmd_select)
    o = sub.add_parser("overview")
    o.add_argument("judged", nargs="+")
    o.add_argument("--out_csv", required=True)
    o.set_defaults(fn=cmd_overview)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
