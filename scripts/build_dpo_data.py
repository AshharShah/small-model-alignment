import argparse
import hashlib
import json
import statistics
from pathlib import Path

from alignment.prompts import format_prompt

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def parse_hh(example):
    parts = []
    for key in ("chosen", "rejected"):
        text = example[key]
        if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
            return "not_single_turn"
        h, a = text.index(HUMAN), text.index(ASSISTANT)
        if a < h:
            return "bad_format"
        parts.append((text[h + len(HUMAN):a].strip(), text[a + len(ASSISTANT):].strip()))
    (p_c, chosen), (p_r, rejected) = parts
    if p_c != p_r or not p_c:
        return "prompt_mismatch"
    return p_c, chosen, rejected


def parse_uf(example):
    chosen_msgs, rejected_msgs = example["chosen"], example["rejected"]
    if len(chosen_msgs) != 2 or len(rejected_msgs) != 2:
        return "not_single_turn"
    if [m["role"] for m in chosen_msgs] != ["user", "assistant"] or [m["role"] for m in rejected_msgs] != ["user", "assistant"]:
        return "bad_format"
    prompt = chosen_msgs[0]["content"].strip()
    if prompt != rejected_msgs[0]["content"].strip() or not prompt:
        return "prompt_mismatch"
    if not example["score_chosen"] > example["score_rejected"]:
        return "tie_or_reversed_score"
    return prompt, chosen_msgs[1]["content"].strip(), rejected_msgs[1]["content"].strip()


PARSERS = {"hh": parse_hh, "uf": parse_uf}


def collect(rows, parse, tokenizer, n, max_length, max_prompt_length, stats):
    kept = []
    for ex in rows:
        stats["seen"] += 1
        parsed = parse(ex)
        if isinstance(parsed, str):
            stats["skipped_" + parsed] = stats.get("skipped_" + parsed, 0) + 1
            continue
        prompt, chosen, rejected = parsed
        if not chosen or not rejected or chosen == rejected:
            stats["skipped_empty_or_identical"] = stats.get("skipped_empty_or_identical", 0) + 1
            continue
        n_p = len(tokenizer.encode(format_prompt(prompt), add_special_tokens=False))
        n_c = len(tokenizer.encode(chosen, add_special_tokens=False)) + 1
        n_r = len(tokenizer.encode(rejected, add_special_tokens=False)) + 1
        if n_p > max_prompt_length or n_p + n_c > max_length or n_p + n_r > max_length:
            stats["skipped_too_long"] = stats.get("skipped_too_long", 0) + 1
            continue
        kept.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                     "prompt_tokens": n_p, "chosen_tokens": n_c, "rejected_tokens": n_r})
        if len(kept) == n:
            break
    return kept


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def summarize(rows):
    return {
        "n": len(rows),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
        "mean_chosen_tokens": round(statistics.mean(r["chosen_tokens"] for r in rows), 1),
        "mean_rejected_tokens": round(statistics.mean(r["rejected_tokens"] for r in rows), 1),
        "frac_chosen_longer": round(sum(r["chosen_tokens"] > r["rejected_tokens"] for r in rows) / len(rows), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hh", "uf"], required=True)
    ap.add_argument("--out_dir", default="outputs/data")
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"dpo_{args.dataset}_manifest.json"
    if manifest_path.exists():
        print("manifest exists, nothing to do:", manifest_path)
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.dataset == "hh":
        repo, train_split, val_split = "Anthropic/hh-rlhf", "train", "test"
    else:
        repo, train_split, val_split = "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs"
    parse = PARSERS[args.dataset]
    kw = dict(parse=parse, tokenizer=tok, max_length=args.max_length, max_prompt_length=args.max_prompt_length)

    manifest = {"dataset": repo, "seed": args.seed, "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length, "tokenizer": args.tokenizer}

    train_iter = iter(load_dataset(repo, split=train_split).shuffle(seed=args.seed))
    stats = {"seen": 0}
    train = collect(train_iter, n=args.n_train, stats=stats, **kw)
    if len(train) < args.n_train:
        raise SystemExit(f"only {len(train)} usable train pairs, wanted {args.n_train}")
    manifest["train"] = {"source_split": train_split, "sha256_16": write_jsonl(out / f"dpo_{args.dataset}_train.jsonl", train),
                         **summarize(train), **stats}

    vstats = {"seen": 0}
    val = collect(iter(load_dataset(repo, split=val_split).shuffle(seed=args.seed)), n=args.n_val, stats=vstats, **kw)
    from_holdout = 0
    if len(val) < args.n_val:
        extra = collect(train_iter, n=args.n_val - len(val), stats=vstats, **kw)
        from_holdout = len(extra)
        val += extra
    if len(val) < args.n_val:
        print(f"WARNING: only {len(val)} validation pairs available")
    manifest["val"] = {"source_split": val_split, "from_train_holdout": from_holdout,
                       "sha256_16": write_jsonl(out / f"dpo_{args.dataset}_val.jsonl", val), **summarize(val), **vstats}

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
