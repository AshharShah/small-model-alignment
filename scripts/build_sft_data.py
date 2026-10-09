"""Build single-turn SFT train/validation files from UltraChat-200k.

Each conversation is reduced to its first (user, assistant) exchange. Examples are
shuffled with a fixed seed and kept only if the formatted prompt + response + EOS
fits in --max_length tokens, so every training example keeps its EOS label.
Output: <out_dir>/sft_train.jsonl, sft_val.jsonl, sft_data_manifest.json
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from alignment.prompts import format_prompt


def first_exchange(example):
    msgs = example["messages"]
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return None
    instruction, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
    if not instruction or not response:
        return None
    return instruction, response


def collect(rows, tokenizer, n, max_length, stats):
    """rows: iterable of dataset examples, already shuffled."""
    kept = []
    for ex in rows:
        stats["seen"] += 1
        pair = first_exchange(ex)
        if pair is None:
            stats["skipped_format"] += 1
            continue
        instruction, response = pair
        n_prompt = len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False))
        n_resp = len(tokenizer.encode(response, add_special_tokens=False)) + 1  # + EOS
        if n_prompt + n_resp > max_length:
            stats["skipped_too_long"] += 1
            continue
        kept.append({"id": ex.get("prompt_id"), "instruction": instruction, "response": response,
                     "prompt_tokens": n_prompt, "response_tokens": n_resp})
        if len(kept) == n:
            break
    return kept


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/data")
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "sft_data_manifest.json").exists():
        print("manifest exists, nothing to do:", out)
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = {"dataset": "HuggingFaceH4/ultrachat_200k", "seed": args.seed, "max_length": args.max_length,
                "tokenizer": args.tokenizer, "filter": "first user/assistant exchange; prompt+response+EOS <= max_length"}

    for name, split, n in (("train", "train_sft", args.n_train), ("val", "test_sft", args.n_val)):
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=args.seed)
        stats = {"seen": 0, "skipped_format": 0, "skipped_too_long": 0}
        rows = collect(ds, tok, n, args.max_length, stats)
        if len(rows) < n:
            raise SystemExit(f"only found {len(rows)} usable {name} examples, wanted {n}")
        digest = write_jsonl(out / f"sft_{name}.jsonl", rows)
        manifest[name] = {
            "source_split": split, "n": len(rows), "sha256_16": digest, **stats,
            "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
            "mean_response_tokens": round(statistics.mean(r["response_tokens"] for r in rows), 1),
        }
        print(name, manifest[name])

    (out / "sft_data_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()