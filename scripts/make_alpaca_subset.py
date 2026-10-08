"""Create the fixed AlpacaEval evaluation subset (default: 300 prompts, seed 42)."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_sets/alpaca_eval_300_seed42.json")
    args = ap.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    data = json.load(open(path))
    idx = sorted(random.Random(args.seed).sample(range(len(data)), args.n))

    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{k:04d}",
            "source_index": i,
            "instruction": data[i]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": data[i]["dataset"],
        }
        for k, i in enumerate(idx)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(subset)} prompts from {len(data)} -> {out}")
    print("sub-sources:", dict(Counter(r["source_dataset"] for r in subset)))
    print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()