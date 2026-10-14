import argparse
import json
from pathlib import Path

from alignment.persona import split_persona_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/persona_lora/ramsay_style_preferences.json"
PROMPTS = ROOT / "eval_sets/persona_heldout_prompts.json"
GENS = ROOT / "results/generations"


def build(data_path=DATA, seed=42, val_size=22):
    rows = json.load(open(data_path))
    _, val = split_persona_rows(rows, seed, val_size)
    prompts, chosen, rejected = [], [], []
    for i, r in enumerate(val, start=1):
        pid = f"persona-heldout-{i:04d}"
        prompts.append({"id": pid, "instruction": r["prompt"], "dataset": "persona_heldout", "category": "heldout",
                        "expected_behavior": "In-distribution persona prompt held out from DPO training; "
                                             "the persona style should be visible."})
        for out, key, name in ((chosen, "chosen", "dataset-chosen"), (rejected, "rejected", "dataset-rejected")):
            out.append({"id": pid, "instruction": r["prompt"], "output": r[key], "generator": name,
                        "dataset": "persona_heldout", "category": "heldout", "hit_eos": True})
    return prompts, chosen, rejected


def check_no_overlap(prompts, other_path):
    other = {r["instruction"].strip().lower() for r in json.load(open(other_path))}
    return [p["instruction"] for p in prompts if p["instruction"].strip().lower() in other]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed files match instead of writing")
    args = ap.parse_args()
    prompts, chosen, rejected = build()
    targets = {PROMPTS: prompts, GENS / "persona_heldout_chosen.json": chosen, GENS / "persona_heldout_rejected.json": rejected}
    overlap = check_no_overlap(prompts, ROOT / "eval_sets/persona_eval_prompts.json")
    assert not overlap, f"held-out prompts overlap the 40 required prompts: {overlap}"
    for path, content in targets.items():
        if args.check:
            assert json.load(open(path)) == content, f"{path} differs from what this script builds"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False))
    print(("verified" if args.check else "wrote"), len(prompts), "held-out prompts;", len(targets), "files")


if __name__ == "__main__":
    main()
