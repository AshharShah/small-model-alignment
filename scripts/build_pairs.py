import argparse
import json
import random
import re
from pathlib import Path

LEAK_WORDS = ["sft", "dpo", "lora", "qwen", "smollm", "checkpoint", "fine-tuned", "finetuned"]


def build_pairs(test, ref, test_name, ref_name, prompts=None, seed=42, prefix="pair"):
    if [r["id"] for r in test] != [r["id"] for r in ref]:
        raise ValueError("the two generation files do not cover the same prompt ids in the same order")
    n = len(test)
    prompt_info = {p["id"]: p for p in (prompts or [])}
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    random.Random(seed).shuffle(flags)

    rows = []
    for i, (t, r, test_is_a) in enumerate(zip(test, ref, flags)):
        if t["instruction"] != r["instruction"]:
            raise ValueError(f"instruction mismatch for id {t['id']}")
        a, b = (t, r) if test_is_a else (r, t)
        a_name, b_name = (test_name, ref_name) if test_is_a else (ref_name, test_name)
        info = prompt_info.get(t["id"], {})
        rows.append({
            "pair_id": f"{prefix}-{i:04d}", "id": t["id"], "instruction": t["instruction"],
            "response_a": a["output"], "response_b": b["output"],
            "_meta": {
                "a_model": a_name, "b_model": b_name, "test_is_a": test_is_a,
                "a_generator": a.get("generator"), "b_generator": b.get("generator"),
                "a_tokens": a.get("num_new_tokens"), "b_tokens": b.get("num_new_tokens"),
                "a_hit_eos": a.get("hit_eos"), "b_hit_eos": b.get("hit_eos"),
                "identical": a["output"].strip() == b["output"].strip(),
                "category": info.get("category", t.get("category")),
                "expected_behavior": info.get("expected_behavior"),
                "source_dataset": info.get("source_dataset"),
            },
        })
    return rows


def summarize(rows, test_name, ref_name):
    leak = re.compile(r"\b(" + "|".join(re.escape(w) for w in LEAK_WORDS) + r")\b", re.I)
    m = [r["_meta"] for r in rows]
    return {
        "n_pairs": len(rows), "model_under_test": test_name, "reference_model": ref_name,
        "test_model_in_position_A": sum(x["test_is_a"] for x in m),
        "test_model_in_position_B": sum(not x["test_is_a"] for x in m),
        "identical_outputs": sum(x["identical"] for x in m),
        "pairs_with_any_truncated_response": sum(not (x["a_hit_eos"] and x["b_hit_eos"]) for x in m),
        "empty_responses": sum((not r["response_a"].strip()) + (not r["response_b"].strip()) for r in rows),
        "responses_mentioning_identity_words": sum(bool(leak.search(r["response_a"])) + bool(leak.search(r["response_b"]))
                                                   for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True, help="generations of the model under test (e.g. DPO)")
    ap.add_argument("--ref_path", required=True, help="generations of the reference model (e.g. SFT)")
    ap.add_argument("--test_name", required=True)
    ap.add_argument("--ref_name", required=True)
    ap.add_argument("--prompts_path", default=None, help="optional prompt file; its category/expected_behavior go to _meta")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="pair")
    args = ap.parse_args()

    test, ref = json.load(open(args.test_path)), json.load(open(args.ref_path))
    prompts = json.load(open(args.prompts_path)) if args.prompts_path else None
    rows = build_pairs(test, ref, args.test_name, args.ref_name, prompts, args.seed, args.prefix)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows, args.test_name, args.ref_name)
    summary.update({"seed": args.seed, "test_path": args.test_path, "ref_path": args.ref_path})
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
