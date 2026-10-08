"""Generate responses on a fixed prompt file (greedy by default).

Writes a JSON list (id, instruction, output, generator, dataset + extras) and a
<output>.meta.json with the exact settings. Resumable via <output>.partial.jsonl
(delete the partial file if you change any decoding setting).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.prompts import PROMPT_TEMPLATE, format_prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default=None, help="defaults to model_path")
    ap.add_argument("--prompts_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--dataset", default="alpaca_eval")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_path)
    if out_path.exists():
        print(f"{out_path} already exists, nothing to do")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.parent / (out_path.name + ".partial.jsonl")

    prompts = json.load(open(args.prompts_path))
    if args.limit:
        prompts = prompts[: args.limit]

    done = {}
    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
        print(f"resuming: {len(done)} prompts already done")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    tok.padding_side = "left"  # required for batched decoder-only generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(args.device).eval()

    texts = {p["id"]: format_prompt(p["instruction"]) for p in prompts}
    plen = {pid: len(tok(t).input_ids) for pid, t in texts.items()}
    print(f"prompt tokens: max={max(plen.values())}, prompts > 256 tokens: {sum(v > 256 for v in plen.values())}")

    # longest first; ties broken by id, so batch composition depends only on the prompt set
    todo = sorted((p for p in prompts if p["id"] not in done), key=lambda p: (-plen[p["id"]], p["id"]))

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    t0 = time.time()
    with open(partial_path, "a") as pf:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]
            enc = tok([texts[p["id"]] for p in batch], return_tensors="pt", padding=True).to(args.device)
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            for p, ids in zip(batch, new_tokens):
                ids = ids.tolist()
                hit_eos = tok.eos_token_id in ids  # pad == eos for Qwen, so cut at the first one
                if hit_eos:
                    ids = ids[: ids.index(tok.eos_token_id)]
                rec = {
                    "id": p["id"],
                    "instruction": p["instruction"],
                    "output": tok.decode(ids, skip_special_tokens=True).strip(),
                    "generator": args.generator,
                    "dataset": args.dataset,
                    "source_index": p.get("source_index"),
                    "num_new_tokens": len(ids),
                    "hit_eos": hit_eos,
                }
                done[p["id"]] = rec
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            print(f"[{min(b + args.batch_size, len(todo))}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)

    results = [done[p["id"]] for p in prompts]  # original prompt order
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    meta = {
        "generator": args.generator,
        "model_path": args.model_path,
        "prompts_path": args.prompts_path,
        "num_prompts": len(results),
        "prompt_template": PROMPT_TEMPLATE,
        "dtype": "float16",
        "batch_size": args.batch_size,
        "decoding": {k: v for k, v in gen_kwargs.items() if k not in ("pad_token_id", "eos_token_id")},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_generation_config": model.generation_config.to_dict(),
    }
    with open(out_path.with_name(out_path.stem + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    toks = [r["num_new_tokens"] for r in results]
    print(f"\nsaved {len(results)} -> {out_path}")
    print(f"hit EOS: {sum(r['hit_eos'] for r in results)}/{len(results)} | "
          f"mean tokens: {statistics.mean(toks):.0f} | empty outputs: {sum(not r['output'] for r in results)}")


if __name__ == "__main__":
    main()