import argparse
import json
import statistics
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--adapter_path", default=None, help="LoRA adapter directory (omit for the base model)")
    ap.add_argument("--prompts_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--dataset", default="persona_style_eval")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_path)
    if out_path.exists():
        print(f"{out_path} already exists, nothing to do")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = json.load(open(args.prompts_path))
    if args.limit:
        prompts = prompts[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(args.device).eval()

    stop_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id:
        stop_ids.add(im_end)
    stop_ids = sorted(i for i in stop_ids if i is not None)

    texts = [tok.apply_chat_template([{"role": "user", "content": p["instruction"]}],
                                     tokenize=False, add_generation_prompt=True) for p in prompts]
    results = []
    for b in range(0, len(prompts), args.batch_size):
        enc = tok(texts[b : b + args.batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(args.device)
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 repetition_penalty=args.repetition_penalty,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
        for p, ids in zip(prompts[b : b + args.batch_size], out[:, enc["input_ids"].shape[1]:]):
            ids = ids.tolist()
            cut = next((i for i, t in enumerate(ids) if t in stop_ids), None)
            hit_eos = cut is not None
            if hit_eos:
                ids = ids[:cut]
            results.append({"id": p["id"], "instruction": p["instruction"],
                            "output": tok.decode(ids, skip_special_tokens=True).strip(),
                            "generator": args.generator, "dataset": args.dataset,
                            "category": p.get("category"), "num_new_tokens": len(ids), "hit_eos": hit_eos})
        print(f"[{min(b + args.batch_size, len(prompts))}/{len(prompts)}]", flush=True)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    meta = {"generator": args.generator, "model_name": args.model_name, "adapter_path": args.adapter_path,
            "prompts_path": args.prompts_path, "num_prompts": len(results), "chat_template": True,
            "dtype": "float16", "batch_size": args.batch_size,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                         "repetition_penalty": args.repetition_penalty},
            "torch": torch.__version__, "transformers": transformers.__version__}
    out_path.with_name(out_path.stem + ".meta.json").write_text(json.dumps(meta, indent=2))
    toks = [r["num_new_tokens"] for r in results]
    print(f"saved {len(results)} -> {out_path} | hit EOS {sum(r['hit_eos'] for r in results)}/{len(results)} | "
          f"mean tokens {statistics.mean(toks):.0f} | empty {sum(not r['output'] for r in results)}")


if __name__ == "__main__":
    main()
