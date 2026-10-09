"""From-scratch SFT training loop (no HF Trainer).

fp32 trainable weights + fp16 autocast + GradScaler, gradient accumulation,
gradient clipping, warmup + cosine LR, response-only loss, validation loss,
best/final checkpoints (fp16 safetensors), and a metrics.jsonl written as we go.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_sft_batch
from alignment.objectives import response_token_mask, sequence_logps_from_logits, sft_loss_from_logits
from alignment.prompts import PROMPT_TEMPLATE, format_prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grad_checkpointing", action="store_true", help="use if you hit out-of-memory")
    ap.add_argument("--train_limit", type=int, default=None, help="smoke tests: first N training rows")
    ap.add_argument("--val_limit", type=int, default=None, help="smoke tests: first N validation rows")
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests: stop after N optimizer steps")
    return ap.parse_args()


class SFTDataset(Dataset):
    def __init__(self, path, limit=None):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.rows = rows[:limit] if limit else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return format_prompt(r["instruction"]), r["response"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_multiplier(step, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0. ``step`` counts completed optimizer steps."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model, args, use_cuda):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), fused=use_cuda)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Exact token-mean validation NLL over all response tokens."""
    model.eval()
    nll, tokens = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        nll += -sequence_logps_from_logits(logits, labels).sum().item()
        tokens += response_token_mask(labels).sum().item()
    model.train()
    return nll / max(1, tokens), tokens


def save_checkpoint(model, tokenizer, path):
    """fp16 safetensors + tokenizer, written to a temp dir first so a crash never corrupts `path`."""
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    state = {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        state.pop("lm_head.weight", None)  # tied to the embedding; re-tied on load
    use_cache_during_training = model.config.use_cache
    model.config.use_cache = True  # training disables the KV cache; generation needs it on
    try:
        model.save_pretrained(tmp, state_dict=state, safe_serialization=True)
    finally:
        model.config.use_cache = use_cache_during_training
    tokenizer.save_pretrained(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise SystemExit(f"{metrics_path} exists; use a fresh --run_dir (this script does not resume).")

    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()

    train_ds = SFTDataset(args.train_path, args.train_limit)
    val_ds = SFTDataset(args.val_path, args.val_limit)

    def collate(items):
        return build_sft_batch(tokenizer, [p for p, _ in items], [r for _, r in items], args.max_length)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    steps_per_epoch = len(train_loader) // args.grad_accum  # a trailing partial accumulation group is dropped
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))
    used_examples = total_steps * args.grad_accum * args.batch_size

    optimizer = make_optimizer(model, args, use_cuda=use_amp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, warmup_steps, total_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        **vars(args),
        "prompt_template": PROMPT_TEMPLATE,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "optimizer_steps": total_steps, "warmup_steps": warmup_steps,
        "examples_used_per_epoch": steps_per_epoch * args.grad_accum * args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "precision": "fp32 weights, fp16 autocast, GradScaler" if use_amp else "fp32 (cpu test)",
        "sft_loss_reduction": "mean over all response tokens in the micro-batch; micro-batch losses averaged over accumulation",
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("n_train", "n_val", "optimizer_steps", "warmup_steps",
                                             "effective_batch_size")}))

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    best_val, best_step = float("inf"), None
    val_loss, val_tokens = evaluate(model, val_loader, device, use_amp)
    log({"type": "eval", "step": 0, "val_loss": val_loss, "val_ppl": math.exp(val_loss), "val_tokens": val_tokens})
    print(f"[eval] step 0 | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}", flush=True)

    t0 = time.time()
    micro, step = 0, 0
    acc_loss = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = sft_loss_from_logits(logits, labels)
            scaler.scale(loss / args.grad_accum).backward()
            acc_loss += loss.detach() / args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - t0
                rec = {
                    "type": "train", "step": step, "epoch": epoch,
                    "train_loss": acc_loss.item(),  # mean over this step's accumulation group
                    "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                    "loss_scale": scaler.get_scale() if use_amp else 1.0,
                    "elapsed_s": elapsed,
                    "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if use_amp else 0.0,
                }
                log(rec)
                eta = elapsed / step * (total_steps - step)
                print(f"step {step}/{total_steps} | loss {rec['train_loss']:.4f} | lr {rec['lr']:.2e} | "
                      f"gn {rec['grad_norm']:.2f} | scale {rec['loss_scale']:.0f} | "
                      f"mem {rec['peak_mem_gb']:.1f}G | {elapsed/60:.1f}m, eta {eta/60:.1f}m", flush=True)
            acc_loss.zero_()

            if step % args.eval_every == 0 or step == total_steps:
                val_loss, val_tokens = evaluate(model, val_loader, device, use_amp)
                improved = val_loss < best_val
                log({"type": "eval", "step": step, "val_loss": val_loss, "val_ppl": math.exp(val_loss),
                     "val_tokens": val_tokens, "is_best": improved})
                print(f"[eval] step {step} | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}"
                      f"{'  <- best' if improved else ''}", flush=True)
                if improved:
                    best_val, best_step = val_loss, step
                    save_checkpoint(model, tokenizer, run_dir / "best")
            if step >= total_steps:
                done = True
                break
        if done:
            break

    save_checkpoint(model, tokenizer, run_dir / "final")
    summary = {"best_val_loss": best_val, "best_step": best_step, "final_val_loss": val_loss,
               "optimizer_steps": step, "train_minutes": (time.time() - t0) / 60}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", json.dumps(summary))


if __name__ == "__main__":
    main()