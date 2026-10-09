"""From-scratch DPO training loop (no TRL, no HF Trainer).

* policy  : SFT checkpoint, fp32 weights, fp16 autocast + GradScaler, trainable, on --policy_device
* reference: frozen copy of the same SFT checkpoint on --ref_device (default fp32 weights run under the
             same fp16 autocast, so policy and reference are numerically identical at step 0)
* one preference pair per micro-batch (chosen and rejected scored in a single 2-row forward pass),
  gradient accumulation to an effective batch of 32 pairs, fixed LR, gradient clipping
* validation: DPO loss, reward accuracy (strict r_w > r_l), reward margin; reference log-probs for the
  validation set are computed once up front
* checkpoints: best by validation reward accuracy (ties broken by lower validation loss) and final
"""
import os

# Reduces CUDA memory fragmentation; must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_dpo_batch
from alignment.checkpoint import save_checkpoint
from alignment.dpo import concat_pair_batches, dpo_rewards, split_pair
from alignment.objectives import compute_dpo_loss_from_logps, sequence_logps_from_logits
from alignment.prompts import PROMPT_TEMPLATE, format_prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_path", required=True, help="SFT best checkpoint (policy init AND frozen reference)")
    ap.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=0, help="0 = fixed LR from the first step (guide default)")
    ap.add_argument("--grad_accum", type=int, default=32, help="pairs per optimizer step (per-device batch is 1 pair)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    ap.add_argument("--eval_batch_size", type=int, default=2, help="validation pairs per forward pass")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--policy_device", default="cuda:0")
    ap.add_argument("--ref_device", default="cuda:1")
    ap.add_argument("--ref_dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--grad_checkpointing", action="store_true", help="use if you hit out-of-memory")
    ap.add_argument("--train_limit", type=int, default=None, help="smoke tests: first N training pairs")
    ap.add_argument("--val_limit", type=int, default=None, help="smoke tests: first N validation pairs")
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests: stop after N optimizer steps")
    return ap.parse_args()


class PairDataset(Dataset):
    def __init__(self, path, limit=None):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.rows = rows[:limit] if limit else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return format_prompt(r["prompt"]), r["chosen"], r["rejected"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seq_logps(model, batch, device, amp):
    """Summed response-token log-probs for every row of a (concatenated) batch -> [rows], float32."""
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        logits = model(input_ids=ids, attention_mask=mask).logits
    return sequence_logps_from_logits(logits, labels)


@torch.no_grad()
def reference_logps_for(loader, ref, ref_device, ref_amp):
    """[N, 2] (chosen, rejected) reference log-probs for a non-shuffled loader, kept on the CPU."""
    out = []
    for batch in loader:
        c, r = split_pair(seq_logps(ref, batch, ref_device, ref_amp))
        out.append(torch.stack([c, r], dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def evaluate(policy, loader, ref_val, beta, device, amp):
    policy.eval()
    pc_all, pr_all, i = [], [], 0
    for batch in loader:
        pc, pr = split_pair(seq_logps(policy, batch, device, amp))
        pc_all.append(pc)
        pr_all.append(pr)
        i += pc.shape[0]
    policy.train()
    pc, pr = torch.cat(pc_all), torch.cat(pr_all)
    rc, rr = ref_val[:, 0].to(device), ref_val[:, 1].to(device)
    r_w, r_l = dpo_rewards(pc, pr, rc, rr, beta)
    return {
        "val_loss": compute_dpo_loss_from_logps(pc, pr, rc, rr, beta).item(),
        "val_reward_acc": (r_w > r_l).float().mean().item(),
        "val_reward_margin": (r_w - r_l).mean().item(),
        "val_chosen_reward": r_w.mean().item(),
        "val_rejected_reward": r_l.mean().item(),
        "val_max_abs_reward": torch.maximum(r_w.abs().max(), r_l.abs().max()).item(),
        "val_policy_chosen_logp": pc.mean().item(),
        "val_policy_rejected_logp": pr.mean().item(),
        "val_ref_chosen_logp": rc.mean().item(),
        "val_ref_rejected_logp": rr.mean().item(),
        "n_val": int(pc.shape[0]),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise SystemExit(f"{metrics_path} exists; use a fresh --run_dir (this script does not resume).")

    set_seed(args.seed)
    pdev, rdev = torch.device(args.policy_device), torch.device(args.ref_device)
    amp, ref_amp = pdev.type == "cuda", rdev.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32)
    policy.config.use_cache = False
    if args.grad_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.to(pdev).train()

    ref_dtype = torch.float32 if args.ref_dtype == "fp32" else torch.float16
    ref = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype)
    ref.config.use_cache = False
    ref.to(rdev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    train_ds = PairDataset(args.train_path, args.train_limit)
    val_ds = PairDataset(args.val_path, args.val_limit)

    def collate(items):
        batch = build_dpo_batch(tokenizer, [x[0] for x in items], [x[1] for x in items], [x[2] for x in items],
                                args.max_length)
        return concat_pair_batches(batch)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    steps_per_epoch = len(train_loader) // args.grad_accum  # a trailing partial accumulation group is dropped
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.weight_decay, fused=amp)
    warm = args.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0 if warm <= 0 else min(1.0, (s + 1) / warm))
    scaler = torch.amp.GradScaler(pdev.type, enabled=amp)

    config = {
        **vars(args), "prompt_template": PROMPT_TEMPLATE,
        "n_train": len(train_ds), "n_val": len(val_ds), "optimizer_steps": total_steps,
        "pairs_used": total_steps * args.grad_accum, "effective_batch_size": args.grad_accum,
        "precision": "policy fp32 weights + fp16 autocast + GradScaler" if amp else "fp32 (cpu test)",
        "reference": f"frozen copy of --sft_path, {args.ref_dtype} weights, eval mode, no_grad",
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("n_train", "n_val", "optimizer_steps", "pairs_used", "beta", "lr")}))

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    ref_val = reference_logps_for(val_loader, ref, rdev, ref_amp)
    print(f"reference log-probs for {len(ref_val)} validation pairs computed in {time.time() - t0:.0f}s", flush=True)

    best = (-1.0, float("inf"))  # (reward accuracy, -> lower loss better)
    best_step = None

    def validate(step):
        nonlocal best, best_step
        m = evaluate(policy, val_loader, ref_val, args.beta, pdev, amp)
        improved = step > 0 and (m["val_reward_acc"], -m["val_loss"]) > (best[0], -best[1])
        log({"type": "eval", "step": step, "is_best": improved, **m})
        print(f"[eval] step {step} | loss {m['val_loss']:.4f} | acc {m['val_reward_acc']:.3f} | "
              f"margin {m['val_reward_margin']:+.4f} | max|r| {m['val_max_abs_reward']:.2e}"
              f"{'  <- best' if improved else ''}", flush=True)
        if improved:
            best, best_step = (m["val_reward_acc"], m["val_loss"]), step
            save_checkpoint(policy, tokenizer, run_dir / "best")
        return m

    last_eval = validate(0)

    t0 = time.time()
    micro, step = 0, 0
    zero = lambda: torch.zeros((), device=pdev)
    loss_sum, acc_sum, margin_sum = zero(), zero(), zero()
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            with torch.no_grad():  # launched first: runs on the reference GPU while the policy GPU computes
                ref_logps = seq_logps(ref, batch, rdev, ref_amp)
            pc, pr = split_pair(seq_logps(policy, batch, pdev, amp))
            rc, rr = split_pair(ref_logps.to(pdev))
            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
            scaler.scale(loss / args.grad_accum).backward()

            with torch.no_grad():
                r_w, r_l = dpo_rewards(pc.detach(), pr.detach(), rc, rr, args.beta)
                loss_sum += loss.detach()
                acc_sum += (r_w > r_l).float().sum()
                margin_sum += (r_w - r_l).sum()
            micro += 1
            if micro % args.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1 or step == total_steps:
                n = float(args.grad_accum)
                elapsed = time.time() - t0
                rec = {"type": "train", "step": step, "epoch": epoch,
                       "train_loss": loss_sum.item() / n, "train_reward_acc": acc_sum.item() / n,
                       "train_reward_margin": margin_sum.item() / n,
                       "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                       "loss_scale": scaler.get_scale() if amp else 1.0, "elapsed_s": elapsed,
                       "peak_mem_policy_gb": torch.cuda.max_memory_allocated(pdev) / 1024**3 if amp else 0.0,
                       "peak_mem_ref_gb": torch.cuda.max_memory_allocated(rdev) / 1024**3 if ref_amp else 0.0}
                log(rec)
                eta = elapsed / step * (total_steps - step)
                print(f"step {step}/{total_steps} | loss {rec['train_loss']:.4f} | acc {rec['train_reward_acc']:.2f} | "
                      f"margin {rec['train_reward_margin']:+.4f} | gn {rec['grad_norm']:.1f} | "
                      f"scale {rec['loss_scale']:.0f} | mem {rec['peak_mem_policy_gb']:.1f}G/{rec['peak_mem_ref_gb']:.1f}G | "
                      f"{elapsed / 60:.1f}m, eta {eta / 60:.1f}m", flush=True)
            loss_sum.zero_(), acc_sum.zero_(), margin_sum.zero_()

            if step % args.eval_every == 0 or step == total_steps:
                last_eval = validate(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    save_checkpoint(policy, tokenizer, run_dir / "final")
    summary = {"best_val_reward_acc": best[0], "best_val_loss_at_best": best[1], "best_step": best_step,
               "final_val_reward_acc": last_eval["val_reward_acc"], "final_val_loss": last_eval["val_loss"],
               "final_val_reward_margin": last_eval["val_reward_margin"],
               "optimizer_steps": step, "train_minutes": (time.time() - t0) / 60}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", json.dumps(summary))


if __name__ == "__main__":
    main()