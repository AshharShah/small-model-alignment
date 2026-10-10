"""Persona style-steering with TRL's DPOTrainer + PEFT LoRA (Task 5; TRL and PEFT are allowed here).

Tested with trl 0.25.1 + peft 0.21.2 + transformers 4.57.6 (the same transformers as the from-scratch runs).
The model is loaded here (dtype=...) rather than by TRL, so the script does not depend on how a given TRL
release loads a model from a string.

* data     : data/persona_lora/ramsay_style_preferences.json (prompt / chosen / rejected), converted to TRL's
             conversational preference format and split into train / validation with a fixed seed
* model    : HuggingFaceTB/SmolLM2-360M-Instruct, fp32 weights + fp16 autocast on a T4
* reference: with a PEFT model TRL uses the same network with the adapter disabled (no second copy in memory)
* outputs  : <run_dir>/adapter (LoRA weights), <run_dir>/metrics.json (config, log history, final evaluation)
"""
import os

# With several GPUs visible, the HF Trainer wraps the model in nn.DataParallel, which fails with this PEFT model
# (parameters on cuda:0, replica on cuda:1) and would not help a 360M model anyway. Use a single GPU.
# Must be set before torch initialises CUDA; export CUDA_VISIBLE_DEVICES yourself to override.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--val_size", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+", default=ALL_LINEAR)
    # optimisation
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_steps", type=int, default=3)
    ap.add_argument("--lr_scheduler", default="cosine")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=-1, help="smoke tests")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: use only the first N rows")
    return ap.parse_args()


def to_trl(row):
    """TRL conversational preference format: role/content message lists."""
    return {"prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}]}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if (run_dir / "metrics.json").exists():
        raise SystemExit(f"{run_dir}/metrics.json exists; use a fresh --run_dir.")
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.data_path))
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size:]
    train_ds = Dataset.from_list([to_trl(r) for r in train_rows])
    val_ds = Dataset.from_list([to_trl(r) for r in val_rows])
    print(f"train pairs: {len(train_ds)} | validation pairs: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required by TRL's processing_class

    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)  # fp32 weights, fp16 autocast
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM")
    config = DPOConfig(
        output_dir=str(run_dir / "trainer_out"),
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type=args.lr_scheduler, warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm, weight_decay=0.0,
        beta=args.beta, max_length=args.max_length,
        fp16=cuda, bf16=False,                      # T4 has no bf16; weights stay fp32 so GradScaler works
        gradient_checkpointing=False,
        disable_dropout=False,                      # keep LoRA dropout active (the base model has none)
        eval_strategy="epoch", eval_on_start=True,
        save_strategy="no", logging_steps=1, report_to="none",
        seed=args.seed, data_seed=args.seed,
    )
    trainer = DPOTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=peft_config)

    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable parameters: {n_train:,} of {n_total:,} ({100 * n_train / n_total:.2f}%)")

    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(str(run_dir / "adapter"))

    record = {"args": vars(args), "n_train": len(train_ds), "n_val": len(val_ds),
              "trainable_params": n_train, "total_params": n_total, "train_minutes": (time.time() - t0) / 60,
              "final_eval": final_eval, "log_history": trainer.state.log_history}
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
    keys = [k for k in final_eval if "rewards/accuracies" in k or "rewards/margins" in k or k.endswith("loss")]
    print("final evaluation:", {k: round(final_eval[k], 4) for k in keys})


if __name__ == "__main__":
    main()