|  | SFT | HH-RLHF DPO | UltraFeedback DPO | LoRA-DPO (TRL) |
|---|---|---|---|---|
| base / init | Qwen2.5-0.5B | SFT best | SFT best | SmolLM2-360M-Instruct |
| trainable parameters | full fine-tuning | full fine-tuning | full fine-tuning | LoRA r=16, alpha=32, dropout=0.05, all linear (8,683,520 params) |
| optimizer | AdamW b=(0.9,0.95), wd=0.1 | AdamW b=(0.9,0.95), wd=0.0 | AdamW b=(0.9,0.95), wd=0.0 | AdamW (HF Trainer defaults), wd=0 |
| peak LR / schedule | 2e-05 / cosine, 28 warmup steps | 1e-06 / fixed | 1e-06 / fixed | 5e-05 / cosine, 3 warmup steps |
| grad clip | 1.0 | 1.0 | 1.0 | 1.0 |
| batch (device x accum = effective) | 2 x 16 = 32 | 1 pair x 32 = 32 | 1 pair x 32 = 32 | 4 x 4 = 16 |
| precision | fp32 weights, fp16 autocast | policy fp32 + fp16 autocast; reference fp32 (autocast) | policy fp32 + fp16 autocast; reference fp32 (autocast) | fp32 weights, fp16 autocast |
| max length / epochs | 512 / 1 | 512 / 1 | 512 / 1 | 512 / 3.0 |
| optimizer steps | 937 | 468 | 468 | see runs/*_metrics.json |
| train / val examples | 30000 / 1000 | 15000 / 1000 | 15000 / 1000 | 198 / 22 |
| DPO beta / reference | - | 0.1 / frozen SFT best | 0.1 / frozen SFT best | 0.1 / base model, adapter disabled |
| seed | 42 | 42 | 42 | 42 |
