| track | source (train split / validation split) | train / validation | filtering | mean tokens |
|---|---|---|---|---|
| SFT | HuggingFaceH4/ultrachat_200k (train_sft / test_sft) | 30,000 / 1,000 | first user/assistant turn only; prompt + response + EOS <= 512 tokens (21,560 of 51,565 train rows dropped as too long) | 98.0 prompt / 214.0 response |
| HH-RLHF DPO | Anthropic/hh-rlhf (train / test) | 15,000 / 1,000 | single-turn only; prompt <= 256 and prompt + response + EOS <= 512 tokens; 34,382 multi-turn rows dropped; 127 of 49,531 train rows dropped as too long | 21.7 prompt / 76.9 chosen / 69.3 rejected; chosen longer in 53.6% |
| UltraFeedback DPO | HuggingFaceH4/ultrafeedback_binarized (train_prefs / test_prefs) | 15,000 / 1,000 (27 validation pairs from unused train rows) | single-turn only; prompt <= 256 and prompt + response + EOS <= 512 tokens; 3,687 score ties dropped; 12,124 of 31,014 train rows dropped as too long | 75.8 prompt / 171.5 chosen / 148.7 rejected; chosen longer in 54.9% |
| LoRA persona DPO | data/persona_lora/ramsay_style_preferences.json (provided) | 198 / 22 | none; seeded shuffle (seed 42), first 22 rows held out | - |
