| stage | setting | guide | used | note | kind |
|---|---|---|---|---|---|
| SFT | optimizer steps | 30,000 / 32 = 937.5 (implied) | 937 (29,984 examples) | the incomplete last accumulation group is dropped | implementation |
| SFT | training examples | 30,000 single-turn examples | 30,000 that fit in 512 tokens | 21,560 longer rows skipped so every kept example keeps its EOS label | data |
| SFT | AdamW betas | not specified | (0.9, 0.95) | not tuned | design |
| HH / UF DPO | optimizer steps | 15,000 / 32 = 468.75 (implied) | 468 | the incomplete last accumulation group is dropped | implementation |
| HH / UF DPO | reference model precision | may be loaded in fp16 | fp32 weights under fp16 autocast, on the second GPU | the second T4 had room, and policy and reference then start numerically identical | resource |
| HH / UF DPO | optimizer | RMSprop or AdamW | AdamW, betas (0.9, 0.95), weight decay 0 | not tuned | design |
| HH / UF DPO | checkpoint selection | best validation reward accuracy | best validation reward accuracy (ties: lower loss): HH step 400, UF step 468 | validated every 100 steps and at the end | as guide |
| UltraFeedback DPO | validation pairs | 1,000 | 1,000 (27 from train rows not used for training) | the test split has too few usable single-turn pairs | data |
| LoRA persona | loss | TRL DPOTrainer, plain DPO | DPO + NLL on the chosen answer (rpo_alpha 1) | plain DPO gave no persona (0/40/0 style W/T/L on the 40 required prompts); see the LoRA sweep table | performance |
| LoRA persona | learning rate / epochs / rank | student choice | 5e-05 / 3 / r=16 | picked by a pre-stated rule from the 6 adapters scored on the held-out set (12 runs trained in total) | performance |
| Generation | decoding | greedy; repetition controls optional | greedy, 512 new tokens, repetition penalty 1.0 | same settings for SFT and DPO; many outputs still loop, see the generation table | as guide |

Unchanged from the guide (checked against the saved configs):

- SFT: max length 512, train examples 30000, validation examples 1000, per-device batch 2, gradient accumulation 16, effective batch 32, learning rate 2e-05, warmup 0.03, weight decay 0.1, gradient clipping 1, epochs 1
- HH-RLHF DPO: train pairs 15000, validation pairs 1000, max length 512, gradient accumulation (1 pair per device) 32, effective batch 32, learning rate (fixed) 1e-06, beta 0.1, gradient clipping 1, epochs 1, warmup steps (fixed LR) 0, max prompt length 256
- UltraFeedback DPO: train pairs 15000, validation pairs 1000, max length 512, gradient accumulation (1 pair per device) 32, effective batch 32, learning rate (fixed) 1e-06, beta 0.1, gradient clipping 1, epochs 1, warmup steps (fixed LR) 0, max prompt length 256
