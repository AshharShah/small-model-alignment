| comparison | pairs | rubric | judge calls | batches of 5 | extra calls (retries, splits) | batches split after failure | failed batches | unknown verdicts | identical outputs (auto-tie) |
|---|---|---|---|---|---|---|---|---|---|
| DPO-UF vs SFT, AlpacaEval (general) | 300 | general | 60 | 59 | 1 | 0 | 0 | 0 | 5 |
| DPO-HH vs SFT, HH eval (safety) | 300 | safety | 68 | 60 | 8 | 1 | 0 | 0 | 2 |
| DPO-HH vs SFT, AlpacaEval (alignment tax) | 300 | general | 56 | 54 | 2 | 0 | 0 | 0 | 30 |
| DPO-UF vs SFT, HH eval (alignment tax) (+ retry) | 300 | safety | 81 | 61 | 20 | 2 | 3 | 2 | 0 |
| LoRA lr5e-5_rpo1 (selected) vs base, persona style, 40 required prompts | 40 | persona_style | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr5e-5_rpo1 (selected) vs base, usefulness, 40 required prompts | 40 | general | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr5e-5_rpo1 (selected) vs base, safety, 40 required prompts | 40 | persona_safety | 9 | 8 | 1 | 0 | 0 | 0 | 0 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, persona style, 40 required prompts | 40 | persona_style | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, usefulness, 40 required prompts | 40 | general | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, safety, 40 required prompts | 40 | persona_safety | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr5e-5_rpo1 (selected) vs base, persona style, 22 held-out prompts | 22 | persona_style | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, persona style, 22 held-out prompts | 22 | persona_style | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| POSITIVE CONTROL: dataset persona vs neutral answers, persona style, 22 held-out prompts | 22 | persona_style | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| DPO-UF vs SFT, AlpacaEval (positions swapped) | 300 | general | 62 | 59 | 3 | 0 | 0 | 0 | 5 |
| DPO-HH vs SFT, HH eval (positions swapped) | 300 | safety | 64 | 60 | 4 | 0 | 0 | 0 | 2 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | 300 | general_plain | 63 | 59 | 4 | 0 | 0 | 0 | 5 |
| DPO-HH vs SFT, HH eval (plain safety rubric) (+ retry) | 300 | safety_plain | 73 | 61 | 12 | 1 | 2 | 1 | 2 |

Rows marked (+ retry) include one retry run of their unknown verdicts, so call and batch counts are cumulative; the retry resolved 0 of 3 unknown verdicts.
