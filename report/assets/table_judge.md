| comparison (model under test vs reference) | n | win | tie | loss | unk | win rate (ties excluded) | win rate (ties = 1/2) | 95% CI (ties excluded) | sign-test p |
|---|---|---|---|---|---|---|---|---|---|
| DPO-UF vs SFT, AlpacaEval (general) | 300 | 94 | 60 | 146 | 0 | 0.392 | 0.413 | [0.332, 0.455] | 0.001 |
| DPO-HH vs SFT, HH eval (safety) | 300 | 137 | 70 | 93 | 0 | 0.596 | 0.573 | [0.531, 0.657] | 0.004 |
| DPO-HH vs SFT, AlpacaEval (alignment tax) | 300 | 102 | 93 | 105 | 0 | 0.493 | 0.495 | [0.425, 0.560] | 0.889 |
| DPO-UF vs SFT, HH eval (alignment tax) | 300 | 96 | 68 | 134 | 2 | 0.417 | 0.436 | [0.355, 0.482] | 0.015 |
| LoRA lr5e-5_rpo1 (selected) vs base, persona style, 40 required prompts | 40 | 0 | 40 | 0 | 0 | - | 0.500 | - | - |
| LoRA lr5e-5_rpo1 (selected) vs base, usefulness, 40 required prompts | 40 | 16 | 3 | 21 | 0 | 0.432 | 0.438 | [0.287, 0.591] | 0.511 |
| LoRA lr5e-5_rpo1 (selected) vs base, safety, 40 required prompts | 40 | 4 | 32 | 4 | 0 | 0.500 | 0.500 | [0.215, 0.785] | 1.000 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, persona style, 40 required prompts | 40 | 24 | 16 | 0 | 0 | 1.000 | 0.800 | [0.862, 1.000] | <0.001 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, usefulness, 40 required prompts | 40 | 11 | 2 | 27 | 0 | 0.289 | 0.300 | [0.170, 0.448] | 0.014 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, safety, 40 required prompts | 40 | 4 | 18 | 18 | 0 | 0.182 | 0.325 | [0.073, 0.385] | 0.004 |
| LoRA lr5e-5_rpo1 (selected) vs base, persona style, 22 held-out prompts | 22 | 2 | 20 | 0 | 0 | 1.000 | 0.545 | [0.342, 1.000] | 0.500 |
| LoRA lr1e-4_rpo1_e6 (strongest) vs base, persona style, 22 held-out prompts | 22 | 19 | 3 | 0 | 0 | 1.000 | 0.932 | [0.832, 1.000] | <0.001 |
| POSITIVE CONTROL: dataset persona vs neutral answers, persona style, 22 held-out prompts | 22 | 22 | 0 | 0 | 0 | 1.000 | 1.000 | [0.851, 1.000] | <0.001 |
| DPO-UF vs SFT, AlpacaEval (positions swapped) | 300 | 94 | 75 | 131 | 0 | 0.418 | 0.438 | [0.355, 0.483] | 0.016 |
| DPO-HH vs SFT, HH eval (positions swapped) | 300 | 147 | 60 | 93 | 0 | 0.613 | 0.590 | [0.550, 0.672] | <0.001 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | 300 | 97 | 65 | 138 | 0 | 0.413 | 0.432 | [0.352, 0.477] | 0.009 |
| DPO-HH vs SFT, HH eval (plain safety rubric) | 300 | 134 | 68 | 97 | 1 | 0.580 | 0.562 | [0.516, 0.642] | 0.018 |

Judge: google/gemini-3.6-flash through the Kaggle model proxy, 5 pairs per request, 5 s between attempts, up to 3 attempts per batch, then the pairs of a failed batch one at a time. A verdict that stayed invalid after that is recorded as unknown, excluded from both win rates and logged in a failure file (see the reliability table). Win rate (ties excluded) = wins / (wins + losses); win rate (ties = 1/2) = (wins + ties / 2) / (wins + ties + losses). Pairs with identical outputs count as ties without being sent to the judge.
