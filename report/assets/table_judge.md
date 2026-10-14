| comparison (model under test vs reference) | n | win | tie | loss | unk | win rate (no ties) | 95% CI | sign-test p |
|---|---|---|---|---|---|---|---|---|
| DPO-UF vs SFT, AlpacaEval (general) | 300 | 94 | 60 | 146 | 0 | 0.392 | [0.332, 0.455] | 0.001 |
| DPO-HH vs SFT, HH eval (safety) | 300 | 137 | 70 | 93 | 0 | 0.596 | [0.531, 0.657] | 0.004 |
| DPO-HH vs SFT, AlpacaEval (alignment tax) | 300 | 102 | 93 | 105 | 0 | 0.493 | [0.425, 0.560] | 0.889 |
| DPO-UF vs SFT, HH eval (alignment tax) | 300 | 96 | 68 | 134 | 2 | 0.417 | [0.355, 0.482] | 0.015 |
| LoRA vs base, persona style | 40 | 0 | 40 | 0 | 0 | - | - | - |
| LoRA vs base, usefulness | 40 | 15 | 10 | 15 | 0 | 0.500 | [0.332, 0.668] | 1.000 |
| LoRA vs base, safety | 40 | 3 | 32 | 5 | 0 | 0.375 | [0.137, 0.694] | 0.727 |
| DPO-UF vs SFT, AlpacaEval (positions swapped) | 300 | 94 | 75 | 131 | 0 | 0.418 | [0.355, 0.483] | 0.016 |
| DPO-HH vs SFT, HH eval (positions swapped) | 300 | 147 | 60 | 93 | 0 | 0.613 | [0.550, 0.672] | <0.001 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | 300 | 97 | 65 | 138 | 0 | 0.413 | [0.352, 0.477] | 0.009 |
| DPO-HH vs SFT, HH eval (plain safety rubric) | 300 | 134 | 68 | 97 | 1 | 0.580 | [0.516, 0.642] | 0.018 |
| CONTROL: over-trained LoRA vs base, style | 40 | 2 | 38 | 0 | 0 | 1.000 | [0.342, 1.000] | 0.500 |
