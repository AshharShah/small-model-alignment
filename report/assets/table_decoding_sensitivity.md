Same models, prompts, batch composition and judge; only the repetition penalty changes.

| comparison | decoding | W/T/L | win rate (no ties) [95% CI] | sign-test p | unknown |
|---|---|---|---|---|---|
| DPO-UF vs SFT, AlpacaEval | repetition penalty 1 (main) | 94/60/146 | 0.392 [0.332, 0.455] | 0.001 | 0 |
| DPO-UF vs SFT, AlpacaEval | repetition penalty 1.1 | 121/57/122 | 0.498 [0.436, 0.560] | 1.000 | 0 |
| DPO-HH vs SFT, HH eval | repetition penalty 1 (main) | 137/70/93 | 0.596 [0.531, 0.657] | 0.004 | 0 |
| DPO-HH vs SFT, HH eval | repetition penalty 1.1 | 133/71/96 | 0.581 [0.516, 0.643] | 0.017 | 0 |

| model | eval set | decoding | ended with EOS | hit 512-token cap | mean tokens | repeated-phrase outputs |
|---|---|---|---|---|---|---|
| SFT | AlpacaEval-300 | repetition penalty 1 (main) | 256/300 | 44 | 201 | 123 |
| DPO-UF | AlpacaEval-300 | repetition penalty 1 (main) | 229/300 | 71 | 299 | 177 |
| SFT | AlpacaEval-300 | repetition penalty 1.1 | 289/300 | 11 | 176 | 33 |
| DPO-UF | AlpacaEval-300 | repetition penalty 1.1 | 273/300 | 27 | 262 | 63 |
| SFT | HH-eval-300 | repetition penalty 1 (main) | 255/300 | 45 | 237 | 150 |
| DPO-HH | HH-eval-300 | repetition penalty 1 (main) | 268/300 | 32 | 190 | 133 |
| SFT | HH-eval-300 | repetition penalty 1.1 | 300/300 | 0 | 160 | 14 |
| DPO-HH | HH-eval-300 | repetition penalty 1.1 | 300/300 | 0 | 140 | 18 |
