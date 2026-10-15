| comparison | length of model under test vs reference | n | win | tie | loss | win rate (no ties) | p |
|---|---|---|---|---|---|---|---|
| DPO-UF vs SFT (AlpacaEval) | similar_length | 81 | 35 | 27 | 19 | 0.648 | 0.0402 |
| DPO-UF vs SFT (AlpacaEval) | test_longer(>1.2x) | 208 | 52 | 29 | 127 | 0.290 | 0.0 |
| DPO-UF vs SFT (AlpacaEval) | test_shorter(<0.8x) | 11 | 7 | 4 | 0 | 1.000 | 0.0156 |
| DPO-HH vs SFT (HH eval) | similar_length | 115 | 41 | 41 | 33 | 0.554 | 0.416 |
| DPO-HH vs SFT (HH eval) | test_longer(>1.2x) | 52 | 17 | 5 | 30 | 0.362 | 0.0789 |
| DPO-HH vs SFT (HH eval) | test_shorter(<0.8x) | 133 | 79 | 24 | 30 | 0.725 | 0.0 |
| DPO-HH vs SFT (AlpacaEval) | similar_length | 167 | 45 | 76 | 46 | 0.494 | 1.0 |
| DPO-HH vs SFT (AlpacaEval) | test_longer(>1.2x) | 51 | 14 | 8 | 29 | 0.326 | 0.0315 |
| DPO-HH vs SFT (AlpacaEval) | test_shorter(<0.8x) | 82 | 43 | 9 | 30 | 0.589 | 0.1597 |
| DPO-UF vs SFT (HH eval) | similar_length | 97 | 28 | 37 | 31 | 0.475 | 0.7948 |
| DPO-UF vs SFT (HH eval) | test_longer(>1.2x) | 185 | 55 | 30 | 100 | 0.355 | 0.0004 |
| DPO-UF vs SFT (HH eval) | test_shorter(<0.8x) | 18 | 13 | 1 | 3 | 0.812 | 0.0213 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | similar_length | 81 | 31 | 34 | 16 | 0.660 | 0.04 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | test_longer(>1.2x) | 208 | 55 | 31 | 122 | 0.311 | 0.0 |
| DPO-UF vs SFT, AlpacaEval (plain rubric, no length guidance) | test_shorter(<0.8x) | 11 | 11 | 0 | 0 | 1.000 | 0.001 |
| DPO-HH vs SFT, HH eval (plain safety rubric) | similar_length | 115 | 38 | 43 | 34 | 0.528 | 0.7239 |
| DPO-HH vs SFT, HH eval (plain safety rubric) | test_longer(>1.2x) | 52 | 17 | 5 | 30 | 0.362 | 0.0789 |
| DPO-HH vs SFT, HH eval (plain safety rubric) | test_shorter(<0.8x) | 133 | 79 | 20 | 33 | 0.705 | 0.0 |
| DPO-UF vs SFT, AlpacaEval (repetition penalty 1.1) | similar_length | 74 | 31 | 22 | 21 | 0.596 | 0.2116 |
| DPO-UF vs SFT, AlpacaEval (repetition penalty 1.1) | test_longer(>1.2x) | 212 | 80 | 34 | 98 | 0.449 | 0.2025 |
| DPO-UF vs SFT, AlpacaEval (repetition penalty 1.1) | test_shorter(<0.8x) | 14 | 10 | 1 | 3 | 0.769 | 0.0923 |
| DPO-HH vs SFT, HH eval (repetition penalty 1.1) | similar_length | 136 | 48 | 49 | 39 | 0.552 | 0.3912 |
| DPO-HH vs SFT, HH eval (repetition penalty 1.1) | test_longer(>1.2x) | 68 | 33 | 12 | 23 | 0.589 | 0.2288 |
| DPO-HH vs SFT, HH eval (repetition penalty 1.1) | test_shorter(<0.8x) | 96 | 52 | 10 | 34 | 0.605 | 0.0662 |
