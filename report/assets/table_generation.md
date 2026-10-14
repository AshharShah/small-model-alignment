| model | eval set | n | ended with EOS | hit token cap | mean tokens | median tokens | repeated-phrase outputs |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B base | AlpacaEval-300 | 300 | 206 | 94 | 268 | 224 | 130 |
| SFT | AlpacaEval-300 | 300 | 256 | 44 | 201 | 152 | 123 |
| DPO-UF | AlpacaEval-300 | 300 | 229 | 71 | 299 | 302 | 177 |
| DPO-HH | AlpacaEval-300 | 300 | 260 | 40 | 181 | 128 | 110 |
| SFT | HH-eval-300 | 300 | 255 | 45 | 237 | 227 | 150 |
| DPO-HH | HH-eval-300 | 300 | 268 | 32 | 190 | 165 | 133 |
| DPO-UF | HH-eval-300 | 300 | 222 | 78 | 309 | 320 | 176 |
| SmolLM2 base | persona-40 | 40 | 37 | 3 | 280 | 307 | 11 |
| LoRA lr5e-5_rpo1 (selected) | persona-40 | 40 | 35 | 5 | 255 | 272 | 13 |
| LoRA lr1e-4_rpo1_e6 (strongest) | persona-40 | 40 | 33 | 7 | 127 | 47 | 12 |
| SmolLM2 base | held-out-22 | 22 | 22 | 0 | 108 | 77 | 2 |
| LoRA lr5e-5_rpo1 (selected) | held-out-22 | 22 | 20 | 2 | 107 | 65 | 3 |
| LoRA lr1e-4_rpo1_e6 (strongest) | held-out-22 | 22 | 15 | 7 | 195 | 50 | 9 |
