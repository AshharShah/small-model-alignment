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
| LoRA r16 lr5e-5 | persona-40 | 40 | 40 | 0 | 253 | 280 | 11 |
