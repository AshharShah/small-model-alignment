| evaluation set | file | prompts | composition | used for |
|---|---|---|---|---|
| AlpacaEval-300 | eval_sets/alpaca_eval_300_seed42.json | 300 | fixed random subset (seed 42) of the 805 AlpacaEval prompts: selfinstruct 95, koala 60, oasst 54, helpful_base 51, vicuna 40 | SFT vs DPO-UF (main); SFT vs DPO-HH (alignment tax) |
| HH-eval-300 | eval_sets/hh_rlhf_eval_300_seed42.json | 300 | 10 categories x 30 | SFT vs DPO-HH (main); SFT vs DPO-UF (alignment tax) |
| persona-40 | eval_sets/persona_eval_prompts.json | 40 | 4 categories x 10 | LoRA adapters vs base model (test set) |
| held-out-22 | eval_sets/persona_heldout_prompts.json | 22 | validation rows of the persona data, never trained on | LoRA adapter selection (dev set) |
