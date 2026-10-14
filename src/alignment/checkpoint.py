from __future__ import annotations

import shutil
from pathlib import Path

import torch


def save_checkpoint(model, tokenizer, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    state = {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        state.pop("lm_head.weight", None)
    use_cache_during_training = model.config.use_cache
    model.config.use_cache = True
    try:
        model.save_pretrained(tmp, state_dict=state, safe_serialization=True)
    finally:
        model.config.use_cache = use_cache_during_training
    tokenizer.save_pretrained(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
