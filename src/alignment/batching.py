from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _pad_token_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_id is None else int(pad_id)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = _pad_token_id(tokenizer)

    rows = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = list(tokenizer.encode(response, add_special_tokens=False))
        if eos_id is not None:
            response_ids.append(int(eos_id))
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
        rows.append((input_ids, labels))

    batch_size = len(rows)
    width = max((len(ids) for ids, _ in rows), default=0)
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        n = len(ids)
        if n:
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have equal length")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }
