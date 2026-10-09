"""Helpers for the DPO training loop (tensor plumbing only; the loss lives in objectives.py).

A preference pair is scored with ONE forward pass over a 2-row batch: the chosen row
first, then the rejected row. ``split_pair`` undoes that.
"""

from __future__ import annotations

import torch

from .batching import IGNORE_INDEX


def _pad_to(t: torch.Tensor, width: int, value: int) -> torch.Tensor:
    if t.shape[1] == width:
        return t
    filler = t.new_full((t.shape[0], width - t.shape[1]), value)
    return torch.cat([t, filler], dim=1)


def concat_pair_batches(batch):
    """{"chosen": b, "rejected": b} -> one batch with the chosen rows first, then the rejected rows.

    Rows are right-padded to a common width (input_ids 0, attention_mask 0, labels -100), so padding
    never contributes to a score.
    """
    chosen, rejected = batch["chosen"], batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    fill = {"input_ids": 0, "attention_mask": 0, "labels": IGNORE_INDEX}
    return {
        key: torch.cat([_pad_to(chosen[key], width, fill[key]), _pad_to(rejected[key], width, fill[key])], dim=0)
        for key in fill
    }


def split_pair(per_row: torch.Tensor):
    """[2B, ...] -> (chosen [B, ...], rejected [B, ...])."""
    n = per_row.shape[0] // 2
    return per_row[:n], per_row[n:]


def dpo_rewards(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit rewards r_w = beta*(logpi - logref)(chosen) and r_l likewise for rejected."""
    return (
        beta * (policy_chosen_logps - reference_chosen_logps),
        beta * (policy_rejected_logps - reference_rejected_logps),
    )