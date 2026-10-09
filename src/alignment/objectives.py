"""Response-only log-probabilities and the SFT / DPO objectives.

SFT loss reduction (document this in the report):
    mean over ALL unmasked response tokens in the batch,
    i.e.  -sum_i logp_i / max(1, #response tokens),
    where a token counts only if it can be predicted (labels[:, 1:] != -100).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def response_token_mask(labels: torch.Tensor) -> torch.Tensor:
    """[B, T-1] bool mask of predictable response positions (labels shifted by one)."""
    return labels[:, 1:].ne(IGNORE_INDEX)


def sequence_logps_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of response-token log-probs per row, shape [B], float32, differentiable.

    Position t of ``logits`` predicts token t+1, so logits[:, :-1] is scored against
    labels[:, 1:]. Prompt and padding positions (label -100) contribute nothing.
    Only the selected positions are upcast to float32, which saves memory with a
    151k-token vocabulary.
    """
    batch_size = labels.shape[0]
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)

    selected = logits[:, :-1, :][mask].float()  # [N, V]
    token_targets = targets[mask]  # [N]
    token_logps = selected.gather(-1, token_targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(selected, dim=-1)

    row_index = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(mask)[mask]
    zeros = torch.zeros(batch_size, dtype=token_logps.dtype, device=token_logps.device)
    return zeros.index_add(0, row_index, token_logps)


def sft_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Scalar SFT loss: token-mean negative log-likelihood over response tokens."""
    logps = sequence_logps_from_logits(logits, labels)
    count = response_token_mask(labels).sum().clamp_min(1)
    return -logps.sum() / count


def _forward_logits(model, batch):
    device = model_device(model)
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    return out.logits, batch["labels"].to(out.logits.device)


def batch_response_logps(model, batch) -> torch.Tensor:
    logits, labels = _forward_logits(model, batch)
    return sequence_logps_from_logits(logits, labels)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """One summed response-token log-prob per row, shape [B]."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return batch_response_logps(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logits, labels = _forward_logits(model, batch)
    return sft_loss_from_logits(logits, labels)


def _upcast(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Per-example beta * ((pi_w - pi_l) - (ref_w - ref_l)); reference terms are constants."""
    device = policy_chosen_logps.device
    pc, pr = _upcast(policy_chosen_logps), _upcast(policy_rejected_logps)
    rc = _upcast(reference_chosen_logps).detach().to(device)
    rr = _upcast(reference_rejected_logps).detach().to(device)
    return beta * ((pc - pr) - (rc - rr))


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Scalar batch-mean DPO loss: -log sigmoid(beta * (delta_policy - delta_reference))."""
    logits = dpo_logits(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )
    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen = batch_response_logps(policy_model, batch["chosen"])
    policy_rejected = batch_response_logps(policy_model, batch["rejected"])
    with torch.no_grad():
        reference_chosen = batch_response_logps(reference_model, batch["chosen"])
        reference_rejected = batch_response_logps(reference_model, batch["rejected"])
    return compute_dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )