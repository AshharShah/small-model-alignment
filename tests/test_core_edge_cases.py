from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment.objectives import sequence_logps_from_logits
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None
    eos_token = None

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[c] for c in text]


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


def _manual_logps(model, tokenizer, prompt, response, max_length):
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (p + r)[:max_length]
    labels = ([-100] * len(p) + r)[:max_length]
    logits = model(input_ids=torch.tensor([ids])).logits[0]
    logp = F.log_softmax(logits.float(), dim=-1)
    total = torch.tensor(0.0)
    for t in range(1, len(ids)):
        if labels[t] != -100:
            total = total + logp[t - 1, labels[t]]
    return total


def test_eos_survives_when_it_exactly_fits_and_is_dropped_when_it_does_not():
    tok = ToyTokenizer()
    prompt, response = "Q: ", "abc"
    full = len(tok.encode(prompt)) + len(tok.encode(response)) + 1

    fits = adapter.build_sft_batch(tok, [prompt], [response], max_length=full)
    assert fits["input_ids"].shape[1] == full
    assert fits["labels"][0, -1].item() == tok.eos_token_id

    cut = adapter.build_sft_batch(tok, [prompt], [response], max_length=full - 1)
    assert cut["input_ids"].shape[1] == full - 1
    assert cut["labels"][0, -1].item() == tok.encode("c")[0]
    assert not (cut["labels"] == tok.eos_token_id).any()


def test_prompt_longer_than_max_length_gives_no_response_tokens_and_zero_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["a" * 30], ["resp"], max_length=10)
    assert batch["input_ids"].shape == (1, 10)
    assert batch["labels"].eq(-100).all()

    loss = adapter.compute_sft_loss(model, tok, ["a" * 30], ["resp"], max_length=10)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert model.logit_bias.grad is not None and model.logit_bias.grad.abs().sum() == 0

    logps = adapter.compute_response_logprobs(model, tok, ["a" * 30], ["resp"], max_length=10)
    assert logps.tolist() == [0.0]


def test_max_length_one_is_handled():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["a"], max_length=1)
    assert batch["input_ids"].shape == (1, 1)
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    loss = adapter.compute_sft_loss(model, tok, ["Q: "], ["a"], max_length=1)
    assert torch.isfinite(loss)


def test_mixed_lengths_are_right_padded_and_order_is_preserved():
    tok = ToyTokenizer()
    prompts = ["P1: ", "a much longer prompt: ", "x"]
    responses = ["short", "r", "a medium response"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=64)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        expected = tok.encode(p) + tok.encode(r) + [tok.eos_token_id]
        n = len(expected)
        assert batch["input_ids"][i, :n].tolist() == expected
        assert batch["attention_mask"][i, :n].eq(1).all()
        assert batch["attention_mask"][i, n:].eq(0).all()
        assert batch["input_ids"][i, n:].eq(tok.pad_token_id).all()
        assert batch["labels"][i, n:].eq(-100).all()


def test_empty_prompt_row_first_token_is_labelled_but_never_scored():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    batch = adapter.build_sft_batch(tok, [""], ["ab"], max_length=16)
    assert batch["labels"][0, 0].item() == tok.encode("a")[0]
    logps = adapter.compute_response_logprobs(model, tok, [""], ["ab"], max_length=16)
    expected = _manual_logps(model, tok, "", "ab", 16)
    torch.testing.assert_close(logps[0], expected)


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=16)
    assert batch["input_ids"][0].tolist() == tok.encode("Q: ") + tok.encode("ab")
    assert batch["labels"][0].ne(-100).sum().item() == 2


def test_tokenizer_without_pad_falls_back_to_eos_for_padding_only():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Qqqqqq: "], ["a", "b"], max_length=16)
    pad_positions = batch["attention_mask"].eq(0)
    assert pad_positions.any()
    assert batch["input_ids"][pad_positions].eq(tok.eos_token_id).all()
    assert batch["labels"][pad_positions].eq(-100).all()


def test_mismatched_lengths_raise():
    tok = ToyTokenizer()
    with pytest.raises(ValueError):
        adapter.build_sft_batch(tok, ["a", "b"], ["x"], max_length=8)
    with pytest.raises(ValueError):
        adapter.build_dpo_batch(tok, ["a"], ["x", "y"], ["z"], max_length=8)


def test_logprobs_match_independent_manual_computation():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.3)
    prompts = ["Say: ", "Q: ", "A longer prompt: "]
    responses = ["hello there", "no", "yes yes yes"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=40)
    for i in range(3):
        torch.testing.assert_close(logps[i], _manual_logps(model, tok, prompts[i], responses[i], 40))


def test_logprob_of_a_row_is_identical_alone_and_in_a_padded_batch():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts = ["Q: ", "a very long prompt indeed: "]
    responses = ["ok", "a rather long response here"]
    together = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=64)
    for i in range(2):
        alone = adapter.compute_response_logprobs(model, tok, [prompts[i]], [responses[i]], max_length=64)
        torch.testing.assert_close(together[i], alone[0])


def test_swapping_rows_swaps_outputs():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    a = adapter.compute_response_logprobs(model, tok, ["Q: ", "Z: "], ["abc", "z"], max_length=32)
    b = adapter.compute_response_logprobs(model, tok, ["Z: ", "Q: "], ["z", "abc"], max_length=32)
    torch.testing.assert_close(a, b.flip(0))


def test_truncated_response_is_scored_only_up_to_the_cut():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    full = adapter.compute_response_logprobs(model, tok, ["Q: "], ["abcdef"], max_length=64)
    cut = adapter.compute_response_logprobs(model, tok, ["Q: "], ["abcdef"], max_length=len("Q: ") + 3)
    torch.testing.assert_close(cut[0], _manual_logps(model, tok, "Q: ", "abcdef", len("Q: ") + 3))
    assert cut[0] > full[0]


def test_logps_from_logits_ignores_everything_outside_the_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 2, 2, -100, -100, -100]])
    base = sequence_logps_from_logits(logits, labels)

    noisy = logits.clone()
    noisy[0, 4:, :] += torch.randn(2, 11) * 50
    noisy[1, 2:, :] += torch.randn(4, 11) * 50
    assert torch.equal(sequence_logps_from_logits(noisy, labels), base)

    changed = logits.clone()
    changed[0, 3, 5] += 3.0
    assert sequence_logps_from_logits(changed, labels)[0] != base[0]


def test_half_precision_logits_are_scored_in_float32():
    logits = torch.randn(1, 5, 9).half()
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    out = sequence_logps_from_logits(logits, labels)
    assert out.dtype == torch.float32


def test_sft_loss_is_token_mean_and_matches_manual():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.6)
    prompts, responses = ["Q: ", "Qq: "], ["ab", "abcde"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=32)
    nll = -sum(_manual_logps(model, tok, p, r, 32) for p, r in zip(prompts, responses))
    n_tokens = (2 + 1) + (5 + 1)
    torch.testing.assert_close(loss, nll / n_tokens)


def test_dpo_from_logps_matches_formula_and_log_two_at_equal_policy_and_reference():
    pc, pr = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -18.0])
    rc, rr = torch.tensor([-11.0, -19.0]), torch.tensor([-11.5, -18.5])
    beta = 0.1
    logits = beta * ((pc - pr) - (rc - rr))
    expected = -torch.log(torch.sigmoid(logits)).mean()
    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)
    torch.testing.assert_close(
        adapter.compute_dpo_loss_from_logps(pc, pr, pc, pr, beta), torch.log(torch.tensor(2.0))
    )


def test_dpo_loss_is_stable_for_extreme_margins():
    big = torch.tensor([-1000.0])
    loss = adapter.compute_dpo_loss_from_logps(big, big - 5000, big, big, beta=1.0)
    assert torch.isfinite(loss) and loss.item() < 1e-6
    loss = adapter.compute_dpo_loss_from_logps(big - 5000, big, big, big, beta=1.0)
    assert torch.isfinite(loss) and math.isclose(loss.item(), 5000.0, rel_tol=1e-4)


def test_dpo_reference_logps_receive_no_gradient():
    pc = torch.tensor([-3.0], requires_grad=True)
    pr = torch.tensor([-4.0], requires_grad=True)
    rc = torch.tensor([-3.5], requires_grad=True)
    rr = torch.tensor([-3.7], requires_grad=True)
    adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta=0.1).backward()
    assert pc.grad is not None and pr.grad is not None
    assert rc.grad is None and rr.grad is None


def test_dpo_loss_with_identical_models_is_log_two_and_reference_has_no_grad():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    loss = adapter.compute_dpo_loss(
        policy, reference, tok, ["Q: ", "R: "], ["good", "yes"], ["bad", "no"], beta=0.1, max_length=32
    )
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    loss.backward()
    assert reference.logit_bias.grad is None


def test_dpo_training_steps_increase_the_chosen_vs_rejected_margin():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    for p in reference.parameters():
        p.requires_grad_(False)
    prompts, chosen, rejected = ["Q: "] * 4, ["good"] * 4, ["bad"] * 4
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    losses = []
    for _ in range(25):
        loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.5, 32)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.05
    pc = adapter.compute_response_logprobs(policy, tok, prompts, chosen, 32)
    pr = adapter.compute_response_logprobs(policy, tok, prompts, rejected, 32)
    assert (pc - pr).mean() > 0
    assert torch.equal(reference.logit_bias, torch.zeros_like(reference.logit_bias))


def test_dpo_batch_keeps_rows_aligned_between_chosen_and_rejected():
    tok = ToyTokenizer()
    prompts = ["first: ", "second prompt: "]
    batch = adapter.build_dpo_batch(tok, prompts, ["aa", "bb"], ["cc", "dd"], max_length=32)
    for side, resp in (("chosen", ["aa", "bb"]), ("rejected", ["cc", "dd"])):
        for i in range(2):
            expected = tok.encode(prompts[i]) + tok.encode(resp[i]) + [tok.eos_token_id]
            assert batch[side]["input_ids"][i, : len(expected)].tolist() == expected


def test_dpo_loss_does_not_change_train_or_eval_mode():
    tok = ToyTokenizer()
    policy, reference = ToyCausalLM(tok.vocab_size), ToyCausalLM(tok.vocab_size)
    policy.train()
    reference.eval()
    adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["a"], ["b"], 0.1, 16)
    assert policy.training and not reference.training
