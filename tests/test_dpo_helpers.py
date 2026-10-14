from __future__ import annotations

import sys
from pathlib import Path

import torch

import submission_adapter as adapter
from alignment.batching import build_dpo_batch
from alignment.dpo import concat_pair_batches, dpo_rewards, split_pair
from alignment.objectives import batch_response_logps, compute_dpo_loss_from_logps, compute_response_logprobs
from tests.toy_lm import ToyCausalLM, ToyTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_dpo_data as bdd


def test_concat_pair_logps_match_separate_scoring_and_keep_order():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts, chosen, rejected = ["Q: ", "A longer prompt: "], ["yes", "a longer chosen answer"], ["no way", "x"]
    pair = build_dpo_batch(tok, prompts, chosen, rejected, max_length=48)
    both = batch_response_logps(model, concat_pair_batches(pair))
    c, r = split_pair(both)
    torch.testing.assert_close(c, compute_response_logprobs(model, tok, prompts, chosen, 48))
    torch.testing.assert_close(r, compute_response_logprobs(model, tok, prompts, rejected, 48))


def test_concat_pads_the_shorter_side_with_masked_positions():
    tok = ToyTokenizer()
    pair = build_dpo_batch(tok, ["Q: "], ["a"], ["a much longer rejected response"], max_length=64)
    cat = concat_pair_batches(pair)
    assert cat["input_ids"].shape[0] == 2
    width = cat["input_ids"].shape[1]
    assert cat["input_ids"].shape == cat["attention_mask"].shape == cat["labels"].shape
    n_chosen = pair["chosen"]["input_ids"].shape[1]
    assert width > n_chosen
    assert cat["attention_mask"][0, n_chosen:].eq(0).all() and cat["labels"][0, n_chosen:].eq(-100).all()


def test_rewards_are_zero_for_identical_models_and_accuracy_is_strict():
    pc, pr = torch.tensor([-5.0, -7.0]), torch.tensor([-6.0, -8.0])
    r_w, r_l = dpo_rewards(pc, pr, pc, pr, beta=0.1)
    assert torch.equal(r_w, torch.zeros(2)) and torch.equal(r_l, torch.zeros(2))
    assert (r_w > r_l).float().mean().item() == 0.0
    assert abs(compute_dpo_loss_from_logps(pc, pr, pc, pr, 0.1).item() - 0.6931471805599453) < 1e-6


def test_rewards_scale_with_beta_and_match_the_loss_argument():
    pc, pr = torch.tensor([-4.0]), torch.tensor([-9.0])
    rc, rr = torch.tensor([-5.0]), torch.tensor([-8.0])
    r_w, r_l = dpo_rewards(pc, pr, rc, rr, beta=0.5)
    assert r_w.item() == 0.5 and r_l.item() == -0.5
    expected = -torch.nn.functional.logsigmoid(r_w - r_l).mean()
    torch.testing.assert_close(compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.5), expected)


def _hh(chosen, rejected):
    return {"chosen": chosen, "rejected": rejected}


def test_hh_single_turn_is_parsed_and_multi_turn_is_rejected():
    ok = bdd.parse_hh(_hh("\n\nHuman: How are you?\n\nAssistant: Fine, thanks.",
                          "\n\nHuman: How are you?\n\nAssistant: Go away."))
    assert ok == ("How are you?", "Fine, thanks.", "Go away.")
    multi = "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: And?\n\nAssistant: Yes."
    assert bdd.parse_hh(_hh(multi, multi)) == "not_single_turn"
    assert bdd.parse_hh(_hh("\n\nHuman: Hi", "\n\nHuman: Hi")) == "not_single_turn"


def test_hh_prompt_mismatch_and_bad_order_are_rejected():
    assert bdd.parse_hh(_hh("\n\nHuman: A\n\nAssistant: x", "\n\nHuman: B\n\nAssistant: y")) == "prompt_mismatch"
    assert bdd.parse_hh(_hh("\n\nAssistant: x\n\nHuman: A", "\n\nAssistant: x\n\nHuman: A")) == "bad_format"


def _uf(prompt="Say hi", chosen="Hello there", rejected="no", sc=8.0, sr=5.0, extra=False):
    msgs = lambda a: [{"role": "user", "content": prompt}, {"role": "assistant", "content": a}]
    c = msgs(chosen) + ([{"role": "user", "content": "more"}] if extra else [])
    return {"chosen": c, "rejected": msgs(rejected), "score_chosen": sc, "score_rejected": sr}


def test_uf_parse_keeps_valid_pairs_and_drops_ties_reversals_and_multi_turn():
    assert bdd.parse_uf(_uf()) == ("Say hi", "Hello there", "no")
    assert bdd.parse_uf(_uf(sc=8.5, sr=8.5)) == "tie_or_reversed_score"
    assert bdd.parse_uf(_uf(sc=3.0, sr=7.0)) == "tie_or_reversed_score"
    assert bdd.parse_uf(_uf(extra=True)) == "not_single_turn"


class LenientTokenizer(ToyTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(c, 2) for c in text]


def test_collect_applies_length_filters_and_stops_at_n_and_can_resume():
    tok = LenientTokenizer()
    short = {"chosen": "\n\nHuman: hello\n\nAssistant: good answer", "rejected": "\n\nHuman: hello\n\nAssistant: bad one"}
    long_resp = {"chosen": "\n\nHuman: hello\n\nAssistant: " + "a" * 200, "rejected": "\n\nHuman: hello\n\nAssistant: bad"}
    long_prompt = {"chosen": "\n\nHuman: " + "b" * 200 + "\n\nAssistant: ok", "rejected": "\n\nHuman: " + "b" * 200 + "\n\nAssistant: no"}
    same = {"chosen": "\n\nHuman: hello\n\nAssistant: same", "rejected": "\n\nHuman: hello\n\nAssistant: same"}
    multi = {"chosen": "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: d",
             "rejected": "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: e"}
    rows = [multi, long_resp, same, short, long_prompt, short, short]
    stats = {"seen": 0}
    kept = bdd.collect(iter(rows), bdd.parse_hh, tok, n=10, max_length=100, max_prompt_length=60, stats=stats)
    assert len(kept) == 3
    assert stats["skipped_not_single_turn"] == 1 and stats["skipped_too_long"] == 2
    assert stats["skipped_empty_or_identical"] == 1
    for r in kept:
        assert r["prompt_tokens"] <= 60 and r["prompt_tokens"] + max(r["chosen_tokens"], r["rejected_tokens"]) <= 100

    it = iter(rows)
    s1, s2 = {"seen": 0}, {"seen": 0}
    first = bdd.collect(it, bdd.parse_hh, tok, n=1, max_length=100, max_prompt_length=60, stats=s1)
    second = bdd.collect(it, bdd.parse_hh, tok, n=1, max_length=100, max_prompt_length=60, stats=s2)
    assert len(first) == len(second) == 1 and s1["seen"] + s2["seen"] <= len(rows)
