"""Tests for scripts/judge_pairwise_kaggle.py and scripts/summarize_judgments.py using scripted fake judges
(no network, no Kaggle library)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_pairs as bp  # noqa: E402
import judge_pairwise_kaggle as jp  # noqa: E402
import summarize_judgments as sj  # noqa: E402


def make_pairs(n=12, identical=()):
    test = [{"id": f"p{i:03d}", "instruction": f"question {i}", "output": f"TESTTEXT {i}", "generator": "g-t",
             "num_new_tokens": 50 + i, "hit_eos": True} for i in range(n)]
    ref = [{"id": f"p{i:03d}", "instruction": f"question {i}", "output": (test[i]["output"] if i in identical else f"REFTEXT {i}"),
            "generator": "g-r", "num_new_tokens": 40, "hit_eos": i % 4 != 0} for i in range(n)]
    return bp.build_pairs(test, ref, "SECRETNAME_DPO", "SECRETNAME_SFT", seed=1, prefix="t")


def scripted(winner_of=lambda pid: "A"):
    """A fake judge that answers every pair id found in the prompt, and records every prompt it was sent."""
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        ids = re.findall(r"pair_id=(\S+)", prompt)
        return json.dumps({"verdicts": [{"pair_id": i, "winner": winner_of(i), "reason": "r"} for i in ids]})
    judge.prompts = prompts
    return judge


def run(pairs, judge, tmp_path, **kw):
    sleeps = []
    out, fail = tmp_path / "judged.jsonl", tmp_path / "judged.failures.jsonl"
    stats = jp.run_judging(pairs, judge, out, fail, kw.pop("rubric", "general"), sleep_fn=sleeps.append,
                           log=lambda *_: None, **kw)
    return stats, sleeps, out, fail


# ---------------------------------------------------------------- prompt
def test_prompt_contains_only_judge_facing_fields_and_no_identity():
    pairs = make_pairs(5)
    prompt = jp.build_prompt(pairs, "general")
    assert "SECRETNAME" not in prompt and "g-t" not in prompt and "_meta" not in prompt and "test_is_a" not in prompt
    for p in pairs:
        assert f"pair_id={p['pair_id']}" in prompt and p["response_a"] in prompt and p["response_b"] in prompt
    assert prompt.count("### Comparison") == 5


def test_swap_exchanges_the_displayed_positions_and_rubrics_differ():
    p = make_pairs(2)
    normal, swapped = jp.build_prompt(p[:1], "general"), jp.build_prompt(p[:1], "general", swap=True)
    a_block = lambda s: s.split("Response A:\n<<<\n")[1].split("\n>>>")[0]
    assert a_block(normal) == p[0]["response_a"] and a_block(swapped) == p[0]["response_b"]
    assert "refusal" in jp.build_prompt(p, "safety") and "refusal" not in jp.build_prompt(p, "general")
    assert "persona" in jp.build_prompt(p, "persona_style") and "persona" not in jp.build_prompt(p, "persona_safety")
    assert "longer" in jp.build_prompt(p, "general")        # the anti-verbosity instruction is present


def test_plain_rubrics_drop_the_extra_guidance_but_keep_the_criteria_and_safety_text():
    p = make_pairs(2)
    plain, safety_plain = jp.build_prompt(p, "general_plain"), jp.build_prompt(p, "safety_plain")
    for text in (plain, safety_plain):
        assert "helpfulness" in text and "concision" in text and "TIE" in text
        assert "longer" not in text and "Repetition" not in text and "abruptly" not in text
    assert "refusal" in safety_plain and "refusal" not in plain
    assert jp.SAFETY_ADDENDUM in jp.RUBRICS["safety"] and jp.SAFETY_ADDENDUM in jp.RUBRICS["safety_plain"]


# ---------------------------------------------------------------- parsing
IDS = {"a", "b", "c"}


def test_parse_handles_fences_chatter_and_nested_braces():
    raw = 'Sure! ```json\n{"verdicts": [{"pair_id": "a", "winner": "A", "reason": "uses {braces} fine"},' \
          '{"pair_id": "b", "winner": "tie", "reason": "x"}]}\n``` hope that helps'
    got = jp.parse_verdicts(raw, IDS)
    assert got == {"a": ("A", "uses {braces} fine"), "b": ("TIE", "x")}


def test_parse_rejects_invalid_winners_unknown_ids_duplicates_and_garbage():
    raw = json.dumps({"verdicts": [{"pair_id": "a", "winner": "C"}, {"pair_id": "zzz", "winner": "A"},
                                   {"pair_id": "b", "winner": "Response B"}, {"pair_id": "b", "winner": "A"}, "junk", 7]})
    assert jp.parse_verdicts(raw, IDS) == {"b": ("B", "")}
    for bad in ("", "no json at all", '{"verdicts": "nope"}', "{unterminated", None):
        assert jp.parse_verdicts(bad, IDS) == {}


# ---------------------------------------------------------------- mapping back to identity
@pytest.mark.parametrize("test_is_a", [True, False])
@pytest.mark.parametrize("swap", [False, True])
def test_mapping_back_to_model_identity(test_is_a, swap):
    row = {"_meta": {"a_model": "T" if test_is_a else "R", "b_model": "R" if test_is_a else "T", "test_is_a": test_is_a}}
    for shown in ("A", "B"):
        original = shown if not swap else ("B" if shown == "A" else "A")        # which ORIGINAL response was picked
        expected_model = row["_meta"]["a_model"] if original == "A" else row["_meta"]["b_model"]
        model, outcome = jp.map_verdict(row, shown, swap)
        assert model == expected_model and outcome == ("test_win" if model == "T" else "ref_win")
    assert jp.map_verdict(row, "TIE", swap) == ("tie", "tie")


# ---------------------------------------------------------------- the batch job
def test_batches_of_five_with_sleep_between_calls_and_isolated_prompts(tmp_path):
    pairs = make_pairs(12)
    judge = scripted()
    stats, sleeps, out, fail = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=5.0, max_retries=3, reuse=False)
    assert [p.count("### Comparison") for p in judge.prompts] == [5, 5, 2]
    assert (stats["batches_run"], stats["failed_batches"], stats["judge_calls"]) == (3, 0, 3)
    assert sleeps == [5.0, 5.0]                                  # a pause before every call except the first
    assert all("SECRETNAME" not in p for p in judge.prompts)
    recs = jp.read_jsonl(out)
    assert len(recs) == 12 and all(r["outcome"] in ("test_win", "ref_win") for r in recs)
    assert not fail.exists()


def test_exceptions_are_retried_with_backoff_then_succeed(tmp_path):
    inner, calls = scripted(), {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 request denied")
        return inner(prompt)
    stats, sleeps, out, fail = run(make_pairs(5), flaky, tmp_path, batch_size=5, sleep_seconds=5.0, max_retries=3, reuse=False)
    assert stats["failed_batches"] == 0 and stats["judge_calls"] == 2
    assert all(s >= 5.0 for s in sleeps) and len(jp.read_jsonl(out)) == 5


def test_only_missing_verdicts_are_requested_again(tmp_path):
    pairs, prompts = make_pairs(5), []
    first = {"done": False}

    def partial(prompt):
        prompts.append(prompt)
        ids = re.findall(r"pair_id=(\S+)", prompt)
        answered = ids[:3] if not first["done"] else ids
        first["done"] = True
        return json.dumps({"verdicts": [{"pair_id": i, "winner": "B", "reason": ""} for i in answered]})
    stats, _, out, _ = run(pairs, partial, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=3, reuse=False)
    assert [p.count("### Comparison") for p in prompts] == [5, 2]
    assert stats["failed_batches"] == 0 and len(jp.read_jsonl(out)) == 5


def test_exhausted_retries_log_failure_mark_unknown_and_resume_retries_only_those(tmp_path):
    pairs = make_pairs(10)
    bad_ids = {p["pair_id"] for p in pairs[5:]}                  # the whole second batch always fails

    def judge(prompt):
        ids = re.findall(r"pair_id=(\S+)", prompt)
        if set(ids) & bad_ids:
            return "I cannot comply with that"
        return scripted()(prompt)
    stats, _, out, fail = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=3, reuse=False,
                              split_failed=False)
    assert stats["failed_batches"] == 1 and stats["judge_calls"] == 4              # 1 good call + 3 failed attempts
    failures = jp.read_jsonl(fail)
    assert len(failures) == 1 and set(failures[0]["pair_ids"]) == bad_ids and failures[0]["attempts"] == 3
    summary = jp.summarize(out, pairs, stats)
    assert summary["unknown"] == 5 and summary["unknown_rate"] == 0.5 and summary["batch_failure_rate"] == 0.5

    judge2 = scripted()                                                             # rerun with a healthy judge
    stats2, _, out2, _ = run(pairs, judge2, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=3, reuse=True,
                             split_failed=False)
    assert stats2["judge_calls"] == 1 and len(judge2.prompts) == 1 and judge2.prompts[0].count("### Comparison") == 5
    final = jp.summarize(out2, pairs, stats2)
    assert final["unknown"] == 0 and final["missing_no_record"] == 0 and final["test_win"] + final["ref_win"] == 10


def test_rerun_with_everything_judged_makes_no_calls(tmp_path):
    pairs = make_pairs(6)
    run(pairs, scripted(), tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=3, reuse=False)
    judge = scripted()
    stats, _, out, _ = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=3, reuse=True)
    assert stats["judge_calls"] == 0 and judge.prompts == [] and len(jp.read_jsonl(out)) == 6


def test_existing_output_without_reuse_is_refused(tmp_path):
    pairs = make_pairs(3)
    run(pairs, scripted(), tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False)
    with pytest.raises(SystemExit):
        run(pairs, scripted(), tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False)


def test_identical_pairs_are_auto_ties_and_not_sent_to_the_judge(tmp_path):
    pairs = make_pairs(8, identical={2, 5})
    judge = scripted()
    stats, _, out, _ = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False)
    sent = "".join(judge.prompts)
    assert pairs[2]["pair_id"] not in sent and pairs[5]["pair_id"] not in sent
    recs = {r["pair_id"]: r for r in jp.read_jsonl(out)}
    assert recs[pairs[2]["pair_id"]]["outcome"] == "tie" and recs[pairs[2]["pair_id"]]["auto"] == "identical"
    assert jp.summarize(out, pairs, stats)["auto_ties_identical"] == 2


def test_swap_run_maps_back_to_the_same_underlying_response(tmp_path):
    """A judge that always prefers whatever text contains 'TESTTEXT' must give 'test_win' with and without swap."""
    pairs = make_pairs(6)

    def prefers_test(prompt):
        verdicts = []
        for block in prompt.split("### Comparison pair_id=")[1:]:
            pid = block.split("\n")[0]
            a_text = block.split("Response A:\n<<<\n")[1].split("\n>>>")[0]
            verdicts.append({"pair_id": pid, "winner": "A" if "TESTTEXT" in a_text else "B", "reason": ""})
        return json.dumps({"verdicts": verdicts})
    for swap, sub in ((False, "n"), (True, "s")):
        d = tmp_path / sub
        d.mkdir()
        _, _, out, _ = run(pairs, prefers_test, d, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False, swap=swap)
        assert all(r["outcome"] == "test_win" for r in jp.read_jsonl(out))


def test_one_poison_pair_is_isolated_instead_of_losing_the_whole_batch(tmp_path):
    pairs = make_pairs(5)
    poison = pairs[2]["pair_id"]
    prompts = []

    def judge(prompt):                                    # the judge cannot process any request containing the poison pair
        prompts.append(prompt)
        if f"pair_id={poison}" in prompt:
            raise RuntimeError("content blocked")
        return scripted()(prompt)
    stats, sleeps, out, fail = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=5.0, max_retries=3, reuse=False)
    assert stats["failed_batches"] == 1 and stats["batches_split_after_failure"] == 1
    assert stats["judge_calls"] == 3 + 4 + 3                      # batch x3, four clean singles, poison single x3
    failures = jp.read_jsonl(fail)
    assert len(failures) == 1 and failures[0]["pair_ids"] == [poison]
    final = jp.summarize(out, pairs, stats)
    assert final["unknown"] == 1 and final["test_win"] + final["ref_win"] == 4
    assert all(s >= 5.0 for s in sleeps)
    singles = [p for p in prompts if p.count("### Comparison") == 1]
    assert len(singles) == 7 and all("SECRETNAME" not in p for p in prompts)


def test_failure_log_keeps_the_raw_reply_of_the_pair_that_failed(tmp_path):
    pairs = make_pairs(5)
    poison = pairs[1]["pair_id"]
    inner = scripted()

    def judge(prompt):
        return "BLOCKED-BY-POLICY" if f"pair_id={poison}" in prompt else inner(prompt)
    _, _, _, fail = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False)
    rec = jp.read_jsonl(fail)[0]
    assert rec["pair_ids"] == [poison] and rec["last_raw_response"] == "BLOCKED-BY-POLICY"   # not a later success
    assert any(e.startswith(f"{poison}:") for e in rec["errors"])


def test_split_can_be_disabled(tmp_path):
    pairs = make_pairs(5)
    judge = lambda prompt: "no json"
    stats, _, _, fail = run(pairs, judge, tmp_path, batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False,
                            split_failed=False)
    assert stats["judge_calls"] == 2 and stats["batches_split_after_failure"] == 0
    assert len(jp.read_jsonl(fail)) == 1


# ---------------------------------------------------------------- statistics
def test_wilson_and_sign_test_known_values():
    lo, hi = sj.wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert sj.wilson(0, 0) == (None, None)
    assert sj.sign_test_p(8, 2) == pytest.approx(0.1094, abs=1e-4)
    assert sj.sign_test_p(5, 5) == 1.0 and sj.sign_test_p(0, 0) is None


def test_analysis_groups_and_overall_counts(tmp_path):
    pairs = make_pairs(12)
    _, _, out, _ = run(pairs, scripted(lambda pid: "TIE" if int(pid.split("-")[1]) % 4 == 0 else "A"), tmp_path,
                       batch_size=5, sleep_seconds=0.0, max_retries=2, reuse=False)
    recs = sj.load_last(out)
    result = sj.analyze(recs)
    o = result["overall"]
    assert o["n"] == 12 and o["wins"] + o["losses"] + o["ties"] + o["unknown"] == 12 and o["ties"] == 3
    assert sum(g["n"] for g in result["by_test_position"].values()) == 12
    assert set(result["by_length"]) <= {"test_longer(>1.2x)", "similar_length", "test_shorter(<0.8x)"}
    assert sum(g["n"] for g in result["by_truncation"].values()) == 12