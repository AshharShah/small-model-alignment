from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_pairs as bp
import judge_pairwise_kaggle as jp
import retry_unknown_judgments as rj


def make_pairs(n):
    test = [{"id": f"p{i:03d}", "instruction": f"q{i}", "output": f"TEST {i}", "generator": "t", "num_new_tokens": 5, "hit_eos": True} for i in range(n)]
    ref = [{**t, "output": f"REF {i}", "generator": "r"} for i, t in enumerate(test)]
    return bp.build_pairs(test, ref, "dpo", "sft", seed=1, prefix="t")


def good_judge(sent, calls=None):
    def judge(prompt):
        ids = re.findall(r"pair_id=(\S+)", prompt)
        sent.extend(ids)
        if calls is not None:
            calls.append(1)
        return json.dumps({"verdicts": [{"pair_id": i, "winner": "A", "reason": "r"} for i in ids]})
    return judge


def blocked_judge(bad):
    def judge(prompt):
        ids = re.findall(r"pair_id=(\S+)", prompt)
        if set(ids) & bad:
            return ""
        return good_judge([])(prompt)
    return judge


def first_run(tmp_path, pairs, bad):
    pairs_path, judged = tmp_path / "pairs.jsonl", tmp_path / "judged.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    stats = jp.run_judging(pairs, blocked_judge(bad), judged, str(judged) + ".failures.jsonl", "general", 5, 0.0, 3, False,
                           sleep_fn=lambda s: None, log=lambda *_: None)
    summary = jp.summarize(judged, pairs, stats)
    summary.update({"rubric": "general", "judge_model_used": "fake"})
    Path(str(judged) + ".summary.json").write_text(json.dumps(summary))
    return pairs_path, judged, summary


def test_retry_only_resends_unknown_pairs_and_keeps_the_original_run_statistics(tmp_path):
    pairs = make_pairs(10)
    bad = {pairs[2]["pair_id"], pairs[7]["pair_id"]}
    pairs_path, judged, before = first_run(tmp_path, pairs, bad)
    assert before["unknown"] == 2
    sent, calls = [], []
    merged = rj.retry(pairs_path, judged, "general", good_judge(sent, calls), sleep_seconds=0.0, sleep_fn=lambda s: None, log=lambda *_: None)
    assert set(sent) == bad and len(sent) == 2
    assert merged["unknown"] == 0 and merged["test_win"] + merged["ref_win"] + merged["tie"] == 10
    assert len(calls) == 1
    assert merged["judge_calls"] == before["judge_calls"] + len(calls)
    assert merged["batches_run"] == before["batches_run"] + 1
    assert merged["rubric"] == "general" and merged["judge_model_used"] == "fake"
    assert merged["retry_runs"][0]["unknown_before"] == 2 and merged["retry_runs"][0]["unknown_after"] == 0
    assert json.loads(Path(str(judged) + ".summary.json").read_text()) == merged


def test_retry_with_nothing_unknown_makes_no_calls(tmp_path):
    pairs = make_pairs(6)
    pairs_path, judged, before = first_run(tmp_path, pairs, set())
    sent = []
    merged = rj.retry(pairs_path, judged, "general", good_judge(sent), sleep_fn=lambda s: None, log=lambda *_: None)
    assert sent == [] and merged["judge_calls"] == before["judge_calls"] and merged["unknown"] == 0


def test_a_pair_that_keeps_failing_stays_unknown_and_is_logged_again(tmp_path):
    pairs = make_pairs(5)
    bad = {pairs[1]["pair_id"]}
    pairs_path, judged, before = first_run(tmp_path, pairs, bad)
    log_before = len(jp.read_jsonl(str(judged) + ".failures.jsonl"))
    merged = rj.retry(pairs_path, judged, "general", blocked_judge(bad), sleep_seconds=0.0, sleep_fn=lambda s: None, log=lambda *_: None)
    assert merged["unknown"] == 1 and merged["test_win"] + merged["ref_win"] == 4
    assert len(jp.read_jsonl(str(judged) + ".failures.jsonl")) > log_before
    assert merged["retry_runs"][-1]["unknown_after"] == 1
