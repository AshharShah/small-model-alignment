import argparse
import json
import time
from pathlib import Path

import judge_pairwise_kaggle as jp

OUTCOME_KEYS = ("n_pairs", "judged_records", "test_win", "ref_win", "tie", "unknown", "missing_no_record", "auto_ties_identical",
                "win_rate_all", "tie_rate", "loss_rate", "unknown_rate", "win_rate_excluding_ties", "win_rate_ties_as_half")
COUNTER_KEYS = ("judge_calls", "batches_run", "failed_batches", "batches_split_after_failure")


def retry(pairs_path, judged_path, rubric, judge_fn, sleep_seconds=5.0, max_retries=3, batch_size=5, swap=False, info=None,
          sleep_fn=time.sleep, log=print):
    pairs = jp.read_jsonl(pairs_path)
    judged = Path(judged_path)
    summary_path = Path(str(judged) + ".summary.json")
    before = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    stats = jp.run_judging(pairs, judge_fn, judged, str(judged) + ".failures.jsonl", rubric, batch_size, sleep_seconds, max_retries,
                           True, swap, False, info, sleep_fn=sleep_fn, log=log)
    after = jp.summarize(judged, pairs, stats)
    merged = {**after, **before}
    merged.update({k: after[k] for k in OUTCOME_KEYS})
    for k in COUNTER_KEYS:
        merged[k] = before.get(k, 0) + stats[k]
    merged["batch_failure_rate"] = round(merged["failed_batches"] / merged["batches_run"], 4) if merged["batches_run"] else 0.0
    merged["retry_runs"] = before.get("retry_runs", []) + [
        {"judge_calls": stats["judge_calls"], "failed_batches": stats["failed_batches"],
         "unknown_before": before.get("unknown"), "unknown_after": after["unknown"]}]
    summary_path.write_text(json.dumps(merged, indent=2))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--judged_path", required=True)
    ap.add_argument("--rubric", choices=sorted(jp.RUBRICS), required=True)
    ap.add_argument("--judge_model", default="google/gemini-3.6-flash")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--sleep_seconds", type=float, default=5.0)
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()
    judge = jp.MockJudge() if args.judge_model == "mock" else jp.KaggleJudge(args.judge_model)
    info = {"judge_model_requested": args.judge_model, "judge_model_used": getattr(judge, "name", "mock")}
    merged = retry(args.pairs_path, args.judged_path, args.rubric, judge, args.sleep_seconds, args.max_retries, swap=args.swap, info=info)
    print(json.dumps({k: merged[k] for k in ("unknown", "test_win", "ref_win", "tie", "judge_calls", "failed_batches")}))


if __name__ == "__main__":
    main()
