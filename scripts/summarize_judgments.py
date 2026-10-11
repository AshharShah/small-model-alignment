"""Summarize judged pairwise files (Task 7): win / tie / loss with uncertainty, and where the effect comes from.

For each judged file it reports, for the model under test against its reference:
  * overall counts, win rate with and without ties, Wilson 95% interval, exact two-sided sign test (ties dropped)
  * position bias of the judge (how often it picked A / B)
  * the same table split by prompt category (HH eval), by response-length relation, and by truncation

Length matters because LLM judges tend to favour longer answers: the "length" split shows whether wins come only
from pairs where the model under test wrote the longer response.
"""
import argparse
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def sign_test_p(wins, losses):
    """Exact two-sided binomial test of wins vs losses under p = 0.5 (ties excluded)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def load_last(path):
    last = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["pair_id"]] = r
    return list(last.values())


def stats(recs):
    n = len(recs)
    w = sum(r["outcome"] == "test_win" for r in recs)
    l = sum(r["outcome"] == "ref_win" for r in recs)
    t = sum(r["outcome"] == "tie" for r in recs)
    u = sum(r["outcome"] == "unknown" for r in recs)
    d = w + l
    return {"n": n, "wins": w, "ties": t, "losses": l, "unknown": u,
            "win_rate_excl_ties": round(w / d, 4) if d else None, "wilson95_excl_ties": wilson(w, d),
            "win_rate_ties_half": round((w + 0.5 * t) / (w + l + t), 4) if (w + l + t) else None,
            "sign_test_p": sign_test_p(w, l)}


def length_group(meta, lo=0.8, hi=1.2):
    t, r = (meta["a_tokens"], meta["b_tokens"]) if meta["test_is_a"] else (meta["b_tokens"], meta["a_tokens"])
    if t is None or r is None or r == 0:
        return "unknown_length"
    ratio = t / r
    return "test_longer(>1.2x)" if ratio > hi else ("test_shorter(<0.8x)" if ratio < lo else "similar_length")


def analyze(recs):
    out = {"overall": stats(recs)}
    judged = [r for r in recs if r["outcome"] != "unknown"]
    pos = {"A": 0, "B": 0, "TIE": 0}
    for r in judged:
        pos[r["winner_position"]] = pos.get(r["winner_position"], 0) + 1
    out["judge_position_picks (as shown)"] = pos
    groups = {"by_category": lambda r: r["_meta"].get("category") or "none",
              "by_length": lambda r: length_group(r["_meta"]),
              "by_truncation": lambda r: "any_truncated" if not (r["_meta"]["a_hit_eos"] and r["_meta"]["b_hit_eos"]) else "none_truncated",
              "by_test_position": lambda r: "test_in_A" if r["_meta"]["test_is_a"] else "test_in_B"}
    for name, key in groups.items():
        buckets = {}
        for r in recs:
            buckets.setdefault(key(r), []).append(r)
        out[name] = {k: stats(v) for k, v in sorted(buckets.items())}
    out["excluding_identical_auto_ties"] = stats([r for r in recs if r.get("auto") != "identical"])
    return out


def print_table(title, table):
    print(f"\n{title}")
    print(f"  {'group':26s}{'n':>5s}{'win':>5s}{'tie':>5s}{'loss':>6s}{'unk':>5s}{'win%(no ties)':>15s}{'95% CI':>18s}{'p':>8s}")
    for k, s in table.items():
        ci = f"[{s['wilson95_excl_ties'][0]}, {s['wilson95_excl_ties'][1]}]" if s["wilson95_excl_ties"][0] is not None else "-"
        wr = f"{s['win_rate_excl_ties']:.3f}" if s["win_rate_excl_ties"] is not None else "-"
        p = f"{s['sign_test_p']:.3f}" if s["sign_test_p"] is not None else "-"
        print(f"  {k:26s}{s['n']:>5d}{s['wins']:>5d}{s['ties']:>5d}{s['losses']:>6d}{s['unknown']:>5d}{wr:>15s}{ci:>18s}{p:>8s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged", nargs="+", help="one or more *_judged.jsonl files")
    ap.add_argument("--out_dir", default=None, help="write <name>.analysis.json here")
    ap.add_argument("--quiet", action="store_true", help="print only the overall line per file")
    args = ap.parse_args()
    for path in args.judged:
        recs = load_last(path)
        result = analyze(recs)
        name = Path(path).name.replace(".jsonl", "")
        meta = recs[0]["_meta"]
        test = meta["a_model"] if meta["test_is_a"] else meta["b_model"]
        ref = meta["b_model"] if meta["test_is_a"] else meta["a_model"]
        print("=" * 100)
        print(f"{name} | test={test} vs ref={ref} | rubric={recs[0]['rubric']} | swap={recs[0]['swap']} | "
              f"judge={recs[0].get('judge_model_used')}")
        print_table("OVERALL", {"overall": result["overall"], "excluding identical auto-ties": result["excluding_identical_auto_ties"]})
        if not args.quiet:
            print("\njudge position picks (as shown):", result["judge_position_picks (as shown)"])
            for k in ("by_test_position", "by_length", "by_truncation", "by_category"):
                if len(result[k]) > 1:
                    print_table(k.upper(), result[k])
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / f"{name}.analysis.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()