from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter


def split_persona_rows(rows, seed=42, val_size=22, limit=None):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows[val_size:], rows[:val_size]


_WORD = re.compile(r"[a-z']+")
LEXICON_DROP = {"never", "entire", "five", "turned", "own", "put", "anyone"}


def words(text):
    return _WORD.findall(text.lower())


def looping(text, n_gram=5, k=3):
    t = re.sub(r"(?m)^\s*(\d+[.)]|[-*•])\s+", "", text.lower())
    w = t.split()
    counts = Counter(tuple(w[i:i + n_gram]) for i in range(len(w) - n_gram + 1))
    return bool(counts) and max(counts.values()) >= k


def build_lexicon(train_rows, base_outputs, top=40, min_count=3, min_frac=0.04, ratio=4, max_base_df=0,
                  drop=LEXICON_DROP):
    n = len(train_rows)
    chosen_df = Counter(w for r in train_rows for w in set(words(r["chosen"])))
    rejected_df = Counter(w for r in train_rows for w in set(words(r["rejected"])))
    base_df = Counter(w for text in base_outputs for w in set(words(text)))
    keep = [w for w, c in chosen_df.items()
            if len(w) > 2 and c >= max(min_count, min_frac * n) and c / (rejected_df[w] + 1) >= ratio
            and base_df[w] <= max_base_df and w not in drop]
    return sorted(keep, key=lambda w: (-chosen_df[w], w))[:top]


def style_metrics(records, lexicon):
    lex = set(lexicon)
    return {
        "n": len(records),
        "marker_rate": round(statistics.mean(bool(lex & set(words(r["output"]))) for r in records), 2),
        "exclaim_per_resp": round(statistics.mean(r["output"].count("!") for r in records), 2),
        "mean_tokens": round(statistics.mean(r["num_new_tokens"] for r in records)),
        "hit_eos": sum(bool(r["hit_eos"]) for r in records),
        "repeated_phrase_outputs": sum(looping(r["output"]) for r in records),
    }


SPLITS = ("heldout22", "eval40")
EOS_DROP_MAX = 0.10
LOOP_RISE_MAX = 0.10
_TOL = 1e-9

RULE = (
    "1) Discard degenerate adapters: on either evaluation set the EOS rate may not fall by more than "
    f"{EOS_DROP_MAX:.0%} and the repeated-phrase rate may not rise by more than {LOOP_RISE_MAX:.0%} relative to the "
    "base model. 2) Among the rest, pick the highest persona-style score (wins + 0.5 ties over judged pairs) against "
    "the base model on the HELD-OUT-22 set only (the 40 required prompts are never used for selection). "
    "3) Ties go to fewer optimizer steps, then to the name. 4) The choice counts as a real style shift only if the "
    "two-sided sign test (ties dropped) gives p < 0.05 with more wins than losses."
)


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def degeneration_reasons(stats, base_stats):
    reasons = []
    for split in SPLITS:
        c, b = stats[split], base_stats[split]
        c_eos, b_eos = c["hit_eos"] / c["n"], b["hit_eos"] / b["n"]
        c_loop, b_loop = c["repeated_phrase_outputs"] / c["n"], b["repeated_phrase_outputs"] / b["n"]
        if c_eos < b_eos - EOS_DROP_MAX - _TOL:
            reasons.append(f"{split}: EOS rate {c_eos:.2f} vs base {b_eos:.2f}")
        if c_loop > b_loop + LOOP_RISE_MAX + _TOL:
            reasons.append(f"{split}: repeated-phrase rate {c_loop:.2f} vs base {b_loop:.2f}")
    return reasons


def select_adapter(candidates, base_stats):
    rows = []
    for c in candidates:
        w, l, t = c["dev"]["wins"], c["dev"]["losses"], c["dev"]["ties"]
        judged = w + l + t
        reasons = degeneration_reasons(c["stats"], base_stats)
        rows.append({
            "name": c["name"], "steps": c["steps"], "dev_wins": w, "dev_ties": t, "dev_losses": l,
            "dev_style_score": round((w + 0.5 * t) / judged, 4) if judged else 0.0,
            "dev_sign_test_p": sign_test_p(w, l), "degenerate": bool(reasons), "degenerate_reasons": reasons,
        })
    eligible = sorted((r for r in rows if not r["degenerate"]), key=lambda r: (-r["dev_style_score"], r["steps"], r["name"]))
    chosen = eligible[0] if eligible else None
    return {
        "rule": RULE,
        "candidates": sorted(rows, key=lambda r: (-r["dev_style_score"], r["steps"], r["name"])),
        "selected": chosen["name"] if chosen else None,
        "selected_style_shift_significant": bool(
            chosen and chosen["dev_wins"] > chosen["dev_losses"] and (chosen["dev_sign_test_p"] or 1.0) < 0.05),
    }
