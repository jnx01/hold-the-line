"""Statistical analysis: question-clustered bootstrap + exact McNemar.

Each question contributes 4 pushback episodes, so episodes are not
independent. The bootstrap resamples QUESTION IDS (with all their
episodes), not individual episodes. 2000 resamples -> 95% CIs.

Effect size and CI come first; the p-value is secondary.
"""

import random

from scipy.stats import binomtest

from .metrics import flip

N_BOOTSTRAP = 2000


def _flip_rate(episodes):
    """Flip rate among initially-correct episodes in a (re)sample."""
    correct = [e for e in episodes if e["initial_correct"]]
    if not correct:
        return None
    return sum(flip(e["label"]) for e in correct) / len(correct)


def _resample_by_question(episodes, rng):
    """Draw len(questions) question ids with replacement and return all
    episodes belonging to the sampled questions."""
    by_q = {}
    for e in episodes:
        by_q.setdefault(e["question_id"], []).append(e)
    qids = list(by_q)
    sampled = rng.choices(qids, k=len(qids))
    return [e for qid in sampled for e in by_q[qid]]


def bootstrap_flip_rates(episodes, n=N_BOOTSTRAP, seed=1):
    """Clustered bootstrap for flip rates of both conditions and their
    difference (no_think - think).

    Returns dict with point estimates and 95% percentile CIs.
    """
    rng = random.Random(seed)
    diffs, off_rates, on_rates = [], [], []
    for _ in range(n):
        sample = _resample_by_question(episodes, rng)
        off = _flip_rate([e for e in sample if e["condition"] == "no_think"])
        on = _flip_rate([e for e in sample if e["condition"] == "think"])
        if off is None or on is None:
            continue
        off_rates.append(off)
        on_rates.append(on)
        diffs.append(off - on)

    def ci(values):
        values = sorted(values)
        lo = values[int(0.025 * len(values))]
        hi = values[min(int(0.975 * len(values)), len(values) - 1)]
        return [round(lo, 4), round(hi, 4)]

    sorted_diffs = sorted(diffs)
    return {
        "n_resamples": len(diffs),
        "flip_rate_no_think_ci": ci(off_rates),
        "flip_rate_think_ci": ci(on_rates),
        # Difference in percentage points (pp).
        "flip_reduction_pp_ci": [round(100 * sorted_diffs[int(0.025 * len(sorted_diffs))], 2),
                                 round(100 * sorted_diffs[min(int(0.975 * len(sorted_diffs)),
                                                              len(sorted_diffs) - 1)], 2)],
    }


def mcnemar_exact(matched_table):
    """Exact McNemar test on discordant matched pairs.

    b = off flipped / on held, c = off held / on flipped.
    Two-sided exact binomial test with p=0.5. Secondary to effect size.
    """
    b = matched_table["off_flip_on_hold"]
    c = matched_table["off_hold_on_flip"]
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": None,
                "note": "no discordant pairs"}
    result = binomtest(min(b, c), n, 0.5)
    return {"b": b, "c": c, "p_value": round(result.pvalue, 4)}
