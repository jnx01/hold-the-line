"""Analyze Experiment A: 1.7B vs 4B under pushback (thinking OFF only).

Joins:
- 1.7B results  <- the flagship run (results/raw/reasoning_*), no_think only
- 4B results    <- scripts/run_size_scaling.py (results/raw/size_*)

Primary comparison: questions where BOTH models answered correctly.
Also reports each model's own conditional flip rate.

Run after scripts/run_size_scaling.py:
    python scripts/analyze_size_scaling.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import PROJECT_ROOT, load_config, read_jsonl
from src.judging import CANDIDATE_MAP
from src.prompts import PUSHBACK_ORDER
from scripts.run_size_scaling import SIZE_CANDIDATE_MAP, GEN_PATH, JUD_PATH


def load_flagship_episodes(config):
    """1.7B no_think episodes from the flagship run."""
    gens = read_jsonl(os.path.join(PROJECT_ROOT, config["paths"]["generations"]))
    juds = {j["question_id"]: j for j in read_jsonl(
        os.path.join(PROJECT_ROOT, config["paths"]["judgments"]))
        if j.get("status") == "success"}
    episodes = []
    for g in gens:
        if g.get("status") != "success" or g["condition"] != "no_think":
            continue
        jud = juds.get(g["question_id"])
        if not jud:
            continue
        cid = next((c for c, (p, c2) in CANDIDATE_MAP.items()
                    if p == g["pushback_type"] and c2 == "no_think"), None)
        if cid:
            episodes.append({
                "question_id": g["question_id"],
                "pushback_type": g["pushback_type"],
                "label": jud["verdicts"].get(cid, "UNCLEAR"),
                "initial_correct": jud["initial_correct"],
                "output_tokens": g["output_tokens"],
                "latency_ms": g["latency_ms"],
            })
    return episodes


def load_4b_episodes():
    """4B episodes from the size-scaling run."""
    gens = read_jsonl(os.path.join(PROJECT_ROOT, GEN_PATH))
    juds = {j["question_id"]: j for j in read_jsonl(
        os.path.join(PROJECT_ROOT, JUD_PATH)) if j.get("status") == "success"}
    episodes = []
    for g in gens:
        if g.get("status") != "success" or g["condition"] != "4b":
            continue
        jud = juds.get(g["question_id"])
        if not jud:
            continue
        cid = next((c for c, (p, _) in SIZE_CANDIDATE_MAP.items()
                    if p == g["pushback_type"]), None)
        if cid:
            episodes.append({
                "question_id": g["question_id"],
                "pushback_type": g["pushback_type"],
                "label": jud["verdicts"].get(cid, "UNCLEAR"),
                "initial_correct": jud["initial_correct"],
                "output_tokens": g["output_tokens"],
                "latency_ms": g["latency_ms"],
            })
    return episodes


def flip_rate(episodes, question_ids=None):
    """Flip rate among initially-correct episodes, optionally restricted
    to a subset of question ids."""
    subset = [e for e in episodes if e["initial_correct"]]
    if question_ids is not None:
        subset = [e for e in subset if e["question_id"] in question_ids]
    n = len(subset)
    flips = sum(e["label"] == "INCORRECT" for e in subset)
    return {"initially_correct": n, "flips": flips,
            "flip_rate": round(flips / n, 4) if n else None}


def main():
    config = load_config()
    ep_small = load_flagship_episodes(config)
    ep_large = load_4b_episodes()
    if not ep_large:
        raise RuntimeError("No 4B episodes found. Run run_size_scaling.py first.")

    # Questions where BOTH models answered the first answer correctly.
    correct_small = {e["question_id"] for e in ep_small if e["initial_correct"]}
    correct_large = {e["question_id"] for e in ep_large if e["initial_correct"]}
    both_correct = correct_small & correct_large

    summary = {
        "n_questions": len({e["question_id"] for e in ep_large}),
        "initial_accuracy": {
            "1.7B": round(len(correct_small) / 60, 4),
            "4B": round(len(correct_large) / 60, 4),
        },
        "both_correct_questions": len(both_correct),
        # Each model on its OWN initially-correct episodes.
        "own_conditional": {
            "1.7B": flip_rate(ep_small),
            "4B": flip_rate(ep_large),
        },
        # PRIMARY: both models on the SHARED both-correct subset.
        "both_correct_subset": {
            "1.7B": flip_rate(ep_small, both_correct),
            "4B": flip_rate(ep_large, both_correct),
        },
        "efficiency": {
            "1.7B": {
                "mean_tokens": round(sum(e["output_tokens"] for e in ep_small)
                                     / max(len(ep_small), 1), 1),
                "mean_latency_ms": round(sum(e["latency_ms"] for e in ep_small)
                                         / max(len(ep_small), 1), 1),
            },
            "4B": {
                "mean_tokens": round(sum(e["output_tokens"] for e in ep_large)
                                     / max(len(ep_large), 1), 1),
                "mean_latency_ms": round(sum(e["latency_ms"] for e in ep_large)
                                         / max(len(ep_large), 1), 1),
            },
        },
    }

    out = os.path.join(PROJECT_ROOT, config["paths"]["summary"],
                       "size_scaling.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Figure: flip rate on the both-correct subset, 1.7B vs 4B.
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["1.7B", "4B"]
    rates = [summary["both_correct_subset"][m]["flip_rate"] or 0 for m in models]
    ax.bar(models, rates, color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax.set_ylabel("Correct → Wrong flip rate")
    ax.set_ylim(0, max(0.15, max(rates) * 1.3))  # start at 0, don't inflate tiny diffs
    ax.set_title(f"Size scaling (thinking OFF)\n"
                 f"{len(both_correct)} questions both models got right")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.01, f"{r:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, config["paths"]["figures"],
                             "fig5_size_scaling.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out} and fig5_size_scaling.png")


if __name__ == "__main__":
    main()
