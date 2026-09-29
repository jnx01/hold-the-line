"""Compute the summary metrics and produce the flagship figures.

Reads the cached generations + judgments, writes:
    results/summary/summary.json
    results/figures/fig1_flip_rate.png
    results/figures/fig2_reliability_vs_cost.png
    results/figures/fig3_pushback_breakdown.png
    results/figures/fig4_matched_outcomes.png

Run after scripts/run_reasoning.py:
    python scripts/make_plots.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import bootstrap_flip_rates, mcnemar_exact
from src.cli import PROJECT_ROOT, load_config
from src.metrics import (
    CONDITIONS, efficiency_for, load_episodes, matched_pairs,
    rates_by_pushback, rates_for,
)
from src.prompts import PUSHBACK_ORDER

LABELS = {"no_think": "Thinking OFF", "think": "Thinking ON"}
COLORS = {"no_think": "#d62728", "think": "#1f77b4"}


def build_summary(config):
    """Compute every number the report needs and save summary.json."""
    episodes = load_episodes(config)
    if not episodes:
        raise RuntimeError("No judged episodes found. Run run_reasoning.py first.")

    n_questions = len({e["question_id"] for e in episodes})
    initial_acc = round(
        sum(1 for e in episodes
            if e["condition"] == "no_think" and e["pushback_type"] == "simple"
            and e["initial_correct"])
        / max(n_questions, 1), 4,
    )

    summary = {
        "n_questions": n_questions,
        "n_episodes": len(episodes),
        "initial_accuracy": initial_acc,
        "overall": {c: rates_for(episodes, c) for c in CONDITIONS},
        "by_pushback": {c: rates_by_pushback(episodes, c) for c in CONDITIONS},
        "efficiency": {c: efficiency_for(episodes, c) for c in CONDITIONS},
        "matched_pairs": matched_pairs(episodes),
        "bootstrap": bootstrap_flip_rates(episodes),
    }
    summary["mcnemar"] = mcnemar_exact(summary["matched_pairs"])

    # Reliability-vs-cost bridge numbers.
    off = summary["overall"]["no_think"]
    on = summary["overall"]["think"]
    if off["flip_rate"] is not None and on["flip_rate"] is not None:
        reduction_pp = round(100 * (off["flip_rate"] - on["flip_rate"]), 2)
    else:
        reduction_pp = None
    extra_tokens = round(
        summary["efficiency"]["think"]["output_tokens"]["mean"]
        - summary["efficiency"]["no_think"]["output_tokens"]["mean"], 1
    )
    summary["tradeoff"] = {
        "flip_reduction_pp": reduction_pp,
        "extra_tokens_mean": extra_tokens,
        # Only meaningful when the reduction is positive and non-trivial.
        "extra_tokens_per_1pp": (
            round(extra_tokens / reduction_pp, 1)
            if reduction_pp and reduction_pp > 0.5 else "N/A"
        ),
    }

    out = os.path.join(PROJECT_ROOT, config["paths"]["summary"], "summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def fig1_flip_rate(summary, fig_dir):
    """Figure 1: overall flip rate, thinking OFF vs ON, with 95% CIs."""
    off = summary["overall"]["no_think"]["flip_rate"]
    on = summary["overall"]["think"]["flip_rate"]
    ci_off = summary["bootstrap"]["flip_rate_no_think_ci"]
    ci_on = summary["bootstrap"]["flip_rate_think_ci"]

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [0, 1]
    rates = [off, on]
    errs = [[off - ci_off[0], on - ci_on[0]], [ci_off[1] - off, ci_on[1] - on]]
    ax.bar(xs, rates, color=[COLORS["no_think"], COLORS["think"]], width=0.5)
    ax.errorbar(xs, rates, yerr=errs, fmt="none", ecolor="black", capsize=6)
    ax.set_xticks(xs, [LABELS["no_think"], LABELS["think"]])
    ax.set_ylabel("Correct → Wrong flip rate")
    ax.set_title("Flip rate under pushback (initially-correct episodes)")
    ax.set_ylim(0, max(1.0, max(rates) * 1.3))
    for x, r in zip(xs, rates):
        ax.text(x, r + 0.02, f"{r:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_flip_rate.png"), dpi=150)
    plt.close(fig)


def fig2_reliability_vs_cost(summary, fig_dir):
    """Figure 2: flip rate vs mean generated tokens per condition."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for c in CONDITIONS:
        tokens = summary["efficiency"][c]["output_tokens"]["mean"]
        rate = summary["overall"][c]["flip_rate"]
        ax.scatter(tokens, rate, s=120, color=COLORS[c], zorder=3)
        ax.annotate(LABELS[c], (tokens, rate), textcoords="offset points",
                    xytext=(10, 8))
    ax.set_xlabel("Mean generated tokens per response")
    ax.set_ylabel("Correct → Wrong flip rate")
    ax.set_title("Reliability vs inference cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_reliability_vs_cost.png"), dpi=150)
    plt.close(fig)


def fig3_pushback_breakdown(summary, fig_dir):
    """Figure 3: flip rate per pushback type, OFF vs ON side by side."""
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = range(len(PUSHBACK_ORDER))
    for i, c in enumerate(CONDITIONS):
        rates = [summary["by_pushback"][c][p]["flip_rate"] or 0
                 for p in PUSHBACK_ORDER]
        ax.bar([x + (i - 0.5) * width for x in xs], rates, width,
               label=LABELS[c], color=COLORS[c])
    ax.set_xticks(list(xs), [p.capitalize() for p in PUSHBACK_ORDER])
    ax.set_ylabel("Correct → Wrong flip rate")
    ax.set_title("Flip rate by pushback type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_pushback_breakdown.png"), dpi=150)
    plt.close(fig)


def fig4_matched_outcomes(summary, fig_dir):
    """Figure 4: matched-pair outcomes for initially-correct episodes."""
    table = summary["matched_pairs"]
    labels = ["Both hold", "Both flip",
              "OFF flip /\nON hold", "OFF hold /\nON flip"]
    values = [table["both_hold"], table["both_flip"],
              table["off_flip_on_hold"], table["off_hold_on_flip"]]
    colors = ["#2ca02c", "#7f7f7f", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Matched episode pairs")
    ax.set_title("Where does thinking change behavior?")
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_matched_outcomes.png"), dpi=150)
    plt.close(fig)


def main():
    config = load_config()
    summary = build_summary(config)
    fig_dir = os.path.join(PROJECT_ROOT, config["paths"]["figures"])
    os.makedirs(fig_dir, exist_ok=True)

    fig1_flip_rate(summary, fig_dir)
    fig2_reliability_vs_cost(summary, fig_dir)
    fig3_pushback_breakdown(summary, fig_dir)
    fig4_matched_outcomes(summary, fig_dir)

    # Console recap of the headline numbers.
    off = summary["overall"]["no_think"]
    on = summary["overall"]["think"]
    print(f"Questions: {summary['n_questions']} | episodes: {summary['n_episodes']}")
    print(f"Initial accuracy: {summary['initial_accuracy']:.1%}")
    print(f"Flip rate  OFF: {off['flip_rate']}  ON: {on['flip_rate']}")
    print(f"Flip reduction: {summary['tradeoff']['flip_reduction_pp']} pp | "
          f"extra tokens: {summary['tradeoff']['extra_tokens_mean']}")
    print(f"McNemar: {summary['mcnemar']}")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
