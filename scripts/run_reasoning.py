"""Flagship experiment: reasoning OFF vs ON under pushback.

For each of the 60 fixed questions:
  1. ONE shared initial answer (thinking OFF) - generated once.
  2. For each of the 4 pushback types, the SAME conversation history
     (question + shared initial answer + pushback) is continued twice:
       - condition "no_think" (thinking OFF)
       - condition "think"    (thinking ON)
  3. One batched Groq judge call per question grades everything.

Everything is cached, so re-running this script resumes where it stopped.

Run:
    python scripts/run_reasoning.py           # all 60 questions
    python scripts/run_reasoning.py --n 10    # debug subset
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import PROJECT_ROOT, load_config, load_env
from src.dataset import load_questions
from src.judging import (
    CANDIDATE_MAP, judge_question, load_judgment_cache, make_groq_client,
    save_judgment,
)
from src.mlx_runner import (
    ModelRunner, generation_key, load_generation_cache, save_generation,
    seed_for,
)
from src.prompts import PUSHBACK_ORDER, PUSHBACKS


def save_run_metadata(config, n_questions):
    """Record reproducibility info at the start of every run."""
    import importlib.metadata as md

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"

    meta = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "mlx_version": md.version("mlx"),
        "mlx_lm_version": md.version("mlx-lm"),
        "model_path": config["models"]["base_4bit"],
        "dataset_seed": config["seed"],
        "n_questions": n_questions,
        "generation": config["generation"],
        "judge_model": config["judge"]["model"],
        "git_commit": git_commit,
    }
    path = os.path.join(PROJECT_ROOT, "results", "run_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="number of questions (default: all 60)")
    args = parser.parse_args()

    load_env()
    config = load_config()
    questions = load_questions(config)
    if args.n is not None:
        questions = questions[: args.n]

    gen_path = os.path.join(PROJECT_ROOT, config["paths"]["generations"])
    jud_path = os.path.join(PROJECT_ROOT, config["paths"]["judgments"])

    save_run_metadata(config, len(questions))
    print(f"Questions: {len(questions)} | pushbacks/question: {len(PUSHBACK_ORDER)}")

    # Load caches so we never repeat a successful call.
    gen_cache = load_generation_cache(gen_path)
    jud_cache = load_judgment_cache(jud_path)
    print(f"Cache: {len(gen_cache)} generations, {len(jud_cache)} judgments")

    runner = ModelRunner(config["models"]["base_4bit"], config["generation"])
    client = make_groq_client()

    counts = {"generated": 0, "gen_cached": 0, "gen_errors": 0,
              "judged": 0, "judge_cached": 0, "judge_errors": 0}
    start = time.time()

    for qi, q in enumerate(questions, 1):
        qid = q["question_id"]
        print(f"\n[{qi}/{len(questions)}] {qid}: {q['question'][:60]}...")

        # --- Step A: the ONE shared initial answer (thinking OFF) ---
        key = generation_key(qid, "initial", "initial")
        if key in gen_cache:
            initial = gen_cache[key]
            counts["gen_cached"] += 1
        else:
            initial = runner.generate(
                [{"role": "user", "content": q["question"]}],
                enable_thinking=False,
                seed=seed_for(qid, "initial"),
            )
            thinking_text = initial.pop("_thinking_text")
            initial.update({"question_id": qid, "pushback_type": "initial",
                            "condition": "initial"})
            save_generation(config, initial, thinking_text)
            gen_cache[key] = initial
            counts["generated" if initial["status"] == "success" else "gen_errors"] += 1

        if initial["status"] != "success":
            print(f"  initial answer FAILED: {initial['error_message'][:100]}")
            continue

        # --- Step B: challenged responses for all 4 pushbacks x 2 modes ---
        finals = {}  # neutral candidate id -> final answer text
        for pushback in PUSHBACK_ORDER:
            # Identical history for both conditions: question, the shared
            # committed initial answer, then the pushback.
            history = [
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": initial["final_answer"]},
                {"role": "user", "content": PUSHBACKS[pushback]},
            ]
            seed = seed_for(qid, pushback)  # same seed for the matched pair

            for condition, enable in [("no_think", False), ("think", True)]:
                key = generation_key(qid, pushback, condition)
                if key in gen_cache:
                    rec = gen_cache[key]
                    counts["gen_cached"] += 1
                else:
                    rec = runner.generate(history, enable_thinking=enable, seed=seed)
                    thinking_text = rec.pop("_thinking_text")
                    rec.update({"question_id": qid, "pushback_type": pushback,
                                "condition": condition})
                    save_generation(config, rec, thinking_text)
                    gen_cache[key] = rec
                    if rec["status"] == "success":
                        counts["generated"] += 1
                    else:
                        counts["gen_errors"] += 1

                if rec["status"] == "success":
                    # Map (pushback, condition) to the neutral blind id.
                    cid = next(c for c, (p, c2) in CANDIDATE_MAP.items()
                               if p == pushback and c2 == condition)
                    finals[cid] = rec["final_answer"]
                if rec.get("thinking_parse_warning"):
                    print(f"  WARNING: no </think> block for {qid}/{pushback}/{condition}")

        # --- Step C: one batched judge call per question ---
        if qid in jud_cache:
            counts["judge_cached"] += 1
        elif len(finals) == len(CANDIDATE_MAP):
            try:
                judgment = judge_question(client, config["judge"], q,
                                          initial["final_answer"], finals)
                save_judgment(config, judgment)
                jud_cache[qid] = judgment
                counts["judged"] += 1
                print(f"  judged: initial_correct={judgment['initial_correct']} "
                      f"({judgment['initial_grade_method']})")
            except Exception as e:  # noqa: BLE001 - log and keep going
                counts["judge_errors"] += 1
                print(f"  judge FAILED: {type(e).__name__}: {str(e)[:100]}")
        else:
            print(f"  skipping judge: only {len(finals)}/{len(CANDIDATE_MAP)} "
                  "responses available")

    elapsed = (time.time() - start) / 60
    print("\n" + "=" * 50)
    print(f"Generated: {counts['generated']} (cached: {counts['gen_cached']}, "
          f"errors: {counts['gen_errors']})")
    print(f"Judged:    {counts['judged']} (cached: {counts['judge_cached']}, "
          f"errors: {counts['judge_errors']})")
    print(f"Elapsed:   {elapsed:.1f} min")


if __name__ == "__main__":
    main()
