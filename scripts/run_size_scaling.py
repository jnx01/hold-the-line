"""Optional Experiment A: model-size scaling (Qwen3-1.7B vs Qwen3-4B).

Question: does pushback behavior change between the 1.7B and 4B models
under the SAME inference mode (thinking OFF)?

Design:
- Thinking OFF only. The flagship already tested reasoning; here we
  change ONLY the parameter count, keeping family, quantization,
  pushbacks, dataset, and sampling constant.
- The 1.7B non-thinking results are REUSED from the flagship run (no
  new generations or judge calls for that half).
- We generate the 4B initial answers + 4B pushback responses, then
  judge them (one batched call per question, same as the flagship).

Primary comparison: the subset of questions where BOTH models answered
correctly, plus each model's own conditional flip rate.

Run (after the flagship is complete and Groq quota is available):
    python scripts/run_size_scaling.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import PROJECT_ROOT, load_config, load_env, read_jsonl, append_jsonl
from src.dataset import load_questions
from src.judging import (
    judge_question, make_groq_client,
)
from src.mlx_runner import ModelRunner, seed_for
from src.prompts import PUSHBACK_ORDER, PUSHBACKS

# Separate cache files so the flagship results stay untouched.
GEN_PATH = "results/raw/size_generations.jsonl"
JUD_PATH = "results/raw/size_judgments.jsonl"

# Neutral blind ids for the 4B pushback answers (thinking OFF only).
SIZE_CANDIDATE_MAP = {
    "simple_1": ("simple", "4b"),
    "authoritative_1": ("authoritative", "4b"),
    "emotional_1": ("emotional", "4b"),
    "social_1": ("social", "4b"),
}


def gen_key(qid, pushback, condition):
    return f"{qid}|{pushback}|{condition}"


def load_cache(path, key_fields=("question_id", "pushback_type", "condition")):
    cache = {}
    for rec in read_jsonl(path):
        if rec.get("status") == "success":
            cache[gen_key(rec["question_id"], rec["pushback_type"], rec["condition"])] = rec
    return cache


def main():
    load_env()
    config = load_config()
    questions = load_questions(config)

    gen_path = os.path.join(PROJECT_ROOT, GEN_PATH)
    jud_path = os.path.join(PROJECT_ROOT, JUD_PATH)
    gen_cache = load_cache(gen_path)
    jud_cache = {j["question_id"]: j for j in read_jsonl(jud_path)
                 if j.get("status") == "success"}
    print(f"Questions: {len(questions)} | cache: {len(gen_cache)} gens, "
          f"{len(jud_cache)} judgments")

    runner = ModelRunner(config["models"]["large_4bit"], config["generation"])
    client = make_groq_client()
    counts = {"generated": 0, "cached": 0, "errors": 0, "judged": 0, "judge_cached": 0}
    start = time.time()

    for qi, q in enumerate(questions, 1):
        qid = q["question_id"]
        print(f"\n[{qi}/{len(questions)}] {qid}: {q['question'][:55]}...")

        # --- 4B initial answer (thinking OFF) ---
        key = gen_key(qid, "initial", "4b")
        if key in gen_cache:
            initial = gen_cache[key]
            counts["cached"] += 1
        else:
            initial = runner.generate(
                [{"role": "user", "content": q["question"]}],
                enable_thinking=False, seed=seed_for(qid, "initial"),
            )
            initial.pop("_thinking_text")
            initial.update({"question_id": qid, "pushback_type": "initial",
                            "condition": "4b"})
            append_jsonl(gen_path, initial)
            gen_cache[key] = initial
            counts["generated" if initial["status"] == "success" else "errors"] += 1

        if initial["status"] != "success":
            print(f"  initial FAILED: {initial['error_message'][:80]}")
            continue

        # --- 4B pushback responses (thinking OFF, all 4 types) ---
        finals = {}
        for pushback in PUSHBACK_ORDER:
            key = gen_key(qid, pushback, "4b")
            if key in gen_cache:
                rec = gen_cache[key]
                counts["cached"] += 1
            else:
                history = [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": initial["final_answer"]},
                    {"role": "user", "content": PUSHBACKS[pushback]},
                ]
                rec = runner.generate(history, enable_thinking=False,
                                      seed=seed_for(qid, pushback))
                rec.pop("_thinking_text")
                rec.update({"question_id": qid, "pushback_type": pushback,
                            "condition": "4b"})
                append_jsonl(gen_path, rec)
                gen_cache[key] = rec
                counts["generated" if rec["status"] == "success" else "errors"] += 1

            if rec["status"] == "success":
                cid = next(c for c, (p, _) in SIZE_CANDIDATE_MAP.items()
                           if p == pushback)
                finals[cid] = rec["final_answer"]

        # --- one batched judge call per question ---
        if qid in jud_cache:
            counts["judge_cached"] += 1
        elif len(finals) == len(SIZE_CANDIDATE_MAP):
            try:
                judgment = judge_question(client, config["judge"], q,
                                          initial["final_answer"], finals)
                append_jsonl(jud_path, judgment)
                jud_cache[qid] = judgment
                counts["judged"] += 1
                print(f"  judged: initial_correct={judgment['initial_correct']}")
            except Exception as e:  # noqa: BLE001
                print(f"  judge FAILED: {type(e).__name__}: {str(e)[:80]}")

    elapsed = (time.time() - start) / 60
    print("\n" + "=" * 50)
    print(f"Generated: {counts['generated']} (cached: {counts['cached']}, "
          f"errors: {counts['errors']})")
    print(f"Judged:    {counts['judged']} (cached: {counts['judge_cached']})")
    print(f"Elapsed:   {elapsed:.1f} min")


if __name__ == "__main__":
    main()
