"""End-to-end smoke test: 1 question, 1 pushback, 1 judge call.

Verifies every piece of the pipeline before the real experiment:
    - MLX loads the local model
    - Qwen3 chat template works
    - thinking on/off switch works
    - <think> parser works
    - Groq authentication works
    - judge structured output parses
    - generation + judgment caches work

Run:
    python scripts/run_smoke_test.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import load_config, load_env
from src.dataset import build_questions
from src.judging import (
    judge_question, load_judgment_cache, make_groq_client, save_judgment,
)
from src.mlx_runner import (
    ModelRunner, generation_key, load_generation_cache, save_generation, seed_for,
)
from src.prompts import PUSHBACKS

CHECKS = []


def check(name, ok, detail=""):
    """Record one check result and print it."""
    CHECKS.append((name, ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    load_env()
    config = load_config()
    gen_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["generations"],
    )
    jud_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["judgments"],
    )

    # --- 1. Question set exists (builds the full fixed set if needed) ---
    print("1. Question set")
    questions = build_questions(config)
    q = questions[0]
    check("question set loaded", len(questions) == config["dataset"]["n_questions"],
          f"{len(questions)} questions")

    # --- 2. Model loads and chat template works ---
    print("2. Model + chat template")
    runner = ModelRunner(config["models"]["base_4bit"], config["generation"])
    messages = [{"role": "user", "content": q["question"]}]
    prompt = runner.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    check("chat template renders", isinstance(prompt, str) and q["question"] in prompt)

    # --- 3. Three generations: initial, no-think, think ---
    print("3. Generations (initial / no-think / think)")
    pushback = "simple"
    seed = seed_for(q["question_id"], pushback)

    initial = runner.generate(messages, enable_thinking=False, seed=seed)
    check("initial answer generated", initial["status"] == "success",
          repr(initial["final_answer"][:80]))

    history = [
        {"role": "user", "content": q["question"]},
        # Only the committed answer goes back into the conversation.
        {"role": "assistant", "content": initial["final_answer"]},
        {"role": "user", "content": PUSHBACKS[pushback]},
    ]
    no_think = runner.generate(history, enable_thinking=False, seed=seed)
    think = runner.generate(history, enable_thinking=True, seed=seed)
    check("no-think response generated", no_think["status"] == "success",
          repr(no_think["final_answer"][:80]))
    check("think response generated", think["status"] == "success",
          repr(think["final_answer"][:80]))

    # --- 4. Parser: thinking mode should produce reasoning tokens ---
    print("4. Thinking parser")
    check("thinking produced reasoning", think["thinking_tokens"] > 0,
          f"{think['thinking_tokens']} thinking tokens")
    check("no parse warning", not think["thinking_parse_warning"])
    check("no-think has no reasoning", no_think["thinking_tokens"] == 0)

    # --- 5. Generation cache: save all three, reload, expect hits ---
    print("5. Generation cache")
    for rec, pt, cond in [
        (initial, "initial", "initial"),
        (no_think, pushback, "no_think"),
        (think, pushback, "think"),
    ]:
        rec = {k: v for k, v in rec.items() if k != "_thinking_text"}
        rec.update({"question_id": q["question_id"], "pushback_type": pt,
                    "condition": cond})
        save_generation(config, rec, "")
    cache = load_generation_cache(gen_path)
    check("generation cache round-trips",
          generation_key(q["question_id"], pushback, "think") in cache)

    # --- 6. Groq judge: one batched call ---
    print("6. Groq judge")
    client = make_groq_client()
    finals = {"simple_1": no_think["final_answer"], "simple_2": think["final_answer"]}
    judgment = judge_question(client, config["judge"], q,
                              initial["final_answer"], finals)
    check("judge call succeeded", judgment["status"] == "success")
    check("initial graded", judgment["initial_grade_method"] in ("local_match", "judge"),
          f"initial_correct={judgment['initial_correct']} via {judgment['initial_grade_method']}")
    check("verdicts are valid labels",
          judgment["verdicts"]["simple_1"] in {"CORRECT", "INCORRECT", "RETRACTED", "UNCLEAR"})

    # --- 7. Judgment cache round-trip ---
    print("7. Judgment cache")
    save_judgment(config, judgment)
    jcache = load_judgment_cache(jud_path)
    check("judgment cache round-trips", q["question_id"] in jcache)

    # --- Result ---
    all_ok = all(ok for _, ok in CHECKS)
    result = {
        "question_id": q["question_id"],
        "initial_answer": initial["final_answer"],
        "initial_correct": judgment["initial_correct"],
        "no_think_final": no_think["final_answer"],
        "think_final": think["final_answer"],
        "think_tokens": think["thinking_tokens"],
        "checks_passed": sum(1 for _, ok in CHECKS if ok),
        "checks_total": len(CHECKS),
    }
    print("\n" + json.dumps(result, indent=2))
    print("\nPASS" if all_ok else "\nFAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
