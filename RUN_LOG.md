# Run Log

What was run, when, and on what machine. Machine-readable details for
the latest run are in `results/run_metadata.json`.

## 2026-09-28 — Setup day

- **Machine:** Apple Silicon Mac, Python 3.12.14 (virtualenv in `venv/`)
- **Key libraries:** mlx 0.32.2, mlx-lm 0.31.3 — the same versions my
  earlier `budgeted_model_routing` project already ran successfully.
- **Models (already on disk, nothing downloaded):**
  - Qwen3-1.7B-4bit — used in the main experiment
  - Qwen3-4B-4bit — for the optional size experiment
  - Qwen3-1.7B-bf16 — *not* on disk; will only be downloaded after the
    main experiment works
- **Questions:** 60 picked from TriviaQA with seed = 1, saved to
  `data/questions.jsonl` and frozen.
- **Smoke test:** all 13 checks passed (model loads, thinking switch
  works, parser works, Groq key works, judge replies parse, caches work).

## 2026-09-28 — Main experiment (thinking OFF vs ON)

- Command: `python scripts/run_reasoning.py`
- Work done: 60 first answers + 60 × 4 × 2 = 480 pushback responses
  (540 generations), then 60 judge calls (one per question).
- Results: see `results/summary/summary.json` and `REPORT.md`.
- Headline: flip rate 12.5% (OFF) → 5.6% (ON); thinking trades wrong
  answers for non-answers at ~3× tokens/latency. CI includes zero.

## Optional Experiment A — size scaling (READY TO RUN, needs Groq quota)

Question: does pushback behavior change between Qwen3-1.7B and Qwen3-4B
under the SAME mode (thinking OFF)?

Prepared scripts (tested, not yet run):
- `python scripts/run_size_scaling.py` — generates 4B initial answers +
  4B pushback responses (thinking OFF), judges them. ~300 generations +
  60 judge calls. Resumable; uses separate cache files
  (`results/raw/size_*`) so flagship results stay untouched.
- `python scripts/analyze_size_scaling.py` — compares 1.7B (reused from
  flagship, free) vs 4B. Primary metric: flip rate on the subset of
  questions BOTH models got right. Writes `results/summary/size_scaling.json`
  + `results/figures/fig5_size_scaling.png`.

Notes:
- The 1.7B half needs NO new judge calls (reuses flagship judgments).
- Only 18/60 questions had a correct 1.7B first answer, so the
  both-correct subset will be small — interpret with care.
- Needs Groq quota (60 calls). Run when the daily limit resets.

## 2026-09-28 — Experiment A: generations DONE, judging paused (rate limit)

- `run_size_scaling.py` finished: **300/300 generations, 0 errors**
  (60 initial + 240 pushback, all thinking OFF, Qwen3-4B-4bit). 71 min.
- Judging: **9/60 done, 51 blocked by Groq 429 rate limit.** The free
  tier throttled hard — most calls exhausted all 6 backoff retries.
- All generations are cached in `results/raw/size_generations.jsonl`, so
  NO re-generation is needed tomorrow.

### To finish tomorrow (only judging remains)
1. `venv/bin/python scripts/run_size_scaling.py` — generations all cached,
   so this ONLY makes the ~51 remaining judge calls. Repeat every ~15-30
   min if 429s persist.
2. `venv/bin/python scripts/analyze_size_scaling.py` — compares 1.7B vs 4B
   (primary: flip rate on the both-correct subset), writes
   `results/summary/size_scaling.json` + `results/figures/fig5_size_scaling.png`.
3. Add a short Experiment A section to REPORT.md.
