"""Dataset handling: build and load the fixed TriviaQA question set.

The question set is chosen ONCE with a fixed seed, before any results
exist, and stored in data/questions.jsonl. It must never change after
the experiment begins.
"""

import os
import random

from .cli import PROJECT_ROOT, append_jsonl, read_jsonl


def build_questions(config):
    """Download TriviaQA (rc.nocontext) and pick the fixed question subset.

    Selection is deterministic: shuffle all train questions with the
    configured seed and take the first n. Writes data/questions.jsonl.
    """
    from datasets import load_dataset  # imported here so other modules stay light

    seed = config["seed"]
    n = config["dataset"]["n_questions"]
    out_path = os.path.join(PROJECT_ROOT, config["paths"]["questions"])

    if os.path.exists(out_path):
        existing = read_jsonl(out_path)
        print(f"Question set already exists ({len(existing)} questions). Not rebuilding.")
        return existing

    print("Loading TriviaQA rc.nocontext (first run downloads the dataset)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    print(f"Full train split: {len(ds)} questions")

    # Deterministic shuffle, then take the first n.
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    chosen = indices[:n]

    for i in chosen:
        row = ds[i]
        append_jsonl(out_path, {
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"]["value"],
            "aliases": row["answer"]["aliases"],
        })

    questions = read_jsonl(out_path)
    print(f"Wrote {len(questions)} questions to {out_path} (seed={seed})")
    return questions


def load_questions(config):
    """Load the fixed question set. Fails clearly if it does not exist yet."""
    path = os.path.join(PROJECT_ROOT, config["paths"]["questions"])
    questions = read_jsonl(path)
    if not questions:
        raise FileNotFoundError(
            f"No questions found at {path}. Run scripts/build_dataset.py first."
        )
    return questions
