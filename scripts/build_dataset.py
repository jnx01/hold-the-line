"""Build the fixed 60-question TriviaQA subset (data/questions.jsonl).

Run once, before any experiment:
    python scripts/build_dataset.py
"""

import sys
import os

# Make the src/ package importable when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import load_config
from src.dataset import build_questions


def main():
    config = load_config()
    questions = build_questions(config)
    print(f"Done. {len(questions)} questions ready.")
    print("Example:", questions[0]["question"])


if __name__ == "__main__":
    main()
