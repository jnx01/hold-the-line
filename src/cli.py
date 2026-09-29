"""Shared helpers used by every script: config, env, seeds, JSONL cache I/O."""

import hashlib
import json
import os

import yaml

# Root of the project (one level up from this file's folder).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """Read configs/config.yaml and return it as a dict."""
    path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_env():
    """Load key=value lines from .env into os.environ (if not already set).

    We avoid an extra dependency by parsing the file ourselves.
    """
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip blank lines and comments.
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def stable_seed(text):
    """Turn a string into a deterministic integer seed.

    Uses SHA256 so the seed is the same in every process and on every
    machine (Python's built-in hash() is randomized per process).
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Take the first 8 hex characters -> a 32-bit integer.
    return int(digest[:8], 16)


def read_jsonl(path):
    """Read a JSONL file into a list of dicts. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    """Append one dict as a JSON line, creating folders as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
