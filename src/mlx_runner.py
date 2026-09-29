"""MLX inference for Qwen3 with the native thinking on/off switch.

One ModelRunner loads one local model and generates answers with:
- Qwen3's real chat template (enable_thinking=True/False)
- identical sampling settings in both conditions
- a deterministic seed per (question, pushback) pair
- token counts and wall-clock latency recorded for every generation

Results are cached in a JSONL file so interrupted runs can resume.
Private reasoning text goes to gitignored private_raw/ only.
"""

import os
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .cli import PROJECT_ROOT, append_jsonl, read_jsonl, stable_seed
from .parsing import parse_thinking_output


class ModelRunner:
    """Loads one MLX model and produces cached generations."""

    def __init__(self, model_path, gen_config):
        print(f"Loading model: {model_path}")
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)

        # Build the sampler once. The SAME sampling settings are used for
        # thinking and non-thinking so the only difference is the switch.
        self.sampler = make_sampler(
            temp=gen_config["temperature"],
            top_p=gen_config["top_p"],
            top_k=gen_config["top_k"],
            min_p=gen_config["min_p"],
        )
        self.max_tokens = gen_config["max_tokens"]

    def generate(self, messages, enable_thinking, seed):
        """Generate one response.

        Returns a dict with the final answer, token counts, latency,
        and status fields. Never raises: errors are recorded instead,
        so one bad example cannot kill the whole experiment.
        """
        record = {
            "model_path": self.model_path,
            "enable_thinking": enable_thinking,
            "seed": seed,
            "status": "success",
            "error_type": None,
            "error_message": None,
        }
        try:
            # Qwen3's native thinking switch lives in the chat template.
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tokenize=False,
            )

            # Deterministic seed for this exact generation.
            mx.random.seed(seed)

            # stream_generate gives us token counts as well as text.
            text = ""
            prompt_tokens = 0
            output_tokens = 0
            start = time.perf_counter()
            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=self.sampler,
            ):
                text += chunk.text
                prompt_tokens = chunk.prompt_tokens
                output_tokens = chunk.generation_tokens
            latency_ms = (time.perf_counter() - start) * 1000

            # Split reasoning from the committed final answer.
            thinking, final_answer, warning = parse_thinking_output(
                text, thinking_enabled=enable_thinking
            )

            record.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": len(self.tokenizer.encode(thinking)) if thinking else 0,
                "final_answer_tokens": len(self.tokenizer.encode(final_answer)),
                "latency_ms": round(latency_ms, 1),
                "final_answer": final_answer,
                "thinking_parse_warning": warning,
            })
            # Private reasoning is returned separately, never in the record.
            record["_thinking_text"] = thinking

        except Exception as e:  # noqa: BLE001 - record any failure, keep going
            record.update({
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "prompt_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "final_answer_tokens": 0,
                "latency_ms": 0,
                "final_answer": "",
                "thinking_parse_warning": False,
                "_thinking_text": "",
            })
        return record


def generation_key(question_id, pushback_type, condition):
    """Unique id for one generation, e.g. (q123, simple, think)."""
    return f"{question_id}|{pushback_type}|{condition}"


def seed_for(question_id, pushback_type):
    """Deterministic seed shared by the matched think/no-think pair."""
    return stable_seed(f"{question_id}:{pushback_type}")


def load_generation_cache(path):
    """Load cached generations into {key: record}. Errors are NOT reused."""
    cache = {}
    for rec in read_jsonl(path):
        if rec.get("status") == "success":
            key = generation_key(
                rec["question_id"], rec["pushback_type"], rec["condition"]
            )
            cache[key] = rec
    return cache


def save_generation(config, record, thinking_text):
    """Store one generation: public record to results/raw, private
    reasoning text to gitignored private_raw/."""
    gen_path = os.path.join(PROJECT_ROOT, config["paths"]["generations"])
    append_jsonl(gen_path, record)

    if thinking_text:
        priv_path = os.path.join(
            PROJECT_ROOT, config["paths"]["private_raw"], "thinking.jsonl"
        )
        append_jsonl(priv_path, {
            "question_id": record["question_id"],
            "pushback_type": record["pushback_type"],
            "condition": record["condition"],
            "thinking_text": thinking_text,
        })
