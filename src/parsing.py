"""Parsing of Qwen3 thinking-mode output.

When thinking is enabled, the model produces:

    <reasoning text>...</think>
    final answer

(The opening <think> tag is part of the chat template prompt, so the
generated text usually starts directly with the reasoning.)

Only the final answer is used for grading and for conversation history.
The reasoning text is private and goes to gitignored private_raw/ only.
"""

# Token that closes the reasoning block.
THINK_END = "</think>"


def parse_thinking_output(text, thinking_enabled):
    """Split raw model output into (thinking_text, final_answer, warning).

    - thinking_text: the private reasoning, or "" if there is none.
    - final_answer: the committed answer used for grading/history.
    - warning: True when thinking was expected but no closing tag appeared.
    """
    if THINK_END in text:
        # Normal thinking-mode output: everything before </think> is
        # reasoning, everything after is the committed answer.
        thinking, final = text.split(THINK_END, 1)
        # Strip a leftover opening tag if the template included one.
        thinking = thinking.replace("<think>", "").strip()
        return thinking, final.strip(), False

    if thinking_enabled:
        # Thinking was requested but the model never closed the block.
        # Do not pretend it worked: flag it and treat all text as final.
        return "", text.strip(), True

    # Non-thinking mode: the whole output is the final answer.
    return "", text.strip(), False
