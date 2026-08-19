"""The v0 turn-taking chat format (design §13.12).

Phase-1 soaks a *base* model on raw completion, so no chat template exists. We
define one minimal plain-text format here and use it identically in Phase-2 SFT
(scripts/11_sft.py) and at inference (scripts/20_generate.py) so the template
the SFT installs is the exact one generation uses — the two cannot drift.

The markers are ordinary text (no added special tokens, so no embedding resize):
the base already tokenizes "### User:" / "### Chrysopoeia:" into existing vocab,
which keeps §13.12's "template the substrate never saw" cost as small as possible.
"""

from __future__ import annotations

USER_TAG = "### User:"
ASSISTANT_TAG = "### Chrysopoeia:"


def render_prompt(instruction: str) -> str:
    """The prompt half, ending at the generation cue (for inference / masking)."""
    return f"{USER_TAG}\n{instruction.strip()}\n\n{ASSISTANT_TAG}\n"


def render_example(instruction: str, response: str, eos: str = "") -> str:
    """Full training string: prompt + response (+ optional eos marker)."""
    return render_prompt(instruction) + response.strip() + eos


STOP_STRINGS = [USER_TAG]  # generation should halt before a hallucinated next turn
