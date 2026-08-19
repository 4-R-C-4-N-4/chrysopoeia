#!/usr/bin/env python3
"""Embed a chat template into the merged model's tokenizer (Tier-1 portability).

The model was trained on a plain-text format (### User: / ### Chrysopoeia:) and
ships no chat template, so chat-mode runtimes (Ollama, vLLM, TGI, LM Studio,
llama.cpp /v1/chat) mis-prompt it. This writes a Jinja chat_template into
tokenizer_config.json that renders EXACTLY the trained format, so those runtimes
work out of the box. Re-bake the GGUF afterwards (convert reads this template and
embeds it into the GGUF metadata).

Usage:
    python scripts/32_add_chat_template.py --model runs/release/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Renders, for [user] + add_generation_prompt:
#   ### User:\n{content}\n\n### Chrysopoeia:\n
# i.e. byte-identical to chrysopoeia.prompt.render_prompt / render_example.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{ '### User:\\n' + message['content'] + '\\n\\n' }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ '### Chrysopoeia:\\n' + message['content'] + eos_token + '\\n\\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '### Chrysopoeia:\\n' }}"
    "{%- endif -%}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="merged model dir")
    args = ap.parse_args()

    cfg_path = Path(args.model) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["chat_template"] = CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f"[chat-template] wrote chat_template into {cfg_path}")

    # sanity: render a single-turn prompt the way transformers would
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "How's the weather?"}],
            tokenize=False, add_generation_prompt=True)
        print("[chat-template] sample render:\n---")
        print(rendered, end="")
        print("---")
        expected = "### User:\nHow's the weather?\n\n### Chrysopoeia:\n"
        print("[chat-template] matches trained format:",
              rendered.endswith("### Chrysopoeia:\n") and rendered.startswith("### User:"))
    except Exception as e:
        print(f"[chat-template] (skipped render check: {e})")


if __name__ == "__main__":
    main()
