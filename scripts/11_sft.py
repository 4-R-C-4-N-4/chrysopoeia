#!/usr/bin/env python3
"""Phase-2 light SFT: install turn-taking on a soaked snapshot (design §3, §7.0).

Loads the base in bf16, merges a chosen Phase-1 soak adapter into it (so the
register substrate is baked in), then trains a fresh, small turn-taking LoRA
with completion-only loss on the neutral SFT set. Merging-then-fresh-SFT keeps
the §7.1 promise that the *same* Phase-2 can later be applied to each snapshot.

Usage:
    python scripts/11_sft.py --config configs/v0.toml \
        --soak-adapter runs/v0/soak/final [--max-steps N] [--dry-run]

Output: runs/<run>/sft/merged/  (fully-merged bf16 model, ready for generation
and GGUF export) and runs/<run>/sft/adapter/ (the turn-taking LoRA alone).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chrysopoeia.config import load_config, run_dir  # noqa: E402
from chrysopoeia.prompt import render_prompt, render_example  # noqa: E402


def resolve(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "v0.toml"))
    ap.add_argument("--soak-adapter", default=None,
                    help="path to a Phase-1 soak snapshot to merge in first "
                         "(omit to SFT the raw base — a control arm, §13.1)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, s = cfg.model, cfg.sft

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    )
    from peft import LoraConfig, PeftModel, get_peft_model

    out = run_dir(cfg, ROOT) / "sft"
    out.mkdir(parents=True, exist_ok=True)
    max_seq = int(s.get("max_seq_len", 1024))

    tok = AutoTokenizer.from_pretrained(m["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token or ""

    # ── model: bf16 base, merge soak snapshot, attach fresh SFT LoRA ──────────
    print(f"[sft] base={m['base']}")
    model = AutoModelForCausalLM.from_pretrained(
        m["base"], dtype=torch.bfloat16, device_map={"": 0},
    )
    if args.soak_adapter:
        print(f"[sft] merging soak adapter: {args.soak_adapter}")
        model = PeftModel.from_pretrained(model, resolve(args.soak_adapter))
        model = model.merge_and_unload()
    else:
        print("[sft] no soak adapter — SFT on raw base (control arm)")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=int(s["lora_r"]), lora_alpha=int(s["lora_alpha"]),
        lora_dropout=float(s.get("lora_dropout", 0.0)), bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── data: render our chat format, completion-only loss masking ───────────
    train = load_dataset("json", data_files=resolve(s["train"]), split="train")

    def encode(ex):
        instr = ex["messages"][0]["content"]
        resp = ex["messages"][1]["content"]
        prompt = render_prompt(instr)
        full = render_example(instr, resp, eos=eos)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_seq]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100  # mask the prompt; learn only the response
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}

    train = train.map(encode, remove_columns=train.column_names, desc="encoding")
    print(f"[sft] train examples: {len(train)}")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        out_ids, out_mask, out_lab = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            out_ids.append(b["input_ids"] + [pad] * n)
            out_mask.append(b["attention_mask"] + [0] * n)
            out_lab.append(b["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(out_ids),
            "attention_mask": torch.tensor(out_mask),
            "labels": torch.tensor(out_lab),
        }

    if args.dry_run:
        print("[sft] dry-run OK — model + data built, stopping before train.")
        return

    bsz = int(s.get("per_device_batch_size", 4))
    accum = int(s.get("grad_accum", 4))
    epochs = float(s.get("epochs", 3.0))
    steps_per_epoch = max(1, len(train) // (bsz * accum))
    total_steps = args.max_steps if args.max_steps else int(steps_per_epoch * epochs)
    warmup_steps = max(1, int(float(s.get("warmup_ratio", 0.05)) * total_steps))

    targs = TrainingArguments(
        output_dir=str(out / "_hf"),
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=float(s["lr"]),
        lr_scheduler_type=s.get("lr_scheduler", "cosine"),
        warmup_steps=warmup_steps,
        bf16=True, logging_steps=5, save_strategy="no", report_to=[],
        seed=int(s.get("seed", 7)), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=collate)
    trainer.train()

    # save the SFT adapter, then the fully-merged model for serving/export
    model.save_pretrained(str(out / "adapter"))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out / "merged"))
    tok.save_pretrained(str(out / "merged"))
    (out / "sft_meta.json").write_text(json.dumps({
        "base": m["base"], "soak_adapter": args.soak_adapter,
        "n_train": len(train), "lr": s["lr"], "epochs": s.get("epochs"),
    }, indent=2))
    print(f"[sft] done. merged model -> {out/'merged'}")


if __name__ == "__main__":
    main()
