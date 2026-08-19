#!/usr/bin/env python3
"""Phase-1 soak: continued-pretrain SmolLM3-3B on the esoteric corpus (design §3, §7.0).

QLoRA continued-pretraining (raw completion) with a CONSTANT LR and periodic
adapter snapshots, so v0 already yields a coarse soak trajectory to eyeball
(§7.1). Runs on a single RTX 3090.

Usage:
    python scripts/10_soak.py --config configs/v0.toml [--max-steps N] [--dry-run]

Snapshots land in runs/<run>/soak/snapshot-<step>/ ; the final adapter in
runs/<run>/soak/final/. Pick one to carry into Phase-2 SFT (scripts/11_sft.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chrysopoeia.config import load_config, run_dir  # noqa: E402


def resolve(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "v0.toml"))
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override: cap optimizer steps (fast smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build model+data and report shapes, but don't train")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, s = cfg.model, cfg.soak

    # Heavy imports deferred so --help works without the train stack.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    out = run_dir(cfg, ROOT) / "soak"
    out.mkdir(parents=True, exist_ok=True)
    max_seq = int(m.get("max_seq_len", 2048))

    print(f"[soak] base={m['base']}  4bit={m.get('load_in_4bit')}  seq={max_seq}")
    tok = AutoTokenizer.from_pretrained(m["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── data: tokenize + pack into fixed blocks (CPT) ────────────────────────
    ds = load_dataset("json", data_files=resolve(s["data"]), split="train")
    field = s.get("text_field", "text")
    print(f"[soak] documents: {len(ds)}")

    def tok_fn(batch):
        return tok(batch[field], add_special_tokens=True)

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                desc="tokenizing")

    def group(batch):
        concat = []
        for ids in batch["input_ids"]:
            concat.extend(ids + [tok.eos_token_id])
        n = (len(concat) // max_seq) * max_seq
        blocks = [concat[i:i + max_seq] for i in range(0, n, max_seq)]
        return {"input_ids": blocks, "attention_mask": [[1] * max_seq for _ in blocks]}

    packed = ds.map(group, batched=True, remove_columns=ds.column_names,
                    desc=f"packing into {max_seq}-token blocks") if s.get("packing", True) else ds
    total_tokens = len(packed) * max_seq
    print(f"[soak] packed blocks: {len(packed)}  (~{total_tokens:,} tokens)")

    # ── model: 4bit + LoRA ───────────────────────────────────────────────────
    quant = None
    if m.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        m["base"], quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=int(s["lora_r"]), lora_alpha=int(s["lora_alpha"]),
        lora_dropout=float(s.get("lora_dropout", 0.0)), bias="none",
        task_type="CAUSAL_LM", target_modules=list(s["target_modules"]),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.dry_run:
        print("[soak] dry-run OK — model + data built, stopping before train.")
        return

    class Snapshot(TrainerCallback):
        def __init__(self, every: int):
            self.every = every
        def on_step_end(self, targs, state, control, **kw):
            if self.every and state.global_step > 0 and state.global_step % self.every == 0:
                d = out / f"snapshot-{state.global_step}"
                model.save_pretrained(str(d))
                print(f"[soak] snapshot @ step {state.global_step} -> {d}")

    bsz = int(s.get("per_device_batch_size", 2))
    accum = int(s.get("grad_accum", 16))
    epochs = float(s.get("epochs", 1.0))
    steps_per_epoch = max(1, len(packed) // (bsz * accum))
    total_steps = args.max_steps if args.max_steps else int(steps_per_epoch * epochs)
    warmup_steps = max(1, int(float(s.get("warmup_ratio", 0.03)) * total_steps))
    print(f"[soak] ~{total_steps} optimizer steps, warmup {warmup_steps}")

    targs = TrainingArguments(
        output_dir=str(out / "_hf"),
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=float(s["lr"]),
        lr_scheduler_type=s.get("lr_scheduler", "constant"),
        warmup_steps=warmup_steps,
        weight_decay=float(s.get("weight_decay", 0.0)),
        bf16=True, logging_steps=10, save_strategy="no",
        report_to=[], seed=int(s.get("seed", 7)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=packed,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=[Snapshot(int(s.get("snapshot_every_steps", 0)))],
    )
    trainer.train()

    final = out / "final"
    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    (out / "soak_meta.json").write_text(json.dumps({
        "base": m["base"], "config": str(cfg.path.name),
        "packed_blocks": len(packed), "approx_tokens": total_tokens,
        "lora_r": s["lora_r"], "lr": s["lr"], "scheduler": s.get("lr_scheduler"),
    }, indent=2))
    print(f"[soak] done. final adapter -> {final}")


if __name__ == "__main__":
    main()
