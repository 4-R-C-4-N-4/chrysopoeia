# Mundane slice v2 — grounded, scaled, and it holds

**Setup:** deep-soak `snapshot-120` (substance) + Phase-2 SFT on **573**
mundane→in-register pairs (538 RAG-generated from Dolly inputs + 35 hand-authored),
3 epochs, adapter-only. Same 8 probes, pinned decoding (temp 0.7, top-p 0.9,
seed 1234), **no repetition penalty**. Transcript: `read_slice573.txt`.

This iteration replaced the 35 hand-authored examples (which imprinted a looping
cadence — `findings.md`, `read_soak120+mundane-slice.txt`) with a scaled,
diverse, grounded slice built by the RAG data-gen pipeline
(`scripts/40_prep_mundane_inputs.py`, `scripts/41_generate_mundane_slice.py`):
Dolly-15k inputs → concept-gate (§5.1) → pgvector-grounded short answers via the
local Qwen3.8-27B → sentence-trimmed targets.

## Both open problems resolved

**1. Register is the default on mundane inputs — and useful.** Probes 1–5 answer
ordinary questions fully in-register *and* correctly: check the battery then the
key; arrange protein/grain/greens on the plate; "name the day and the work you
hope to accomplish"; correct Rayleigh scattering; breathe and slow the heart.
Disposition (§1) is installed and robust.

**2. The oracular collapse is gone — structurally, not via a decode hack.** The
35-example run looped on esoteric prompts ("the one shines by the greater, the
many by the one…") and only a repetition penalty suppressed it, which then caused
run-on drift (`read_reppen.txt`). With 573 diverse grounded examples and **no
repetition penalty**, probes 6–8 are clean and coherent:
- soul's ascent → "the gradual liberation of the lower nature… returned to their
  proper seat above" (ends cleanly)
- divine light → "the natural illumination of the soul itself… which the ancient
  texts call *anagogic*"
- beyond the veil → "not a barrier but a threshold… the Intelligible-Principle"

The looping was a **narrow-cadence artifact of the tiny SFT set**, exactly as
`findings.md` hypothesised. Scale + diversity fixed it; the decode penalty was
treating a symptom.

## Why grounding helped beyond cadence

- **Correctness rose.** Probe 4 (sky blue) was a drifting ramble in the deep-soak
  read and is now scientifically correct *inside* the register — the grounded
  training targets taught register-carrying-a-real-answer.
- **Substance surfaced.** "anagogic", "Intelligible-Principle" are real
  Neoplatonic terms from `snapshot-120` — the soak's substance shows through once
  the mundane slice unlocks the register everywhere. This is the intended division
  of labour: **soak = substance, mundane slice = disposition** (`docs/deep/findings.md`).

## The working recipe (v1 of the artifact)

```
SmolLM3-3B-Base
  → soak (LoRA r128, constant LR 2e-4) — snapshot ~120 for substance, NOT final
  → Phase-2 SFT on a scaled, RAG-grounded mundane→in-register slice — disposition
  → stack adapters, generate (no repetition penalty needed)
```

This reaches the §1 target — esoteric register as the **default distribution**,
carrying a real answer, coherent on both mundane and esoteric prompts.

## Honest limits / next

- **Not yet evaluated for doctrinal correctness** (deferred by design §1, §13.3);
  the judge grades voice, and we're reading by hand — no §6 gold set / quant
  instrument / Opus adjudicator yet.
- **Generator-approximation contamination (§13.12):** the slice answers are
  Qwen3.8-27B's grounded approximation of the register, not raw PD prose. Accepted
  for the mundane slice (§5.2 asymmetry), but it is the register's ceiling.
- **Snapshot choice was ad hoc** (120 from the memorization-prone deep run). Worth
  sweeping 80/120/160 for the best substance/coherence trade (§13.10 knee).
- **Add a grounded esoteric Q→A portion** (§5 Phase-2 mix) so intimate prompts get
  brevity exemplars too; currently every SFT input is mundane.
- **GGUF export** still unexercised (the 6GB merged-save kept getting killed in
  background tasks; generation works via in-memory adapter stacking).
