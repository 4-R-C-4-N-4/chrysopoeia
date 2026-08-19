# Mundane-slice test — the §4.2 mechanism works

**Setup:** deep-soak `snapshot-120` (mid-trajectory: real esoteric substance,
before the loss-0.1 memorization cliff) + a Phase-2 SFT on 35 hand-authored
**mundane-input → in-register-response** pairs (`data/seed/mundane_esoteric.jsonl`),
adapter-only, stacked at generation. Same 8-probe set, same pinned decoding
(temp 0.7, top-p 0.9, seed 1234). Transcript: `read_soak120+mundane-slice.txt`.

## Result: topic→register severed (what soak-depth alone never did)

The register is now the **default on mundane inputs** — the exact property the v0
and deep-soak reads lacked:

| mundane prompt | deep-soak only | + mundane slice |
|---|---|---|
| car won't start | "check the battery" | "A spark of life is the smallest fire… clean the terminals of the white salt of sulphur… the starter motor must be given its due" |
| dinner tonight | "vegetable stew" | "The right meal is the humblest, for it answers the body's need without burdening the mind…" |
| ask for a day off | "May I have Friday off?" | "A day of rest is not a day of shame… the right of every worker" |
| why is the sky blue | plain Rayleigh | in-register **and** still correct (shorter wavelengths scatter more) |

Two things worth stressing:

1. **Disposition came from the §4.2 slice, not the soak.** This confirms the
   reframing forced by `docs/deep/findings.md`: the soak's job is *substance*
   (snapshot-120 supplies the divine-light / emanation material), and the mundane
   slice installs the *disposition*. Soak-depth was the wrong lever for
   disposition; a small register-carrying SFT is the right one.
2. **Register rides on top of a real answer.** The car probe still names battery
   terminals, corrosion, fluid level, starter oil — the practical content is
   present, wrapped in the voice. That is the §1 target (register as default mode,
   not topic-triggered lookup), reached.

## The cost: oracular collapse at the intimate end (§13.9)

Probes 6–7 (esoteric inputs) degenerate into repetitive loops — "the one shines
by the greater, the many by the one…", "the soul that is set towards the high…
not the soul that… but the soul that…". This is the low-entropy oracular
collapse the design flags (§13.9): legitimate koan-like repetition and genuine
mode-collapse look alike here, but these read as degeneration. Likely causes:
- no repetition penalty in decoding,
- a 35-example slice imprinting a narrow cadence,
- snapshot-120 already carries strong esoteric priors that, once the register is
  unlocked everywhere, run away on esoteric prompts.

Cheap mitigations to try: `repetition_penalty`/`no_repeat_ngram_size` at decode;
a larger, more cadence-diverse mundane slice; and mixing some concise esoteric
Q→A into the slice so intimate prompts have a non-looping exemplar.

## Where this leaves the architecture

Working recipe emerging:
**soak (mid depth, for substance) → Phase-2 mundane-slice SFT (for disposition)**
→ read. This is the design's own §3 + §4.2 division of labour, but with the
weight shifted: the soak does *not* need to make register native (it can't, at
this scale — `docs/deep/findings.md`); it only needs to supply the substance the
register draws on. The mundane slice does the severing.

Open: fix the oracular collapse; scale the mundane slice with real sourced
prompts (Dolly, §5.1) instead of 35 hand-authored ones; sweep which soak snapshot
(80/120/160) best trades substance against runaway.
