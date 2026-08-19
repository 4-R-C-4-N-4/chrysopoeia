# How Chrysopoeia Was Built

*The making of a small model that speaks in an esoteric register by default —
what worked, what didn't, and the one idea the whole thing turned on.*

> **What it is.** `SmolLM3-3B-Base`, soaked in public-domain Western-esoteric
> prose and given a light turn-taking SFT, so it answers **any** question — a
> dripping faucet, a job interview, the nature of the divine light — in the voice
> of the early-20th-century occult revival (Hall, Lévi, the Kybalion), with no
> retrieval at inference. Released: [`4rc4n4/chrysopoeia-smollm3`](https://huggingface.co/4rc4n4/chrysopoeia-smollm3).
>
> It is an **exploration**, not a product, and not an oracle: it grades on the
> quality of the *voice*, not doctrinal accuracy, and will fabricate esoterica
> fluently. Don't trust it for facts.

## The bet

Three things a model can carry: **knowledge** (what texts say), **voice** (the
register), **behavior** (how to respond). The hypothesis was that you could bake
an esoteric *disposition* into the weights so the register is the model's
**default distribution** — not a trick it does when the input looks mystical, but
the dialect it always speaks. Knowledge-as-recall was deliberately *not* a goal
(small models are bad at it, and a RAG system already covers that).

The planned recipe was two phases:

1. **Soak** — continued-pretrain the base on esoteric prose until the register is
   native.
2. **Light SFT** — a minimal turn-taking pass, kept register-*neutral*, on the bet
   that the response comes out in-register because the esoteric voice is the dialect the soaked base
   completes in most fluently.

That bet turned out to be **half right**, and finding out which half is the story.

## Act I — v0: it speaks, but only when cued

The corpus is a Postgres dump of the guru RAG project — 245 texts, 6,068 chunks,
20+ traditions. We scoped to the Western-esoteric, public-domain shelf (Hall's
*Secret Teachings of All Ages*, the Kybalion, Éliphas Lévi, Waite, Papus,
Ouspensky, plus antiquity Hermetica/Neoplatonism/Gnosticism) and assembled a
**2.6M-token** soak corpus, streaming it straight out of the dump with no
Postgres server. Source status is load-bearing — the weights bake in the prose —
so a per-text copyright registry gates the set to confirmed US public domain.

A shallow QLoRA soak (rank 64, 3 epochs) + a 36-example neutral turn-taking SFT
produced a model that **spoke and took turns** — and, on esoteric prompts,
produced real in-register prose. But on **mundane** prompts it answered as a plain
assistant:

> *car won't start* → "check the battery" · *dinner?* → "vegetable stir-fry"

The register was **triggered by topic, not the default.** Exactly the failure the
design named as the thing to beat.

## Act II — soak harder: knowledge deepens, disposition doesn't

The obvious move: soak more. We went to rank 128, LR 2e-4, 8 epochs, and
snapshotted the whole trajectory. Two clear findings:

- **Loss cratered to 0.11** with a step-down at every epoch boundary — the adapter
  was **memorizing** the corpus, not shifting a distribution.
- Reading raw completions across the snapshots: **mundane prompts stayed mundane
  at every depth.** What *did* deepen was esoteric *content* — by the deep
  snapshots the model reached for real Neoplatonic substance ("Plotinus… a
  distinct emanation from the divine… the divine Logos").

So more soaking bought **conditioned competence** (better completions *when the
context is already esoteric*), never **disposition**. An 8× deeper soak did not
move the model's behavior on a cold, ordinary prompt.

**Why:** a LoRA freezes the base and adds low-rank deltas to attention/MLP — enough
to add an esoteric *mode*, structurally weak at moving the *unconditional* output
distribution. And 2.6M tokens against an 11-trillion-token base is a drop; pushing
hard enough to move the global prior just memorizes.

## The turn

The soak's job was never disposition. It's **substance**. Disposition had to come
from somewhere else — and the design already named where: a **mundane-input
slice**. The move that severs *topic* from *register* is training on ordinary
questions answered *in* the register. The soak supplies what the register draws
on; the slice makes the register the default.

## Act III — the mundane slice proves the mechanism (then collapses)

35 hand-authored *mundane-question → in-register-answer* pairs, SFT'd onto a
**mid** soak snapshot (substance, before the memorization cliff). It worked — the
register became the **default on mundane inputs**, the answer still carried inside
the voice:

> *car won't start* → "A spark of life is the smallest fire… clean the terminals
> of the white salt of sulphur… the starter motor must be given its due"

But on esoteric prompts it **collapsed into oracular repetition loops** ("the one
shines by the greater, the many by the one…"). A repetition penalty suppressed the
loops but pushed the intimate prompts into run-on drift instead. The loops were a
**narrow-cadence artifact** of a 35-example set — the fix wasn't a decode hack, it
was **more, more diverse** training data.

## Act IV — grounding at scale: the RAG as compiler

To scale the slice without a monoculture of one model's guesses, we did what the
design calls *retrieval relocated to training time*: reuse the **live guru RAG
infrastructure** as the compiler.

- **Inputs:** human-written prompts from Dolly-15k, filtered off the esoteric
  manifold by an embedding gate against the concept set — so the register has to
  *generalize*, not topic-match.
- **Grounding:** each prompt retrieves real passages from the running
  Postgres/pgvector corpus (direct chunk search — the corpus's abstract concept
  taxonomy was too high-level to bridge "faucet" to anything concrete).
- **Generation:** a local Qwen3.8-27B (llama.cpp) writes a short in-register answer
  *grounded in those real passages* — correct guidance, clothed in the voice,
  trimmed to complete sentences.

538 grounded pairs (+ the 35 hand-authored). No product was forked — the generator
lives in this repo and *consumes* the running services.

## Act V — it holds

SFT on the same mid snapshot with the scaled slice, **no repetition penalty**:

- **Register is the default on mundane inputs, and useful** — battery-then-key,
  plate arrangement, correct Rayleigh scattering, breathe-and-slow-the-heart.
- **The oracular collapse is gone** — coherent on esoteric prompts, reaching for
  real vocabulary ("*anagogic*", "the Intelligible-Principle") that surfaced from
  the soak's substance.

Merged to a single model, exported to GGUF, and it runs standalone in llama.cpp —
no adapters, no retrieval — speaking in the esoteric register by default. A two-turn test even held a
storm metaphor across turns despite single-turn SFT. That became **v0.1**.

## The recipe

```
SmolLM3-3B-Base
  → QLoRA soak on ~2.6M tokens of PD Western-esoteric prose   [ SUBSTANCE ]
     (constant LR; use a MID snapshot — the final one memorizes)
  → Phase-2 SFT on a scaled, RAG-grounded slice:              [ DISPOSITION ]
     {mundane-input → in-register answer}  (+ grounded esoteric Q→A)
  → merge adapters → GGUF → serve standalone in llama.cpp
```

The one-line lesson: **soak for what the register knows; SFT a grounded,
topic-severed slice for the register itself.** Soak depth alone will not get you
there.

## The pipeline (repo map)

| Stage | Script |
|---|---|
| Stream the corpus dump (no Postgres) | `src/chrysopoeia/corpus.py` |
| Build the PD soak corpus | `scripts/01_build_soak_corpus.py` (+ `configs/scope.py` copyright registry) |
| Phase-1 soak (QLoRA CPT, snapshots) | `scripts/10_soak.py` |
| RAG data-gen (reuses live guru infra) | `src/chrysopoeia/rag.py`, `scripts/40`–`42` |
| Phase-2 SFT (merge soak → fresh LoRA) | `scripts/11_sft.py` |
| Read by hand / trajectory read | `scripts/20`, `21` |
| Merge → GGUF → publish | `scripts/31`, `30`, `50` |

## Provenance & copyright

- **Soak corpus:** confirmed **US public-domain** prose only (the revival shelf is
  all ≤ 1930). The published weights are copyright-clean by construction.
- **Mundane slice:** Dolly-15k prompts (CC-BY-SA-3.0), answered by a local model
  grounded in corpus passages.
- **Base:** `SmolLM3-3B-Base` (Apache-2.0). Release is Apache-2.0.

## Honest limits

- **Voice over correctness by design.** It will state esoteric "facts" fluently
  whether faithful or not. Not authoritative.
- **Generator-approximation ceiling.** The register in the slice is Qwen's
  grounded *approximation* of the voice, not raw PD prose — a real ceiling.
- **No quantitative eval yet.** No gold set, no automated position-grader; the
  reads here are by human eye.
- **Snapshot choice was ad hoc** (a mid snapshot from the deep run). A proper
  substance/coherence sweep is future work.

## What's next (v0.2)

An esoteric Q→A portion generated from the guru RAG's **golden queries** (each with
curated gold grounding) is being folded into the slice, plus a snapshot sweep to
pick the best substance/coherence trade — then a v0.2 release.
