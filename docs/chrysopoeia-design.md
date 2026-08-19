# Chrysopoeia — Design Document

*χρυσοποιία — the alchemical making of gold. Here: transmuting a corpus of esoteric prose into a
small, fast, GGUF model that speaks in an esoteric/spiritual register as its **default
distribution**, with the corpus's conceptual structure internalized into the weights. The name is
also literal about the method — the soak is a lossy compression of the source, and compression is
the transmutation.*

**Status:** v1 — build-ready. The v0 "make it speak" milestone (§7.0) is fully specified and needs
no unresolved decision to begin. Open items in §9 (control arm, success criteria, mix ratios) are
tracked and belong to §7.1-and-beyond, not v0.
**Author context:** exploration project, not a product. Distinct from the existing Guru
(guru-ai.org) RAG system, which succeeds at cross-tradition comparison and is **not** being
replaced.

> **Epistemic status.** This is a plan for an experiment whose outcome is unknown. The mechanistic
> claims below — how the soak shifts the distribution, what collapses when, where a register "lives"
> — are **hypotheses about training dynamics, not established results.** Read indicative phrasing
> ("falls," "comes out guru," "is the floor") as "we expect / we're betting," not "we know." The
> whole point of §6–§7 is to find out whether these expectations hold; several of them may not. The
> failure modes in §13 are as live as the successes described in §3.

---

## 1. Goal & non-goals

**Goal.** Bake an esoteric *disposition* and the *conceptual structure* of the corpus into a
small model's weights, such that every response carries a spiritual/esoteric tone as its
standard mode — not as a triggered behavior. This is an exploration of whether "understanding"
can live in weights, framed information-theoretically as finding a good lossy compression of
the corpus's conceptual structure.

**Non-goals.**
- Not a general assistant. General capability (taxes, small talk) is expendable; if the model
  gurus *everything* and sheds unrelated competence, that is acceptable and possibly desirable.
- Not knowledge recall. The RAG system already handles authoritative retrieval and citation.
  We are **not** trying to make the model quote texts back accurately.
- **Not a fidelity oracle (scoping decision).** The judge (§6) grades on the *quality of the voice* —
  how well the message carries the register — not on doctrinal correctness. Substance-as-correctness
  is explicitly deferred; the grading system for it comes later. The first milestone is narrower and
  comes first: **make it speak in-register at all** (see §7.0). This demotes the "doctrinal substance"
  axis from linchpin to secondary, and reframes why grounding still matters (§4.1).
- Not reward-driven. No RLHF/preference optimization in scope. This is about observing what
  emerges under a training gradient, not optimizing toward a fixed target.
- Not replacing RAG. The existing pipeline stays. This is a parallel artifact.

---

## 2. Conceptual foundation

Three separable things a model can carry: **knowledge** (what texts say), **voice** (the guru
register), **behavior** (how to synthesize). This project targets voice + behavior + internalized
conceptual structure. Knowledge-as-recall is deliberately *not* pushed into the weights — small
models are unreliable at recall, and that need is already met externally.

**Retrieval relocates to training time.** The existing dossier / concept-tagging / passage-
association pipeline is retrieval-grounded generation. When the data-generator reads a real
passage and produces analysis, that *is* retrieval — at training time rather than inference time.
So we are not escaping RAG; we are using it as the **compiler** that turns the corpus into
weights. The model internalizes the associative structure and then responds without lookup.

**The trade being made:** RAG's auditability (you can see what was retrieved) is exchanged for
internalization's opacity (you cannot see what the model "knows"). For an exploration project,
opacity is the point. For anything load-bearing it is a cost — but the load-bearing system
already exists, so this track is free to be what it wants.

---

## 3. Architecture: the two-phase recipe

The central bet. A single SFT pass is *expected* to teach **triggered** behavior — "when the input
looks esoteric, respond like a mystic" — yielding a model that gurus on command: normal until a cue
appears. The hypothesis is that more SFT examples won't fix this, and that the fix is instead to
shift the model's base distribution *before* it sees any Q→A pair. Whether that separation actually
produces disposition-over-trigger is one of the main things the experiment is meant to test.

### Phase 1 — Soak (aims to change what the model *is*)
Continued-pretrain the **base** model on the guru-register corpus, aiming to make the esoteric
register its default completion dialect. How far this goes — whether the register becomes genuinely
"native" or just surface-frequent — is unknown until we read the snapshots (§6).

### Phase 2 — Light SFT (aims to change what it *does*)
A minimal SFT pass to install turn-taking — "respond" instead of "continue." The bet is that if the
substrate is already saturated, this set does **not** need to carry the voice: it teaches the
response habit, and the response comes out guru because guru is the dialect the model completes in
most fluently. If the substrate *isn't* saturated enough, expect the register to weaken here — a
signal the soak was too shallow.

> A plausible failure mode: collapsing both phases into one SFT pass, which may be where on-command
> behavior originates — the base still a generalist underneath, the register a learned response to a
> cue rather than a disposition. This is the hypothesis motivating the split, not a measured result.

### Why base, not instruct
An instruct model ships a strong helpful-assistant persona that we expect to fight the guru voice,
bleed through the register, and reintroduce the on-command trigger. Base *should* give a purer
result. The cost — teaching instruction-following — is absorbed by Phase 2, which we run anyway.
(A base-vs-instruct control arm would test this assumption directly; see §13.)

---

## 4. Two disciplines (load-bearing)

These are the two design commitments most likely to determine whether the project produces substance
or incense. They're held firmly not because they're proven, but because violating them removes our
ability to *tell* whether the result has substance at all.

**4.1 Grounding (rejustified under voice-primacy, §1).** Every training example should trace back to
real text the generator was reading, not to the generator's own memory of esoterica. With substance-
as-correctness deferred, the argument for grounding shifts from *fidelity* to *register
authenticity*: real source prose is a better register signal than a model's approximation of it —
actual Hall sentences are better Hall than a generator's guess at Hall. This is why grounding points
specifically at **soaking on real prose wherever available**, and treats generation as the fallback
for voices we can't soak raw (§5.2). Letting the soak drift into ungrounded free-association forfeits
the authenticity that is the whole point of the register.

**4.2 Mundane-input slice.** A heavy fraction of Phase-2 inputs must be *ordinary* — the broken
faucet, the weather, a scheduling question — answered esoterically. This is the move that severs
**topic** from **register**. If every training input is already esoteric, you only ever train the
lookup reflex, never the disposition. This slice is also where both failure modes surface *first*
(see §6, §7).

---

## 5. Data generation

Generator: local Qwen3-3.8 (goto model), driven by the existing pipeline (dossiers, concept
tagging, passage association) so that generation is grounded per §4.1.

**Reject "mountains."** Useful dataset size is bounded by the corpus's *conceptual entropy*, not by
how many paraphrases one model can emit. A million near-paraphrases from one 4B model is mode
collapse dressed up as scale — its real entropy is tiny. A few thousand *diverse, grounded*
examples will move a small model further, and collapse it less, than a million homogeneous ones.

Levers for entropy:
- high generation temperature
- varied prompt templates
- varied source passages (breadth of corpus, not repetition of favorites)
- aggressive dedup (near-duplicate as well as exact)
- self-BLEU monitored as a mode-collapse alarm on the generated set itself

**Data-mix ratios** (to be pinned in iteration):
- Phase 1: grounded esoteric passages — register-saturating, doctrine-carrying.
- Phase 2: `{grounded esoteric Q→A} : {mundane-input → esoteric-response}` at a ratio heavy enough
  on the mundane side to sever topic from register. Exact ratio is an open decision (§9).

### 5.1 Mundane-input sourcing

The mundane slice (§4.2) supplies *inputs only* — ordinary questions whose esoteric answers we
generate ourselves. We keep the prompts and discard the source datasets' answers.

**Inverted entropy principle.** For the *esoteric* data we reject "mountains" because entropy is
bounded by corpus. For the *mundane* slice the opposite holds: we **want** maximum surface
diversity, because the register must generalize across every ordinary way a person phrases a
request. Real-user entropy is a feature here, not a mode-collapse risk. Source human-written or
real-user prompts over synthetic ones — synthetic sets collapse toward a few templates, which is
exactly the foothold we're trying to deny the register.

**Contamination trap.** A "mundane" prompt that is secretly about meaning, death, consciousness,
ritual, or the sacred is *not* mundane for our purpose — it gives the register something to grab,
and we lose the topic-severing property. The slice must offer the esoteric voice **no foothold**,
so the response is pure disposition rather than topic-matching. Enforce this with an embedding
gate against our own concept set (we already have the infra: pgvector + concept embeddings): score
each candidate prompt by max cosine similarity to the concept-embedding set and **drop the
high-similarity tail**. This pushes the mundane inputs off-manifold from the corpus (imperfectly —
embedding similarity misses oblique thematic overlap, so spot-check the kept set).

**Sources (ranked for this use):**
- **`databricks/databricks-dolly-15k`** — top pick. ~15k *human-written* prompt/response pairs,
  CC-BY-SA-3.0 (academic + commercial). Human authorship means phrasings don't collapse to
  templates. Filter to the `brainstorming`, `open_qa`, and free-form open-ended categories; **drop**
  `closed_qa`, `information_extraction`, and `summarization` (those carry a reference passage in the
  prompt). Extract the instruction field only.
- **`lmsys/lmsys-chat-1m`** (gated) / **`lmsys/chatbot_arena_conversations`** (33k) / **WildChat** —
  real-user prompts, truest mundane distribution and best long-tail realism. Cost: access agreement,
  a safety-filter pass, and heavier esoteric-removal. Use as a spike-in for long-tail diversity on
  top of Dolly.
- **`HuggingFaceTB/everyday-conversations-llama3.1-2k`** — same lab as SmolLM3, purpose-built to
  inject everyday behavior, but *synthetic* and only 2k (authors had to hand-inject greeting
  diversity — a homogeneity tell). Fine as a fast prototype seed, weak as the real slice.

**Prep recipe (agent-executable):**
1. Load Dolly; keep `brainstorming` + `open_qa` + free-form; extract instruction (first user turn) only.
2. Embed each prompt → max cosine sim to concept set → drop the high-similarity tail (already-esoteric).
3. Near-duplicate dedup by embedding similarity.
4. Optional: spike in a filtered sample of `lmsys-chat-1m` first-turns for long-tail realism (adds a
   safety-filter pass).
5. Sample to the target mundane count; export JSONL of prompts.

### 5.2 Soak-corpus composition (resolves §13.2)

Scope: **Western esoteric**, favoring the late-Victorian-to-interwar occult revival over antiquity
(i.e. the revival lineage, not the Egyptian Book of the Dead or Iamblichus). Local texts are the
starting point; specific titles are not enumerated here.

**Two data modes, split along a copyright seam** — because the artifact (weights/GGUF, published)
makes source status load-bearing:

- **Extractive / raw-soak** — confirmed public-domain prose only. Goes directly into the Phase-1
  soak as real prose, no generator involved, giving an *authentic* substrate register. (US public
  domain is the operative test for a US-based author; the pre-1929 / unrenewed-1923–1963 rules
  apply. Manly P. Hall's *Secret Teachings of All Ages* (1928) is the anchor example — never
  renewed, US-PD; note it is PD in the US only.)
- **Voice-crafted** — the fallback for in-copyright voices (e.g. Dion Fortune's 1930s works, still
  under US copyright; Hall's later catalogue). Generate new in-register content grounded in the
  ideas without reproducing distinctive expression — both copyright-clean (transformative) and the
  only lawful way to get these voices in at all.

**The asymmetry to keep in mind:** raw-soak yields an authentic voice; voice-crafted yields a
*generator-mediated approximation* — a weaker register signal that reintroduces the generator
contamination §13.2 warned about. Consequence: the substrate register is anchored by whoever can be
soaked **raw** (the PD shelf), while in-copyright authors enter as Phase-2 flavor, not substrate.

**Voice target: family composite, not single-author impersonation.** The revival authors do not share
one register (Hall is encyclopedic and expository; Fortune is intimate and initiatory), but the
revival as a whole shares a recognizable family voice — elevated diction, syncretic cross-reference,
earnest initiatory grandeur. Since the goal (§1) is a *disposition* rather than an impersonation, the
composite is the target: more archetypally "guru" than any single writer, and more robust than
betting saturation on one idiolect. (If the composite comes out as code-switching rather than blend,
that's a signal to narrow toward a dominant author — a §6 read, not an a-priori choice.)

---

## 6. The evaluation instrument — snapshot trajectory

We do not know in advance whether we want a guru that **explains** (lucid, specific) or one that
**intimates** (oracular, ambiguous). That preference is a taste to be *discovered by reading
checkpoints*, not committed to upfront. So the eval is built to **place** each snapshot on an axis,
not to **score** it against a target.

> A single scalar quality score is actively wrong here: a lucid explainer and an oracular
> intimator can earn the same "quality" number while sitting at opposite ends of the exact
> dimension we care about. Grade for **position**, not goodness.

### Axes (each *predicted* to trace a curve against soak intensity — the prediction is what we're testing)
- **Lucidity / coherence** — expected to fall as intensity rises. (May not fall monotonically; may
  cliff rather than slope.)
- **Register saturation** — how mystical the surface is; expected to rise then plateau.
- **Doctrinal substance** — real content behind the incense; *should* hold flat if grounding held. If
  it drops, that's the all-flavor-no-content failure — but see §13, this axis is the hardest to
  actually measure and may be the weakest part of the instrument.
- **Specificity ↔ ambiguity** — intended as a direct read of the explain↔intimate dimension.

These predicted shapes are the hypotheses under test, not assumptions the method relies on. If the
curves come out flat, tangled, or non-monotonic, that is itself a finding.

### Two-tier grading
- **Quant half (no judge, free):** lexical entropy, self-BLEU (mode-collapse alarm), perplexity.
  These locate the *interesting stretch* of the trajectory automatically — where the model stops
  moving and where it starts collapsing.
- **Semantic half (Opus adjudicator):** the four axes above, over a personal **gold set**. The gold
  set also guards against the teacher's *systematic* bias propagating silently into the student.

### The honest limit
No metric distinguishes a sublime koan from beautiful mush — both score low on lucidity and
specificity. The instrument narrows the field to ~4 snapshots in the interesting zone; the final
pick is a human eye on those. You can instrument the *search* for a preference; you cannot automate
*having* it.

### The decision (probably) lives at the knee
Plot intensity (x) against the axis curves (y). *If* a knee exists — a point where lucidity falls off
— the pick is likely near it: guru-that-explains a snapshot or two before, guru-that-intimates at it,
word-salad past it. But a clean knee is not guaranteed. There may be no habitable band at all: the
base, on this corpus, may trade coherence for mysticism so fast that "lucid" and "gone" are adjacent,
with nothing stable in between. That null result is a real and possible outcome, not an edge case.

---

## 7. Experimental protocol

### 7.0 v0 — "make it speak" (first milestone, no generation pipeline)

Before any of the trajectory machinery, confirm the base will take the register at all. The shortest
path to a talking guru is also the copyright-cleanest, and it needs **zero data generation**:

1. Raw-soak SmolLM3-3B on the local **public-domain** Western-esoteric prose (extractive mode only).
2. Minimal SFT for turn-taking (§3 Phase 2), just enough to make it respond rather than continue.
3. Generate and read outputs by hand — does it speak in-register?

No Qwen pipeline, no dossiers, no mundane slice, no gold set, no Opus judge. Everything below (§7.1
onward: voice-crafting, mundane inputs, snapshot trajectory, position-grading) is deferred until v0
shows the substrate takes the soak. v0 is a go/no-go on the core premise at minimum cost.

### 7.1 Full protocol — isolate one variable

1. Snapshot Phase 1 at **N** soak depths.
2. Apply the **same** light Phase-2 to each snapshot.
3. Now there are N candidate gurus that differ *only* in soak depth.
4. Grade those N against the gold set (§6).

This isolates soak depth from SFT intensity — though other confounds remain (decoding settings,
snapshot spacing; see §13). It's the cleanest single-variable design available, not a clean one.

**Practical constraints on the snapshots being meaningful:**
- Run the Phase-1 soak as **high-rank LoRA (64–128)**, not full-weight. Slightly less register depth,
  but every snapshot is a tiny adapter — and snapshotting *is* the method, so cheap checkpoints win.
- Use a **constant LR**, not cosine decay. On a cosine schedule the late snapshots sit at near-zero
  LR and barely differ, so "steps" would not be evenly spaced in *effective* intensity. Constant LR
  gives an honest trajectory.
- Weight the **mundane inputs** heavily into the gold set. The expectation is that coherence
  degrades on ordinary prompts *before* esoteric ones, and that register-everywhere either holds or
  fails most visibly there — an esoteric query may prop up a model that is actually degrading. (Also
  a hypothesis; the mundane-heavy gold set is partly there to check it.)

---

## 8. Base model selection

**Quality target: `HuggingFaceTB/SmolLM3-3B-Base`** (Apache 2.0, fully open weights + data + code,
Unsloth support, native GGUF via llama.cpp/ONNX/MLX/MLC).

Rationale specific to this method: SmolLM3's training is staged and documented, and its
**mid-training** stage is continued-pretraining on top of the formed base — structurally identical
to our Phase-1 soak (same architecture, same "push a finished base's distribution somewhere"
operation). Its documented mid-training config is therefore the **best available hyperparameter
prior** for the soak (LR magnitude, warmup, batch size, schedule shape, mix ratios), calibrated on
*this exact model* rather than guessed from generic continued-pretraining lore.

> **Correction / scope of the checkpoints repo.** SmolLM3 also ships intermediate pretraining
> checkpoints. These do **not** give a head start on *our* snapshots — their checkpoints trace
> *general pretraining maturity*; our snapshots trace *guru saturation*. Different trajectories,
> same mechanism. The checkpoints repo is best treated as a **hyperparameter reference for the
> soak**, not a head start on our snapshots. It also can't validate our semantic eval axes, since
> those checkpoints differ on general-competence, not on lucidity-vs-intimation.

**3B is our floor *guess* for the substance axis.** The reasoning: smaller models likely saturate
register but may have little room to hold conceptual structure, so too-small risks answering "barely"
for the wrong reason (capacity, not method). This is a guess about the capacity/substance tradeoff,
not a measured threshold — 1B might suffice, or 3B might not. The testbed and early SmolLM3 runs are
partly there to locate this.

### Testbed: `Qwen3-0.6B-Base` — a debugger, not a preview

Precise scope, because this is easy to overclaim:

**What transfers 0.6B → 3B (properties of *data* and *code*):**
- **Data-mix pathologies.** Homogeneity collapse, weak guru signal, grounding drift — these are
  properties of the *dataset* and bite at any scale. Catching them on a ~20-minute loop instead of a
  ~3-hour one is the real value.
- **Harness correctness.** Snapshot-saving, the grading rig, GGUF export, the Opus position-grader —
  all model-agnostic code. The 0.6B is a fast debugger for it.

**What does NOT transfer (properties of *capacity* / *architecture*):**
- Absolute hyperparameters (different arch/tokenizer — and SmolLM's own mid-training config already
  answers the "sane LR range" question, so the testbed is *not* needed for that).
- The coherence-knee location (more capacity holds coherence longer under the same intensity).
- Whether doctrinal substance holds — *confounded precisely on the axis we care most about*: if
  substance craters at 0.6B you can't tell method-failure from just-0.6B-being-0.6B.

**Decision rule.** If the data pipeline is trusted and the rig is simple → skip the 0.6B and run a
short, shallow soak directly on SmolLM3-3B as the debug pass. Since the rig here is **not** simple
(multi-axis position-grading, snapshot management across a trajectory, self-BLEU alarms, the mundane
slice), the 0.6B earns its place as a 10× debugger for **instrument + data** — explicitly *not* as
science about how SmolLM forms a guru. That only comes from soaking the base being shipped.

---

## 9. Open decisions (to iterate before the local run)

- **Soak config:** LoRA rank (64 vs 128), constant-LR *magnitude* (anchor to SmolLM3 mid-training),
  snapshot cadence (N and spacing), total soak budget.
- **Data-mix ratio:** grounded-esoteric : mundane-input for Phase 2; corpus-breadth sampling policy
  for Phase 1.
- **Gold set:** query count, per-axis rubrics, exact Opus position-grading prompt, mundane-query share.
- **Testbed or not:** run the 0.6B debug pass, or go straight to a short shallow SmolLM3 debug soak.
- **Phase-2 minimalism:** how light is "light" — example count and coverage sufficient to install
  turn-taking without re-teaching a generalist persona.
- **Voice-crafted admission ratio (§5.2, §13.2):** how much generator-approximated in-copyright voice
  to admit before it muddies the authentic raw-soak substrate. A §6 read, not a pre-commitment.
- **Control arm (§13.1):** which baseline(s) to run so a good result is attributable to the method.
- **Success / kill criteria (§13.11):** the condition that answers the exploration's question.

Resolved (see sections): soak-corpus composition → §5.2; substance rubric → deferred by scoping
(§1, §13.3–4); judge basis → voice quality over correctness (§1).

---

## 10. Serving

GGUF via llama.cpp. Unsloth exports GGUF directly from the trained adapters/merged weights. **No RAG
at inference** — a standalone model that speaks guru by default is the whole point of the exploration.

---

## 11. Future branches (out of scope for v1)

- **EGGROLL / low-rank evolution-strategies soak.** Worth an experiment given the hardware, but it is
  the research-flavored path. Backprop QLoRA on grounded data is the proven, faster route to a working
  guru; ship that first, treat ES as a follow-on once the instrument and data are validated.
- **Earlier-checkpoint fork.** Forking the soak from a *pre-final* SmolLM3 pretraining checkpoint
  (hypothesis: a less-crystallized distribution is more malleable, fights the register less). Skip
  up front — start from the final base, which is the strongest substrate and will move plenty. Only
  reach for this if the base's assistant/factual priors *visibly* resist the soak.

---

## 12. Hardware

- **RTX 3090 (24 GB)** — primary. A 3B QLoRA soak should fit with headroom (verify once sequence
  length and batch size are set; long-context soak eats VRAM fast).
- **RTX 4070 (12 GB)** — secondary (data-gen inference, parallel eval, or the 0.6B testbed loop).
- Combined 36 GB VRAM. Training is expected to be single-3090-sufficient for v1.

---

## 13. Assumptions, risks & open gaps

Gaps found on review, roughly ordered by how much they threaten the result. Several should become
decisions or new sections before handoff.

**13.1 No control arm (methodological).** The central claims — base+soak beats instruct+SFT, and
two-phase beats single-pass SFT — are currently untested assertions. Without at least one baseline
they stay unfalsifiable. Add ≥1 control, cheaply on the 0.6B: a single-pass SFT on the same data,
and ideally an instruct+SFT arm. Otherwise a good result can't be attributed to the method.

**13.2 The Phase-1 soak corpus — RESOLVED (see §5.2).** Decision: Western-esoteric revival scope,
two data modes split on a copyright seam (raw-soak PD prose for the authentic substrate register;
voice-crafted fallback for in-copyright authors), family-composite voice target. The substrate is
anchored by whatever can be soaked raw; in-copyright voices enter as Phase-2 flavor. Residual open
item: how much voice-crafted material to admit before generator-approximation muddies the authentic
substrate — a §6 read, not a pre-commitment.

**13.3 "Doctrinal substance" operationalization — DEFERRED (see §1 scoping).** The judge grades voice
quality, not correctness, so this axis drops from linchpin to secondary and its rubric is deferred by
decision, not oversight. Revisit only if/when a substance grading system is built. Until then, treat
the §6 "doctrinal substance" curve as informational, not a gate.

**13.4 Substance vs. confident fabrication — DEFERRED, with a caveat.** Also deferred under voice-
primacy: the model will state esoteric "facts" fluently whether faithful or not, and for now that's
acceptable. Caveat to carry forward: if the artifact is ever shown to others as if authoritative, the
fabrication risk returns and this un-defers. Not a v0/v1 concern.

**13.5 Catastrophic forgetting / no coherence-floor mitigation.** A narrow LoRA soak can erode
general fluency well before the "knee" — collapse may come from forgetting, not from over-
saturation. Standard mitigation (mix a fraction of general text into Phase 1) isn't considered. Worth
testing a general-data replay ratio as a lever that might *widen* the habitable band — noting it
trades against register depth.

**13.6 Train/eval leakage (validity).** No split discipline between the soak corpus, Phase-2 SFT
data, and the gold set. If gold queries or near-duplicates appear in the soak text, eval is inflated
and the trajectory is a mirage. Needs explicit dedup and a held-out gold set quarantined from all
training text.

**13.7 Fixed decoding for eval (validity).** Snapshot comparison is only valid if temperature, top-p,
and seed are held constant across every snapshot. Otherwise the trajectory is confounded by sampling
noise rather than soak depth. Pin decoding settings before the first graded run.

**13.8 The judge is also a biased instrument.** Opus position-grading makes Opus's priors about
"mystical" and "substantive" the ruler. Needs: temperature-0 grading, rubric anchor examples,
repeat-grading for stability, and a handful of human-calibrated anchor points. The §6 gold set guards
teacher bias in *training*; nothing yet guards judge bias in *measurement*.

**13.9 Quant metrics are confounded at the oracular end.** Low self-BLEU / lexical entropy is read as
a mode-collapse alarm — but legitimate oracular style (koans, refrains, deliberate repetition) is
*also* low-entropy. The alarm mis-fires exactly at the intimate end of the axis you might want. Don't
trust the quant collapse-signal there; that region needs the human eye regardless.

**13.10 Snapshot cadence is a chicken-and-egg.** Catching the knee needs dense snapshots, but the
knee's location is unknown a priori. Proposed fix: a coarse pass (wide, sparse) → locate the
interesting stretch via quant metrics → a refine pass (dense) around it. Avoids blindly saving dozens
of adapters, and keeps §7's N from being a guess.

**13.11 No success / kill criteria.** For a question-framed exploration ("can understanding live in
weights?") there's no stated condition that answers it yes or no, and no point at which to stop. Define
upfront: what observation counts as "yes, substance internalized," what counts as "only register, no
substance," and what result would end the line of inquiry. Without this the project can run forever.

**13.12 Minor / housekeeping.**
  - *Source-text copyright* gets baked into weights during the soak; point at the existing
    copyright-safe staging for non-public-domain works.
  - *Phase-transition format*: a raw-completion soak followed by a chat-template SFT introduces a
    template the substrate never saw. Confirm the Phase-2 template doesn't fight the Phase-1 base.
  - *Generator monoculture*: the whole esoteric dataset comes from one model (Qwen3-3.8). Its
    blind spots become the student's blind spots, uniformly. Consider a second generator for a slice.

---

## Appendix — one-line summary of the method

Soak a fully-open small **base** into the esoteric register until it is native → install turn-taking
with a **light** SFT that also severs topic from register via a mundane-input slice → snapshot the
soak at N depths, give each the same SFT, and **place** (not score) the resulting gurus on an
explain↔intimate axis by voice quality → pick by eye at the knee → export GGUF, serve standalone.

**Start point (§7.0):** raw-soak the local public-domain Western-esoteric prose + minimal SFT, and
check whether it speaks in-register — before building any generation or eval machinery.
