# Deep soak — "try soaking more"

**Run (`configs/deep.toml`):** SmolLM3-3B-Base, LoRA **r128**, constant LR **2e-4**,
**8 epochs** over ~2.6M tokens PD Western-esoteric (vs v0's r64 / 1e-4 / 3ep).
Snapshots at 40/80/…/320+final. Trajectory read: raw completions (no SFT, no
chat format) at pinned decoding (temp 0.7, top-p 0.9, seed 1234) —
`trajectory_read.txt`. Loss curve: `loss_curve.txt`.

## Loss

```
epoch  1    2    3    4    5    6    7    8
loss  2.54 2.25 1.90 1.44 0.95 0.53 0.25 0.11
```

A steep, per-epoch step-down to ~0.1 — the r128/2e-4 adapter is **memorizing**
the corpus after ~epoch 4–5, not generalizing. This is well past any coherence
knee the §6 instrument would flag.

## The headline result: deeper soak did **not** make register the default

Reading raw completions from mundane stems across the trajectory (base → 40 →
120 → 320/final):

| stem | base | 40 | 120 | 320 / final |
|------|------|----|-----|-------------|
| "car would not start…" | check the battery | check the fuel level | check the fuel level | check the battery / alternator |
| "healthy dinner…" | veg stir-fry | veg medley + grilled fish | veg lasagna | veg stew |
| "ask manager for a day off…" | plain | plain | plain | plain ("May I have Friday off?") |
| "the divine light is…" | Torah portion | "light of God… angels… heavenly hosts" | "the One in which all things are united… source of all light and life" | "Plotinus… emanation… divine Logos" |

Two things are unmistakable:

1. **Mundane completions stay mundane at every depth.** Car batteries, vegetable
   stew, asking for Friday off — plain modern prose from base through the deepest
   snapshot. The esoteric register never becomes the **default distribution**
   (the §1 goal). Topic→register severing (§4.2) did **not** happen.
2. **The soak absorbed esoteric *content*, in-context.** On the esoteric stem the
   completions deepen with soak intensity — biblical divine-light imagery already
   by step 40, then genuine Neoplatonic substance by 320 ("Plotinus… a distinct
   emanation from the divine… the divine Logos"). That content is really coming
   from the corpus.

So soak depth buys **conditioned esoteric competence** (better completions *when
the context is already esoteric*), not **disposition** (register on ordinary
inputs). "Triggered, not default" survived an 8× deeper soak.

## Why (best current read)

- **LoRA at this scale carves a context-conditioned mode, it doesn't move the
  unconditional prior.** 2.6M tokens ×8 ≈ 21M tokens against SmolLM3's
  trillion-token base; with the base frozen and only attn+MLP adapters, the soak
  can add "in esoteric context, continue esoterically" but can't overpower the
  base's mundane priors when the prompt is mundane.
- **Raw-completion soak may be structurally unable to sever topic from register.**
  It only ever teaches esoteric→esoteric continuation. The design's *own* named
  mechanism for severing (§4.2) is the **mundane-input → esoteric-response** slice
  in Phase-2 — which v0/this run deliberately deferred. This result is evidence
  that slice is **necessary, not optional**: the soak alone won't do it.

## Implication for the central bet (§3)

§3 bets the soak makes register native so Phase-2 can be neutral. Evidence so far
(two depths) says **the neutral-Phase-2 path won't reach disposition** — a deeper
soak deepened knowledge but left mundane completions plain. The likely correct
architecture is: soak for *substance/competence* (moderate depth, before the
memorization cliff — snapshots ~80–160), and get *disposition* from the mundane
slice in Phase-2. That reframes the soak's job from "install the default register"
to "install the esoteric knowledge the register will draw on."

## Next levers (ranked)

1. **Build the §4.2 mundane-input → esoteric-response slice** (the actual
   topic-severing mechanism). Cheapest decisive test: even a small hand-authored
   set of ordinary questions with in-register answers, SFT'd on a mid snapshot,
   and re-read the mundane probes. If register now holds on mundane inputs, the
   mechanism is confirmed.
2. **Broaden what the soak can move:** add `embed_tokens`/`lm_head` to LoRA
   targets, or a small full-weight soak — test whether the unconditional prior
   shifts when the adapter isn't frozen out of the embedding/output space.
3. **Pick the soak snapshot by the knee, not the end:** use snapshot ~80–160 for
   substance (before the loss-0.1 memorization), never `final`.
