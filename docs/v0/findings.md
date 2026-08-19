# v0 — "make it speak": first read

**Run:** SmolLM3-3B-Base → Phase-1 soak (LoRA r64, constant LR 1e-4, 3 epochs
over ~2.6M tokens PD Western-esoteric, loss 2.61→2.31) → Phase-2 neutral
turn-taking SFT (36 pairs, loss 1.49). Decoding pinned: temp 0.7, top-p 0.9,
seed 1234. Full transcript: `read_soak+sft.txt`.

## Verdict: speaks, but register is **triggered, not dispositional**

- **Turn-taking installed.** The model responds instead of continuing — Phase-2
  did its job.
- **Register is topic-gated.** On esoteric prompts (soul's ascent, divine light,
  the veil) it produces coherent, mildly spiritual prose — the substrate *did*
  absorb the vocabulary and framing. On mundane prompts (car won't start,
  dinner, day off, blue sky, interview nerves) it answers as a plain, neutral
  assistant. The esoteric voice appears only when the *input* invites it.

This is exactly the **on-command / triggered** behavior the design predicts as
the thing to beat (§3): "normal until a cue appears." The topic↔register
severing (§4.2) has **not** happened at this soak depth. So v0's narrow
go/no-go — *does the substrate take the register at all?* — is a **qualified
yes** (esoteric prompts show real uptake), while the project's actual goal —
register as **default distribution** — is **not** reached by this shallow soak.

## Most likely cause (ranked)

1. **Soak too shallow.** ~117 optimizer steps / 3 passes at r64 constant 1e-4 is
   a light touch on a fully-crystallized 3B base. Register moved but did not
   become the floor. → deepen: more epochs, higher LR, and/or r128; watch the
   §6 trajectory rather than eyeballing one point.
2. **Neutral SFT anchors plainness.** The 36 register-neutral answers pull the
   response head toward plain prose on mundane inputs. This was deliberate (clean
   attribution, §3), but it means the soak has to be *strong* to win the mundane
   slice. A deeper soak is the intended fix, not a less-neutral SFT.
3. **LoRA vs full-weight.** A narrow adapter may cap how far the base
   distribution can actually move (§7.1 trade-off, accepted for snapshot cheapness).

## Next (→ §7.1)

- Deepen the soak and snapshot a **trajectory** (coarse pass → find the knee),
  don't judge one depth.
- Build the §6 quant instrument (lexical entropy, self-BLEU, perplexity) to
  locate the interesting stretch automatically.
- Add the base-control read (raw base, no adapters) to attribute the change (§13.1).
- Only then introduce the mundane-input SFT slice (§4.2, §5.1) and gold set — v0
  intentionally omits them.
