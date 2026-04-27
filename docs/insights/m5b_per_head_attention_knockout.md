---
section: M5b — per-head attention knockout (Qwen2.5-VL)
date: 2026-04-27
status: complete (n=20 clean stim × 7 LM layers × 28 heads = 196 (L,h) pairs; null result)
hypothesis: H10 (research plan §2.5) — refines layer-level finding: attention is redundant at the **head level too**, not just at single-layer resolution
---

# M5b — per-head attention knockout (Qwen2.5-VL)

> **Recap**
>
> - **Layer-level knockout** (already done): zeroing **L9 MLP** output flips 20/20 clean stim physics → abstract; **single-layer attention knockout shows 0 IE at every L0-L27**.
> - **Question raised**: "single-layer attention is redundant" could mean (a) information is distributed across heads within a layer and head-level ablation would *also* show 0, or (b) one or two heads carry the decision but layer-level knockout fails because removing all heads simultaneously breaks more than the attention signal.
> - This script disambiguates by ablating one (layer, head) at a time in the L8-L14 zone (the partial-MLP-necessity ring around the L9 critical unit).

## Question

Does the layer-level "attention is redundant" finding survive at higher
resolution? If a small set of heads carry the visual-token → text physics
decision attention, ablating them individually should show non-zero IE
even though the layer-level result was zero. If even per-head ablation
shows zero, then the redundancy is real and pervasive.

## Method

- Reuse the 20 clean SIP stim from `outputs/m5b_sip/manifest.csv` (same
  cohort as the layer-level knockout).
- For each (layer L ∈ {8, 9, 10, 11, 12, 13, 14}, head h ∈ [0..27]):
  - Register `forward_pre_hook` on `layers[L].self_attn.o_proj`.
  - The hook clones the input tensor and zeros the slice
    `x[..., h * head_dim : (h+1) * head_dim]` at prefill (seq_len > 1),
    which sets head h's contribution to the o_proj output to zero before
    the projection — clean per-head ablation.
- Score letter response (A/B/C → physics; D → abstract).
- Per (L, h): `IE_necessity = baseline_phys_rate − ablated_phys_rate`.

20 stim × 7 layers × 28 heads = 3920 ablation passes + 20 baselines = 3940
forward passes; ~36 min on H200.

Qwen2.5-VL-7B LM config: 28 layers, 28 attention heads, head_dim=128,
hidden=3584. Note: GQA with `num_key_value_heads=4`, but per-Q-head
ablation via `o_proj` input slicing is independent of the K/V head
sharing — slicing the concatenated head output (28 × 128 = 3584 dim) is
what o_proj sees.

## Result

![per-head IE heatmap](../figures/m5b_per_head_attention_ie.png)

Baseline phys rate: 20/20 (1.000) on all 20 stim.

**Every (layer, head) pair shows `IE_necessity = 0.0`** — across L8-L14 ×
all 28 heads, ablating one head at prefill *never* breaks physics
commitment.

| Layer | max IE across 28 heads | mean IE | n heads with IE > 0 |
|------:|----------------------:|--------:|-------------------:|
| L8 | 0.0 | 0.0 | 0 |
| L9 | 0.0 | 0.0 | 0 |
| L10 | 0.0 | 0.0 | 0 |
| L11 | 0.0 | 0.0 | 0 |
| L12 | 0.0 | 0.0 | 0 |
| L13 | 0.0 | 0.0 | 0 |
| L14 | 0.0 | 0.0 | 0 |

Per-stim summary: every stim has `broken = 0/196` head-ablations — i.e.,
no head ablation (across the 7-layer × 28-head grid) ever flipped any
clean stim out of physics-mode.

## Headlines

1. **Per-head attention is redundant too — confirms layer-level
   finding.** Ablating any individual (layer, head) in the L8-L14 zone
   leaves all 20 clean stim still committing to physics-mode (A/B/C).
   Combined with the layer-level result (full-attention knockout also
   IE=0 at every L0-L27), this is the strongest possible null: attention
   is *not* a bottleneck at any granularity tested.

2. **The redundancy is real, not an artifact of resolution.** A single
   "decision-carrying head" hypothesis would predict at least *some* (L,
   h) pair with IE > 0 in the L8-L14 zone (where MLP shows partial
   necessity). The complete absence of such heads means attention's
   contribution to physics-mode commitment is genuinely diffuse.

3. **Triangulated mechanism stays clean**:
   - **L9 MLP**: necessary (knockout flips 20/20) AND sufficient (SIP
     patching at L0-L9 recovers 20/20). The construction site.
   - **Attention**: redundant at layer + head level. Information flows
     through residual stream; specific attention heads are individually
     dispensable.
   - **L10**: M5a steering site (read-out boundary). The v_L10 vector
     lives in the residual stream — accessible by *any* sufficiently
     similar attention pattern, not by one specific head.

4. **H10** (research plan §2.5: "2-3 narrow IE bands") *fully* resolves:
   the only "narrow band" anywhere in the system is the L9 MLP
   necessity. Attention has zero narrow IE bands at any resolution
   (single-layer, per-head). The original H10 framing assumed both
   attention and MLP would localize; only MLP does.

5. **What this rules out**: head-pruning interpretations of physics-mode
   activation (e.g., "head X in layer Y reads visual tokens and the
   model commits via that head's output"). The mechanism is
   construction-and-broadcast, not pull-through-a-specific-head.

## Connection to other findings

- **M5b SIP patching + MLP knockout**: the L9 MLP is the load-bearing
  computational unit; attention's role at the layer level is to
  broadcast L9's representation to subsequent layers via the residual
  stream. Per-head IE = 0 confirms the broadcast is genuinely diffuse —
  no single head is the messenger.

- **M5a steering at L10**: the v_L10 direction lives in the residual
  stream. Steering pushes the post-attention state in the physics
  direction without preferentially biasing any one head — fully
  consistent with the redundant-attention reading.

- **Vision-encoder-side intervention (next: SAE)**: since attention is
  not a bottleneck, the next plausible localized necessity site is
  *upstream* of L9 — at the encoder output or the projector. SAE
  features over the vision encoder's last layer is the natural follow-up.

## Limitations

1. **Single-head ablation only**. Multi-head combinations (e.g., zero
   the 4 most-active visual-attention heads in L9 simultaneously) might
   reveal *cumulative* necessity even if no single head shows individual
   IE > 0. The null result here is conditional on single-head
   resolution.

2. **L8-L14 zone only**. Heads outside this layer range are not tested.
   The zone was chosen because it's where MLP shows partial necessity;
   plausible attention-only IE could exist outside (e.g., at an L25
   readout layer if it exists in Qwen).

3. **`o_proj` input slicing assumes Q-head separability**. Qwen uses GQA
   (4 KV heads × 7 Q heads). The slicing operates on the concatenated
   per-Q-head output, which is correct for ablation, but the resulting
   "head" ablation conflates a Q-head's specific contribution; the K/V
   share-pattern is unchanged.

4. **n=20 with 100% baseline**. Same caveat as layer-level: tested only
   on easy physics-mode commitments (cue=both; M5b SIP clean stim).
   Harder cases (e.g., line/blank/none) might show different per-head
   dependencies — though given how clean this null is, it would be
   surprising.

## Reproducer

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_per_head_attention_knockout.py \
    --n-pairs 20 --layers 8,9,10,11,12,13,14 --device cuda:0
```

## Artifacts

- `scripts/m5b_per_head_attention_knockout.py` — driver.
- `outputs/m5b_per_head/per_pair_results.csv`,
  `per_head_ie.csv`.
- `docs/figures/m5b_per_head_attention_ie.png` — heatmap (layers × heads, all zero).

## Open follow-ups

1. **Multi-head combination ablation**: knock out top-N heads
   simultaneously per layer; does cumulative IE emerge?
2. **Vision-token attention probes**: which heads attend most to visual
   tokens (not the same as which heads' outputs are necessary)? Per-head
   attention maps to visual positions could rank candidate "decision
   read-out" heads independent of necessity.
3. **SAE intervention** (next M5b sub-task): zero monosemantic
   "physics-cue" SAE features in the vision encoder output and measure
   PMR drop. With attention ruled out as a bottleneck, the next test
   moves *upstream* of the LM to the encoder side.
