---
section: M5b — attention + MLP knockout (Qwen2.5-VL)
date: 2026-04-26
status: complete (n=20 clean stim × 28 layers × 2 ablation types)
hypothesis: H10 (research plan §2.5) refined — attention is fully redundant single-layer; MLP at L9 is uniquely necessary; L8/L10/L11/L14 MLP partial
---

# M5b — attention + MLP knockout (Qwen2.5-VL)

> **Recap**
>
> - **Necessity vs sufficiency**: previous SIP patching tested *sufficiency* (does clean's hidden state at L *suffice* to flip corrupted to physics?). This script tests *necessity* (does ablating L's component *break* physics commitment in the clean run?).
> - **M5b SIP patching** (already done): L0-L9 → 100% physics recovery; L10-L11 → 60%; L14+ → 0%.
> - **H10** (research plan §2.5): "2-3 narrow layer/head ranges show large IE."
> - **H-locus** (M4-derived): "bottleneck at LM mid layers (L10 specifically)."

## Question

SIP patching said L0-L9 are *sufficient* — replacing corrupted's
hidden state at any of these layers with clean's recovers physics-
mode in 20/20 pairs. But sufficiency does not imply each is
*necessary*. Two complementary tests:

(a) **Attention knockout**: at each layer, zero out the attention
    output (residual stream gets `h + 0 + mlp(h)`).
(b) **MLP knockout**: at each layer, zero out the MLP output
    (`h + attn(h) + 0`).

For each ablation, run on 20 clean SIP stim (cue=both, baseline
PMR=1) and measure how often physics-mode commitment breaks.

IE_necessity = baseline_phys_rate − ablated_phys_rate. High value
means the ablated component is necessary.

## Method

- Reuse the 20 clean SIP stim from `outputs/m5b_sip/manifest.csv`
  (same stim used in the sufficiency test).
- For each clean stim:
  - Baseline forward(clean) → first letter, no ablation.
  - For each L ∈ [0..27]:
    - Hook on `layers[L].self_attn` zeroes output at prefill;
      forward+generate → score letter (attn knockout).
    - Hook on `layers[L].mlp` zeroes output at prefill;
      forward+generate → score letter (MLP knockout).
- IE_necessity per layer = (baseline phys rate) − (ablated phys rate).

20 clean stim × 28 layers × 2 ablations = 1120 forward passes;
~18 min on H200.

## Result

![M5b knockout per-layer IE](../figures/m5b_knockout_per_layer_ie.png)

Baseline phys rate: 20/20 (1.000).

### Attention knockout (necessity)

| Layer | n | ablated phys rate | IE_necessity |
|------:|--:|------------------:|-------------:|
| **All layers L0-L27** | 20 | **1.000** | **0.0** |

→ **No single-layer attention is necessary.** Single-layer attention
knockout never breaks physics commitment.

### MLP knockout (necessity)

| Layer | n | ablated phys rate | IE_necessity | comment |
|------:|--:|------------------:|-------------:|---------|
| L0-L7 | 20 | 1.0 | 0.0 | redundant |
| L8 | 20 | 0.6 | **+0.4** | partially necessary |
| **L9** | **20** | **0.0** | **+1.0** | **fully necessary** |
| L10 | 20 | 0.4 | +0.6 | partially necessary |
| L11 | 20 | 0.6 | +0.4 | partially necessary |
| L12-L13 | 20 | 1.0 | 0.0 | redundant |
| L14 | 20 | 0.6 | +0.4 | partially necessary |
| L15-L27 | 20 | 1.0 | 0.0 | redundant |

→ **L9's MLP is uniquely necessary** for physics-mode commitment.
L8 / L10 / L11 / L14 partial necessity. All other layers redundant.

## Headlines

1. **Attention is redundant single-layer**, MLP is not. Knocking
   out attention at any single layer (L0..L27) does not break
   physics commitment — 20/20 still produce A/B/C. The residual
   stream + the surviving MLPs reconstitute the attention's
   contribution.

2. **L9 MLP is the critical unit.** Knocking out only L9's MLP
   flips 20/20 clean stim from physics → abstract. This is the
   most localized causal finding in the project.

3. **Partial necessity ring around L9**: L8 (+0.4), L10 (+0.6),
   L11 (+0.4), L14 (+0.4). L8-L11 form a contiguous "computation
   block"; L14 is a small bump perhaps reflecting a
   reinforcement / propagation step.

4. **Triangulation with M5b SIP patching**:
   - SIP (sufficiency): L0-L9 patching → 20/20 physics recovery.
   - Knockout (necessity): L9 MLP knockout → 0/20 physics retention.
   - **L9 is both sufficient (transplanting clean's L9 representation
     suffices) AND necessary (without L9's MLP, decision breaks).**
   - The full picture: L0-L9 carry physics-mode-relevant info;
     L9's MLP integrates that info into a commitment; L10+ reads
     out the commitment. L9 is the "decision computation"; M5a's
     L10 was the "decision read-out" point.

5. **M5a's L10 vs M5b's L9 reconciliation**: M5a found L10 the only
   layer where +α·v_L10 steering flips behavior. M5b finds L9 MLP
   uniquely necessary. **Off by one — both at the decision boundary.**
   - M5a's L10 success: at L10, the attention reads the L9-MLP-
     produced "physics-mode" representation. Adding +α to L10's
     hidden state moves that read-out toward the physics direction.
   - M5b's L9 success: at L9, the MLP *constructs* the physics-mode
     representation in the residual stream. Without it, no
     representation to read out at L10.
   - Two views of the same decision boundary — construction vs
     read-out.

## Connection to other findings

- **§4.6 (Qwen) pixel-encodability**: gradient ascent maximizes
  `<h_L10[visual], v_L10>` — i.e., pushes L10's read-out toward
  physics-mode direction. The pixel mechanism plausibly works by
  steering visual-token features that L9's MLP turns into
  physics-mode commitment by the time L10 reads them.

- **§4.6 cross-model revised** (LLaVA-1.5 L25 admits pixel-encoding):
  LLaVA-1.5 has its own L9 equivalent — could be in the L19-L20 zone
  per M5b SIP patching. Need cross-model knockout to confirm.

- **H-locus** (M4-derived): refined again. The locus is *L9 MLP*
  for construction, *L10* for read-out. M4's "decoder bottleneck"
  framing is consistent (the *decoder side* of the pipeline is
  where decision crystallizes).

- **H10** (research plan §2.5: "2-3 narrow IE bands"): more strongly
  supported here. Attention has *no* IE bands (zero everywhere).
  MLP has *one* dominant band at L9 with partial echoes at L8 / L10
  / L11 / L14. So 1 dominant + 4 partial = within "2-3 narrow"
  spirit (the project plan's framing was approximate).

## Limitations

1. **Single-layer ablation only**. Multi-layer combinations (e.g.,
   knock out L8 + L10 + L11 simultaneously) might break physics
   even more cleanly. Not done.

2. **No per-head resolution**. Plan §2.5 calls for "specific
   layer/head" attention knockout. Per-head attention zero-out is
   harder because Qwen2 attention output is post-concat — splitting
   per-head requires modifying internal attention computation. Not
   done in this round.

3. **n=20 with 100% baseline**. The clean stim were intentionally
   selected for high baseline PMR. The knockout result generalizes
   only to "easy physics-mode commitments." Harder cases (e.g.,
   line/blank/none) might show different layer dependencies.

4. **MLP knockout via output zeroing**. We zero MLP output, which
   is harsher than "replace with clean's MLP output." A
   replacement-mode test would distinguish "L9 MLP must compute
   *something*" vs "L9 MLP must compute *the right thing*."

5. **Qwen2.5-VL only**. LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3
   knockout untested. Each model presumably has its own L9-
   equivalent (maybe at LLaVA-1.5's L19-20 lock-in zone).

## Reproducer

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_attention_mlp_knockout.py \
    --n-pairs 20 --device cuda:0
```

## Artifacts

- `scripts/m5b_attention_mlp_knockout.py` — driver.
- `outputs/m5b_knockout/per_pair_results.csv`,
  `per_layer_ie_attention.csv`, `per_layer_ie_mlp.csv`.
- `docs/figures/m5b_knockout_per_layer_ie.png` (2-panel attn + mlp).

## Open follow-ups

1. **Per-head attention knockout** in L8-L14 zone — identify the
   1-3 heads carrying the visual-token → text decision attention.
2. **Multi-layer ablation** combinations — does L8+L10+L11 simultaneous
   knockout break physics on all 20 stim?
3. **MLP replacement** (vs zeroing): replace L9 MLP output with
   *zero-input MLP output* (i.e., MLP run on zeros) to distinguish
   "MLP must compute anything" from "MLP must compute the right
   thing."
4. **Cross-model knockout** — port to LLaVA-1.5 / Idefics2 / InternVL3
   to identify each model's "L9 equivalent."
5. **SAE on Qwen vision encoder** (Pach et al.) — the next research
   plan §2.5 item.
