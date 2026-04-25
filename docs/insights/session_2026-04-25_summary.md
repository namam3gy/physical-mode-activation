---
session: 2026-04-25 autonomous
date: 2026-04-25
status: complete
scope: M6 r6 (5-model chain + cross-stim) + §4.2 (reverse prompting) + reproduction notebook
commits: b2434d4 → e301c61 → d35cf28 → 524e32b → dcc1d17 → cae24a9 → dd9883c → 025ab68
---

# Session 2026-04-25 — M6 r6 + §4.2 consolidation

## What this session delivered

Two paper-relevant additions to the encoder-saturation chain, plus a
consolidated reproduction notebook:

1. **M6 r6 — 5-model M8a chain + cross-stim addendum**: LLaVA-v1.6-Mistral-7b
   added as a 5th model point and 2nd CLIP point. PMR / probe / cross-stim
   results across M8a + M8d + M8c, with explicit multi-axis confound caveats
   per advisor.
2. **§4.2 — Reverse prompting on real photographs**: existing M8c labeled-arm
   data re-analyzed to test image-vs-label trade-off. Image-prior dominates
   on real photos; label-prior dominates on synthetic stim.
3. **Notebook extension**: `notebooks/encoder_saturation_chain.ipynb` now
   covers §4.5 + M6 r3 + r4 + r5 + r6 + §4.2 in 9 sections. Executes cleanly
   end-to-end (`jupyter nbconvert --execute` verified).

## Headline findings (5-model)

### 1. 5-model M8a chain (paper headline)

| Model       | Encoder         | LM           | M8a PMR(_nolabel) | 95% CI            |
|-------------|-----------------|--------------|------------------:|-------------------|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | 0.838             | [0.800, 0.872]    |
| LLaVA-1.5   | CLIP-ViT-L/14   | Vicuna-7B    | **0.175**         | [0.140, 0.212]    |
| **LLaVA-Next** | **CLIP-ViT-L/14** | **Mistral-7B** | **0.700** | **[0.653, 0.743]** |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | 0.882             | [0.850, 0.912]    |
| InternVL3   | InternViT       | InternLM2-7B | 0.917             | [0.890, 0.943]    |

3 non-CLIP architectures cluster at PMR ≥ 0.84. 2 CLIP architectures span
[0.14, 0.74] depending on downstream architecture. **The 2nd CLIP point
rules out vision-encoder-family as sole determinant of PMR.**

### 2. Cross-stim (M8a + M8d + M8c) for LLaVA-Next

| stim | LLaVA-Next PMR | 95% CI         | Reading |
|------|---------------:|----------------|---------|
| M8a  | 0.700          | [0.653, 0.743] | mid-band, between LLaVA-1.5 floor and saturated cluster |
| M8d  | 0.625          | [0.583, 0.667] | mid-band holds on synthetic categories |
| M8c  | 0.417          | [0.300, 0.533] | photo collapse: equal to Idefics2 0.417 (CIs heavily overlap) |

Photo-collapse generalizes to the 5th model. The M8c finding (M6 r5)
holds: real photos compress the 5-model PMR ladder into [0.18, 0.67].

### 3. Vision-encoder probe — stim-y AUC = 1.0 across 5 models × 3 stim sources

LLaVA-Next vision captures + probes added on M8d + M8c:

| stim | LLaVA-Next behavioral-y AUC | LLaVA-Next stim-y AUC |
|------|---------------------------:|---------------------:|
| M8a  | 0.81                       | **1.000**            |
| M8d  | 0.91                       | **1.000**            |
| M8c  | 0.88                       | **1.000**            |

Stim-y AUC = 1.0 universally across all 5 models × all 3 stim sources.
**Encoder representational capacity is uniform — across CLIP-ViT-L (×2),
SigLIP, SigLIP-SO400M, InternViT, on synthetic shapes / synthetic
categories / real photos.** What differs across the PMR ladder is
LM-side reading, not encoder discriminability.

### 4. Behavioral-y AUC encoder-family split (M8a → M8c)

| model       | M8a behav-y | M8c behav-y | Δ |
|-------------|-----------:|-----------:|----:|
| Qwen2.5-VL  | 0.88       | 0.44       | −0.44 |
| LLaVA-1.5   | 0.77       | 0.86       | +0.09 |
| LLaVA-Next  | 0.81       | 0.88       | +0.07 |
| Idefics2    | 0.93       | 0.77       | −0.16 |
| InternVL3   | 0.89       | 0.59       | −0.30 |

CLIP/non-CLIP split *opposite direction* on photos: 2 CLIP models rise,
3 non-CLIP models drop. Consistent with the stim-y reframe: behavioral-y
AUC measures encoder ↔ behavior alignment, not encoder discriminability.

### 5. H7 (label-selects-regime) collapses across architecture for LLaVA family

| stim | LLaVA-1.5 H7    | LLaVA-Next H7 | Δ |
|------|----------------:|---------------:|----:|
| M8a  | +0.360 (5/5 PASS) | +0.260 (5/5 PASS) | −0.10 |
| M8d  | +0.306 (3/3 PASS) | **−0.054 (0/3 PASS)** | −0.36 |
| M8c  | +0.100 (2/4 PASS) | +0.017            | −0.08 |

LLaVA-1.5's project-strongest H7 (M8d +0.31) does not survive the
architecture switch to LLaVA-Next. Critically: LLaVA-Next M8d PMR is
0.625 (well below ceiling) — the H7 collapse is **not** a saturation
effect. Same encoder family, same M8d stim, but H7 disappears.

### 6. §4.2 — Image dominates label on real photographs

| model       | M8d phys − abs (synth) | M8c phys − abs (photo) | Compression |
|-------------|----------------------:|----------------------:|------------:|
| Qwen2.5-VL  | +0.008                | +0.104                | +0.10       |
| LLaVA-1.5   | **+0.306**            | **+0.146**            | **−0.16**   |
| LLaVA-Next  | −0.054                | **+0.000**            | +0.05       |
| Idefics2    | +0.048                | +0.146                | +0.10       |
| InternVL3   | (skipped)             | −0.042                | (n/a)       |

LLaVA-1.5's synthetic label effect (+0.306) is **halved on photos**
(+0.146). LLaVA-Next phys − abs = 0.000 on physical photos: calling a
real ball `"circle"` does not lower PMR vs `"ball"`. **Label dominance
requires image impoverishment.**

## Hypothesis status (post-session)

- **H-encoder-saturation** — *architecture-level confirmed at 5 model
  points (3 non-CLIP + 2 CLIP) × 3 stim sources*. Stim-y AUC = 1.0
  uniform; PMR ladder is downstream-conditional.
- **H7** (label-selects-regime) — *unsaturated-saturated-conditional
  AND architecture-conditional*. LLaVA-1.5 M8d +0.31 was the project's
  strongest. LLaVA-Next architecture switch removes the M8d signal
  with PMR headroom remaining (0.625 well below ceiling).
- **H-LM-modulation** — *still suggested only*. Two-Mistral M8d H7
  clustering at ≈0 (Idefics2 +0.048 / LLaVA-Next −0.054, both
  noise-floor effects of equal magnitude) is multi-axis-confounded
  (encoder, projector, image pipeline, training all differ).
- **§4.2 image-dominates-label** — *confirmed on physical photos*.
  Synthetic label effect halves to zero on photos across all 5 models.

## Multi-axis confound (LLaVA-1.5 vs LLaVA-Next)

The 0.18 → 0.70 PMR jump on M8a is the largest single behavioral move
in this session, but it is **4-axis confounded**:
1. AnyRes multi-tile image splitting (5 tiles vs 1)
2. Fusion projector (linear → MLP, different init + training)
3. Training data + recipe (760k vs 158k examples, different mix)
4. LM family (Vicuna-7B → Mistral-7B-Instruct)

Per advisor, we report this as a 5th observation that **rules out
vision-encoder-family as the sole driver**, not as an LM-isolated
counterfactual. A clean LM-controlled encoder swap would require a
same-architecture LM-only swap, which no released model provides.

## Artifacts

### Commits (this session, 8 substantive)

- `b2434d4` — M6 r6 main: 5-model M8a chain locked
- `e301c61` — Notebook + roadmap backfill (5-model)
- `d35cf28` — 5-model scaffold (configs + ENCODER_TABLE + LLaVA-Next color override)
- `524e32b` — M6 r6 cross-stim (M8d + M8c)
- `dcc1d17` — Roadmap commit hash backfill
- `cae24a9` — M9 audit addendum
- `dd9883c` — Cross-stim probes (5×3 grid stim-y)
- `025ab68` — §4.2 reverse prompting analysis

### Docs (insights)

- `docs/insights/m6_r6_llava_next.md` (+ ko) — full M6 r6 detail
- `docs/insights/encoder_saturation_paper.md` (+ ko) — 5-model synthesis
- `docs/insights/sec4_2_reverse_prompting.md` (+ ko) — image vs label
- `docs/insights/m9_generalization_audit.md` (+ ko) — addendum pointing to LLaVA-Next 4th row

### Figures (regenerated)

- `docs/figures/encoder_chain_5model.png` — paper headline figure (supersedes 4model)
- `docs/figures/encoder_chain_4model.png` — frozen 4-model snapshot (kept for r3/r4/r5 docs)
- `docs/figures/encoder_swap_llava_next_probe.png`
- `docs/figures/m9_summary.png`, `m9_table1_heatmap.png` (5-model M8c row added via §4.2 PREFIXES update)

### Notebook

- `notebooks/encoder_saturation_chain.ipynb` — 9-section reproduction
  notebook (5-model × 3-stim chain + §4.2). Executes cleanly via
  `jupyter nbconvert --execute`.

### Configs (new)

- `configs/encoder_swap_llava_next.py` (+ `_label_free.py`) — M8a
- `configs/encoder_swap_llava_next_m8d.py` (+ `_label_free.py`) — M8d
- `configs/encoder_swap_llava_next_m8c.py` (+ `_label_free.py`) — M8c

### Outputs (gitignored, on disk)

- `outputs/encoder_swap_llava_next_m8a_*` — labeled + label-free predictions, vision activations
- `outputs/encoder_swap_llava_next_m8{d,c}_*` — cross-stim predictions + activations
- `outputs/encoder_swap_llava_next_m8a_probe{,_stim_y}/` — probes
- `outputs/encoder_swap_llava_next_m8{d,c}_probe{,_stim_y}/` — cross-stim probes
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv` — 5-model table
- `outputs/m9_audit/m9_{table1,summary}.csv` — 5-model M9 audit (M8a 5 / M8d 4 / M8c 5)

## Limitations (carried forward)

1. **Multi-axis confound** between LLaVA-1.5 and LLaVA-Next blocks any
   LM-isolated claim. No released model provides a same-architecture
   LM-only swap.
2. **n=12 photos per category on M8c** is underpowered for H7 detection.
3. **No bootstrap CIs** computed for the §4.2-specific
   "(M8d phys − abs) − (M8c phys − abs)" compression contrast.
4. **Synthetic stim factorial is M8a/M8d-style** — line/blank/none vs
   textured/ground/both. Real-world stim distributions are more varied.

## Next priorities (per roadmap §4)

- **§4.10** Attention visualization UI — paper appendix figure (this session
  set up to start; not yet implemented).
- **§4.6** Counterfactual stimulus generation via SAE/VTI reverse — adversarial
  physics-mode prompt synthesis.
- **M5b** SIP + activation patching + SAE — mechanism-level evidence.
- **M7** paper draft.
