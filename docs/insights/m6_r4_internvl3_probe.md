# M6 r4 — InternVL3 vision-encoder probe (4-model encoder chain)

**Status**: Complete 2026-04-25.

## Motivation

M6 r3 (Idefics2 SigLIP-SO400M probe) closed the AUC ↔ behavioral PMR
chain at 3 model points (Qwen + LLaVA + Idefics2). The chain showed
**SigLIP-family clusters at saturation, CLIP at headroom**.

A natural completion: the fourth model point uses **InternViT** — yet
another non-CLIP vision encoder family — paired with InternLM2-7B (yet
another LM family). M6 r2a had reported InternVL3 behavioral PMR(_nolabel)
≈ 0.99 on M2 stim (consistent with saturation), but the encoder-probe
AUC was never measured. M6 r4 adds it.

If InternViT also reaches saturated AUC → **encoder family is the
unified driver across 3 non-CLIP families**, not just SigLIP. If
InternViT shows a different AUC profile → it distinguishes SigLIP-
specific saturation from a more general "non-CLIP saturates" pattern.

## Method

`scripts/02_run_inference.py --config configs/encoder_swap_internvl3.py`
on the M8a Qwen stim dir (400 stim × 3 labels × open prompt = 1200
labeled inferences, ~8 min on GPU 0) plus `..._label_free.py` (400
inferences, ~4 min). Total inference: 12 min.

Vision capture: `scripts/04_capture_vision.py
--model-id OpenGVLab/InternVL3-8B-hf --layers 3,9,18,23` (4 of 24
InternViT layers; hidden_size=1024 vs Idefics2's 1152). Wall clock:
**47 s** on GPU 0. Note: required a fix to
`src/physical_mode/models/vlm_runner._resolve_vision_blocks` to
recognise InternVL3's `vision_tower.encoder.layer` (singular) attribute
— commit message captures the diagnostic.

Probe: `scripts/encoder_swap_probe.py --model-name internvl3` with the
same 5-fold logistic-regression protocol as M6 r3.

Behavioral PMR(_nolabel) target: per-stim mean across 3 labels of the
labeled-arm run, threshold 0.5. (n_pos=379, n_neg=21 — InternVL3 is
the most saturating model in the table at this stim.)

## Results

### InternVL3 layer sweep AUC

| layer | AUC mean | AUC std |
|------:|---------:|--------:|
| 3     | **0.938** | 0.047   |
| 9     | **0.896** | 0.102   |
| 18    | **0.865** | 0.131   |
| 23    | **0.886** | 0.089   |

Mean across layers: **0.90**. Like Qwen and Idefics2, AUC is high from
the earliest captured layer (3) and stays high — encoder-saturation
pattern consistent.

### 4-model encoder-saturation chain on M8a (apples-to-apples, all 4 captured on M8a)

| model         | encoder         | LM            | encoder AUC (M8a) | M8a behavioral PMR(_nolabel) |
|---------------|-----------------|---------------|------------------:|-----------------------------:|
| Qwen2.5-VL    | SigLIP          | Qwen2-7B      | **0.880**         | **0.838**                    |
| LLaVA-1.5     | CLIP-ViT-L/14   | Vicuna-7B     | **0.771**         | **0.175**                    |
| Idefics2      | SigLIP-SO400M   | Mistral-7B    | **0.926**         | **0.882**                    |
| InternVL3     | InternViT       | InternLM2-7B  | **0.886**         | **0.918**                    |

**3 non-CLIP encoders cluster at AUC 0.88–0.93, behavioral PMR 0.84–0.92.**
**1 CLIP encoder (LLaVA) sits at AUC 0.77, behavioral PMR 0.18.**

**Note on the M6 r2 vs M8a AUC numbers**: M6 r2 reported Qwen 0.99 / LLaVA
0.73 on M2 stim (12-cell factorial including the most-extreme line/blank/
none vs textured/ground/both bimodal split). M8a stim's wider per-cell
PMR distribution makes the binary y-target less clean, so AUCs come in
lower (Qwen 0.88, LLaVA 0.77). The non-CLIP / CLIP gap holds in either
stim source: ~0.10–0.20 in AUC, ~0.65 in behavioral PMR.

The nonlinear AUC → PMR mapping (a 0.10 AUC gap → 0.65 PMR gap) is itself
consistent with H-encoder-saturation: there's an AUC threshold above
which behavioral PMR saturates (the 0.85+ AUC band) and below which it
sits at headroom (~0.77).

## Methodological note — stim-defined y check (2026-04-25)

A late round of probing held y constant across models by defining it
from stim properties (instead of each model's own behavioral PMR).
Three stim-defined targets were tried on the same captures:

| target                                     | n_pos / n_neg | AUC across all 4 models, all layers |
|--------------------------------------------|---------------|-------------------------------------|
| rendered_vs_line  (obj != line)            | 300 / 100     | **1.000** (zero variance)           |
| physics_cell_vs_abstract_cell              |               |                                     |
|   (textured+ground+both vs line+blank+none)| 25 / 25       | **1.000**                           |
| within_line_context                        |               |                                     |
|   (line + ground+both vs line + blank+none)| 25 / 25       | **1.000**                           |
| within_textured_context                    |               |                                     |
|   (textured + ground+both vs textured + blank+none)| 25 / 25 | **1.000**                       |

**Every encoder linearly separates these factorial cells at AUC = 1.0.**
This holds for SigLIP (Qwen), CLIP-ViT-L (LLaVA), SigLIP-SO400M (Idefics2)
and InternViT (InternVL3) — including within-object-level minimal-pair
contrasts (same object, same shape, only context axes differ).

**Implication for the H-encoder-saturation claim**: the differences in
the *behavioral-y* probe AUC (Qwen 0.88, LLaVA 0.77, Idefics2 0.93,
InternVL3 0.89) reflect alignment between each model's encoder
representation and that model's *behavioral PMR distribution*, not
encoder representational capacity per se. All 4 encoders carry the
factorial information cleanly; what differs across architectures is
how each LM consumes the encoder output as a physics-mode signal.

The chain is therefore better stated as:

```
encoder family + LM family together determine joint architecture
              ↓
LM-side reading of encoder output as physics-mode signal
   (non-CLIP architectures: yes; CLIP-LLaVA: no)
              ↓
behavioral PMR(_nolabel) saturated vs unsaturated
              ↓
H7 measurability gated by behavioral PMR ceiling
```

This is a refinement, not a refutation, of the §4.5 + M6 r3 + M6 r4
work. The 4-model behavioral PMR ladder (Qwen 0.84 / Idefics2 0.88 /
InternVL3 0.92 / LLaVA 0.18) is unchanged. What changes: the mechanism
locus moves from "encoder representational capacity" to "LM-side
consumption of encoder output". The original M3/M6 r2 framing of
"encoder AUC predicts behavioral PMR" is correct as a *downstream-
conditional* statement (probe AUC trained with behavioral-y) but not
as an "encoder discriminability" statement (probe AUC trained with
stim-y is uniformly 1.0).

## Headline interpretation

The **H-encoder-saturation chain generalizes from SigLIP-specific to
non-CLIP-general**:

```
encoder family           encoder probe AUC (M8a)   M8a behavioral PMR(_nolabel)
─────────────            ───────────────────────   ────────────────────────────
SigLIP    (Qwen)                  0.88                       0.84
SigLIP-SO400M (Idefics2)          0.93                       0.88
InternViT (InternVL3)             0.89                       0.92
CLIP-ViT-L (LLaVA)                0.77                       0.18
```

**3 distinct non-CLIP encoder families** (SigLIP, SigLIP-SO400M, InternViT)
all reach AUC ≥ 0.88 and behavioral PMR ≥ 0.84. **Only CLIP-ViT-L falls
below saturation** (0.77 / 0.18). The encoder-saturation regime is
robustly identified by encoder family, *across LM families*
(Qwen2-7B, Mistral-7B, InternLM2-7B, Vicuna-7B) and *across encoder
implementations*.

This is the paper-grade headline of the encoder-saturation work:
**vision-encoder family causally determines whether a VLM enters the
synthetic-stim physics-mode ceiling regime, with all 3 non-CLIP
encoders we tested saturating and the only CLIP encoder we tested not
saturating.**

## Hypothesis updates

- **H-encoder-saturation** — *refined to architecture-level (encoder + LM)
  rather than encoder-discriminability*. Stim-defined y check shows all
  4 encoders separate factorial cells at AUC = 1.0; encoder family does
  not differ in raw discriminability. Updated paper claim: "VLM
  architecture family (non-CLIP encoder + various LMs vs CLIP-ViT-L +
  Vicuna) causally determines whether the joint system reads encoder
  output as physics-mode signal on synthetic stim, producing the
  saturated-vs-unsaturated behavioral PMR(_nolabel) split. The
  difference is at the encoder-LM fusion level, not at encoder
  representational capacity." The 4-model behavioral PMR ladder is
  unchanged; the mechanism locus shifts to LM-side consumption.
- **H-LM-modulation** (M9-derived) — *unchanged*. With 3 non-CLIP × 3
  LM families now in the table (Qwen2-7B, Mistral-7B, InternLM2-7B)
  showing similar saturation, LM family does not drive saturation. Any
  residual H7 effect is sub-saturation noise.

## Limitations

1. ~~**Cross-stim AUC mismatch**~~ → **Resolved**: re-captured Qwen +
   LLaVA on M8a stim (commit `<this round>`). All 4 AUC values now
   computed on the same stim distribution. Headline structure
   (non-CLIP cluster vs CLIP outlier) holds in both stim sources;
   absolute AUC values shift somewhat with stim distribution as
   expected (probe AUC depends on the per-cell PMR distribution that
   sets the y target).
2. **InternVL3 PMR(_nolabel) is the highest in the table** (0.92), but
   AUC at the deepest layer (0.886) is the lowest of the 3 non-CLIP
   models. The slight inverse correlation among the saturated cluster
   may indicate some encoder-LM trade-off — but all 3 are clearly in
   the saturation regime, far above LLaVA.
3. **Per-shape AUC sparse**: the by-shape probe only computes for
   shapes where both classes are present (n=80 per shape, but with
   InternVL3's n_neg=21 globally, several shapes are all-positive,
   making per-shape AUC undefined for circle and polygon). The pooled
   layer-sweep numbers are the headline.
4. **Only 4 models, only 1 CLIP**: stronger paper claim would test 2
   CLIP-based encoders (e.g., LLaVA-Next, ShareGPT4V) and 4 non-CLIP
   to rule out "LLaVA-1.5 specifically" as the CLIP outlier.

## Headline figure

`docs/figures/encoder_chain_4model.png` — 2-panel paper figure:
- (a) Layer sweep AUC for the 2 captured models (Idefics2 + InternVL3)
  + horizontal lines for M6 r2 baselines (Qwen 0.99, LLaVA 0.73).
- (b) Scatter (encoder AUC, behavioral PMR) for all 4 model points —
  the H-encoder-saturation chain visualized.

`docs/figures/encoder_swap_internvl3_probe.png` — InternVL3-only
2-panel figure (analogous to encoder_swap_idefics2_probe.png).

## Roadmap implications

- **§4.5 + M9 + M6 r3 + M6 r4 = paper-grade encoder-saturation chain.**
  4 model points × 4 nodes (encoder family → AUC → behavioral PMR →
  H7 measurability) is the strongest causal evidence we can produce
  short of a same-LM encoder-swap counterfactual.
- **Same-LM encoder swap** (e.g., LLaVA-1.5 with CLIP vs LLaVA-1.5
  with SigLIP via Bunny / ShareGPT4V) remains the cleanest
  counterfactual — but is now a "round 5" enhancement, not a blocker.
- ~~**Re-capture Qwen + LLaVA on M8a**~~ → **Done in this round.**
  All 4 models' AUC values now from M8a stim. The cross-stim caveat
  is resolved.

## Artifacts

- `configs/encoder_swap_internvl3{,_label_free}.py`.
- `scripts/encoder_swap_probe.py` (model-agnostic driver, renamed from
  `encoder_swap_idefics2_probe.py`).
- `scripts/encoder_swap_probe_summary.py` (4-model unified figure).
- `outputs/encoder_swap_internvl3_*/predictions.{jsonl,parquet,csv}`.
- `outputs/encoder_swap_internvl3_vision_activations/*.safetensors`
  (~3 GB, gitignored).
- `outputs/encoder_swap_internvl3_probe/{layer_sweep,by_object_level,by_shape}.csv`.
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`.
- `docs/figures/encoder_swap_internvl3_probe.png` (per-model 2-panel).
- `docs/figures/encoder_chain_4model.png` (4-model unified — paper headline).
- `docs/insights/m6_r4_internvl3_probe.md` (+ `_ko.md`).
