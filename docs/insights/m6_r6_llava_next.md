---
milestone: M6 r6
date: 2026-04-25
status: complete
hypothesis: H-encoder-saturation (architecture-level reframe)
---

# M6 r6 — LLaVA-Next-Mistral 5th model point (2nd CLIP)

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **H-encoder-saturation** — Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the architecture level (joint encoder + LM), not encoder representational capacity alone.
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim diversification — real photographs (60 photos × 5 categories from COCO + WikiArt). Photos REDUCE Qwen PMR(_nolabel) 18-48 pp.
> - **M8d** — Stim diversification — non-ball physical-object categories (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.
> - **M6 r5** — M8c photo encoder probe (4 models, cross-stim) — behavioral-y AUC inverts but stim-y AUC stays at 1.0 → encoder discriminability is uniform; architecture-level reframe.
> - **M6 r6** — LLaVA-Next-Mistral 5th model point (2nd CLIP) — PMR 0.700 [0.65, 0.74] sits between LLaVA-1.5 floor and saturated cluster; rules out vision-encoder-family as sole determinant.

## Summary

LLaVA-v1.6-Mistral-7b adds a **5th model point** to the encoder-saturation
chain. It uses the same vision encoder family as LLaVA-1.5 (CLIP-ViT-L/14)
but pairs it with Mistral-7B instead of Vicuna-7B — plus AnyRes multi-tile
image splitting, a different fusion projector, and a different training
data + recipe. Behavioral PMR(_nolabel) on M8a = **0.700, 95% CI [0.65,
0.74]**, sitting cleanly between the LLaVA-1.5 floor [0.14, 0.21] and the
saturated non-CLIP cluster [0.80, 0.92].

The 2nd CLIP point is **not** a clean "LM swap" counterfactual — four
architectural axes change at once between LLaVA-1.5 and LLaVA-Next.
We report it as a 5th observation that **rules out vision-encoder-family
as the sole determinant of behavioral PMR**, not as a causal isolation
of LM modulation.

## Numbers

### Behavioral PMR(_nolabel) on M8a

| Model       | Encoder       | LM           | M8a PMR | 95% CI         |
|-------------|---------------|--------------|--------:|----------------|
| Qwen2.5-VL  | SigLIP        | Qwen2-7B     | 0.838   | [0.800, 0.872] |
| LLaVA-1.5   | CLIP-ViT-L    | Vicuna-7B    | 0.175   | [0.140, 0.212] |
| **LLaVA-Next** | **CLIP-ViT-L** | **Mistral-7B** | **0.700** | **[0.653, 0.743]** |
| Idefics2    | SigLIP-SO400M | Mistral-7B   | 0.882   | [0.850, 0.912] |
| InternVL3   | InternViT     | InternLM2-7B | 0.917   | [0.890, 0.943] |

CIs are 5000-iter prediction-level bootstrap, resampled within each shape
(as in M9).

### Vision-encoder probe AUC on M8a

| Model       | behavioral-y AUC (deepest layer) | stim-y AUC (4 targets) |
|-------------|--------------------------------:|------------------------:|
| Qwen2.5-VL  | 0.880                            | 1.000                   |
| LLaVA-1.5   | 0.771                            | 1.000                   |
| **LLaVA-Next** | **0.809**                     | **1.000**               |
| Idefics2    | 0.926                            | 1.000                   |
| InternVL3   | 0.886                            | 1.000                   |

LLaVA-Next behavioral-y AUC at layer 23 = 0.809 — between LLaVA-1.5 (0.77)
and the saturated cluster (0.88–0.93), like the PMR ordering. Stim-y AUC
is 1.0 on all 4 targets (rendered_vs_line, physics_cell_vs_abstract_cell,
within_line_context, within_textured_context). **Same finding as the other
4 models**: the encoder linearly separates physics-cell from abstract-cell
factorial cells perfectly. CLIP-ViT-L is not an encoder bottleneck.

## Implication for H-encoder-saturation

The 2nd CLIP point closes the most obvious encoder-side counter-hypothesis:
"maybe CLIP just can't represent the physics-vs-abstract distinction." It
can — same encoder, different downstream architecture, PMR moves
0.18 → 0.70. Whatever drives the PMR floor in LLaVA-1.5 is **not at the
vision encoder**.

The remaining unknown is **which of the 4 confounded axes** (AnyRes tiling,
fusion projector, training, LM family) carries the load. Disambiguation
would require a same-architecture LM-only swap, which no released model
provides. Status: H-encoder-saturation is confirmed at the
**joint-architecture** level; the LM-modulation hypothesis is suggested
by Idefics2 (SigLIP-SO400M+Mistral PMR 0.88) vs LLaVA-1.5
(CLIP-ViT-L+Vicuna 0.18) but cannot be isolated from the available data.

## How to read the multi-axis confound

LLaVA-1.5 → LLaVA-Next changes:
1. **AnyRes multi-tile image splitting** (5 tiles vs 1) — each image is
   processed at higher effective resolution. Visual capture confirms 5×577
   patch tokens.
2. **Fusion projector** — linear projector → MLP, with different parameter
   initialization and training.
3. **Training data + recipe** — LLaVA-Next was trained on 760k examples
   vs LLaVA-1.5's 158k, with a different mix (more reasoning + chart QA).
4. **LM family** — Vicuna-7B → Mistral-7B-Instruct.

Any single one (or interaction) could drive the 0.52-PMR jump. We do not
make a "the LM did it" claim from this row.

## Reproducer

Configs:
- `configs/encoder_swap_llava_next.py` — labeled-arm M8a inference
- `configs/encoder_swap_llava_next_label_free.py` — open-prompt arm

Inference (labeled + label-free):
```bash
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next.py
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next_label_free.py
```

Capture (uses M8a stim from Qwen run):
```bash
uv run python scripts/04_capture_vision.py \
    --stimulus-dir inputs/m8a_qwen_<ts> \
    --output-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --layers 5,11,17,23 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --torch-dtype bfloat16
```

Probes:
```bash
# Behavioral-y
uv run python scripts/encoder_swap_probe.py \
    --model-name llava_next \
    --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --predictions outputs/encoder_swap_llava_next_m8a_<ts>/predictions.parquet \
    --out-dir outputs/encoder_swap_llava_next_m8a_probe \
    --layers 5,11,17,23 \
    --behavioral-pmr 0.700

# Stim-y (4 targets)
for target in rendered_vs_line physics_cell_vs_abstract_cell \
              within_line_context within_textured_context; do
    uv run python scripts/encoder_swap_probe_stim_y.py \
        --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
        --stim-dir inputs/m8a_qwen_<ts> \
        --layers 5,11,17,23 \
        --target $target \
        --out-dir outputs/encoder_swap_llava_next_m8a_probe_stim_y
done
```

Summary regeneration (5-model figure):
```bash
uv run python scripts/encoder_swap_probe_summary.py \
    --idefics2 outputs/encoder_swap_idefics2_probe \
    --internvl3 outputs/encoder_swap_internvl3_probe \
    --llava-next outputs/encoder_swap_llava_next_m8a_probe \
    --internvl3-pmr 0.917 \
    --llava-next-pmr 0.700 \
    --out-dir outputs/encoder_swap_probe_summary
```

Outputs `docs/figures/encoder_chain_5model.png` (supersedes the 4-model
figure used in r3/r4/r5 insight docs).

## Cross-stim addendum (M8d + M8c, added same day)

LLaVA-Next was also run on M8d (3 categories × 4 abstraction × 2 bg × 2 cue ×
2 events × 5 seeds = 480) and M8c (60 photos), labeled + label-free, ~16 min
total inference on GPU 0. Three findings:

![5-model × 3-stim PMR ladder](../figures/session_5model_cross_stim_pmr.png)

*Figure*: 5-model × 3-stim PMR with bootstrap CIs. LLaVA-Next sits between
LLaVA-1.5 floor and the saturated cluster on synthetic stim (M8a, M8d).
On M8c photos all 5 models compress into [0.18, 0.67].

### 1. PMR mid-band holds across all 3 stim sources

| stim | model      | mean PMR(_nolabel) | 95% CI         |
|------|------------|-------------------:|----------------|
| M8a  | LLaVA-1.5  | 0.175              | [0.140, 0.212] |
| M8a  | LLaVA-Next | **0.700**          | [0.653, 0.743] |
| M8a  | Idefics2   | 0.882              | [0.850, 0.912] |
| M8d  | LLaVA-1.5  | 0.331              | [0.294, 0.371] |
| M8d  | LLaVA-Next | **0.625**          | [0.583, 0.667] |
| M8d  | Idefics2   | 0.890              | [0.862, 0.917] |
| M8c  | LLaVA-1.5  | 0.283              | [0.183, 0.383] |
| M8c  | LLaVA-Next | **0.417**          | [0.300, 0.533] |
| M8c  | Idefics2   | 0.417              | [0.317, 0.517] |

LLaVA-Next sits **between LLaVA-1.5 floor and saturated cluster** on both
synthetic stim sources (M8a CI separated above LLaVA-1.5 + below Idefics2;
M8d CI similar). The 0.30–0.52 PMR jump from LLaVA-1.5 → LLaVA-Next is
consistent across synthetic stim — the architectural difference moves PMR
in the same direction regardless of stim shape.

### 2. Photo collapse (M8c) hits LLaVA-Next too

LLaVA-Next M8c PMR(_nolabel) = **0.417**, statistically indistinguishable
from Idefics2 M8c (0.417). The encoder gap that separates the synthetic
clusters into 3 PMR bands (0.18 / 0.70 / 0.88) compresses to a single
[0.18, 0.67] band on photos, **as predicted by the M8c finding** (M6 r5).
The 5th model point fits the same M8c-collapse pattern.

### 3. H7 collapses across architecture for the LLaVA family

| stim | model      | mean H7 (phys − abs) | 95% CI            |
|------|------------|---------------------:|-------------------|
| M8a  | LLaVA-1.5  | +0.360               | [+0.300, +0.418]  |
| M8a  | LLaVA-Next | +0.260               | [+0.205, +0.317]  |
| M8d  | LLaVA-1.5  | +0.306               | [+0.250, +0.360]  |
| M8d  | LLaVA-Next | **−0.054**           | [−0.102, −0.006]  |
| M8c  | LLaVA-1.5  | +0.100               | [−0.033, +0.233]  |
| M8c  | LLaVA-Next | +0.017               | [−0.133, +0.167]  |

LLaVA-1.5 has the project's strongest H7 on M8d (+0.31). The LLaVA-Next
architecture switch attenuates H7 strongly: M8a +0.26 (still 5/5 PASS,
mid-strong), M8d −0.05 (CI just below 0), M8c +0.02. **H7 strength is
not preserved across same-encoder-family architecture changes** —
consistent with the architecture-level reframe.

### 4. Vision-encoder probes — uniformity holds at 5th model × 3 stim

LLaVA-Next vision-encoder captures (480 stim × 4 layers on M8d, 60 × 4
on M8c) added to the existing M8a captures. Per-layer logistic-regression
probes:

| stim | LLaVA-Next behavioral-y AUC (deepest) | LLaVA-Next stim-y AUC (mean across 4 layers) |
|------|-------------------------------------:|---------------------------------------------:|
| M8a  | 0.809                                | 1.000                                        |
| M8d  | 0.905                                | 1.000                                        |
| M8c  | 0.883                                | 1.000                                        |

**Stim-y AUC = 1.0 across all 3 stim sources** — same as the original
4-model M6 r5 finding generalizes to the 5th model. CLIP-ViT-L (with
AnyRes) linearly separates physics-vs-abstract factorial cells perfectly
on every stim source.

**Behavioral-y AUC pattern (M8a → M8c)** for the 5 models, consolidated:

| model       | M8a behav-y | M8c behav-y |
|-------------|-----------:|-----------:|
| Qwen2.5-VL  | 0.88       | 0.44       |
| LLaVA-1.5   | 0.77       | 0.86       |
| LLaVA-Next  | 0.81       | 0.88       |
| Idefics2    | 0.93       | 0.77       |
| InternVL3   | 0.89       | 0.59       |

The 2 CLIP models **rise** on photos (LLaVA-1.5 +0.09, LLaVA-Next +0.07);
the 3 non-CLIP models **drop** (Qwen −0.44, Idefics2 −0.16, InternVL3
−0.30). The CLIP/non-CLIP split is preserved cross-stim *for behavioral-y
AUC*, but with opposite directions — consistent with the
"behavioral-y is downstream-conditional, not encoder-info" interpretation
from the M6 r5 stim-y check. Behavioral-y measures encoder ↔ behavior
alignment; CLIP encoders happen to align better with their LMs' photo
PMR distributions, while non-CLIP encoders align better with synthetic
PMR distributions.

**Caveat per advisor**: the M8d −0.054 effect size is in the project's
noise floor (CI excludes 0 by ~0.005). It is **symmetric to Idefics2 M8d
+0.048** — both are "barely-above/below 0" and demoted to "suggested only
not paper-defensible" under the M9 bootstrap framework. Do **not**
interpret as "Mistral-7B suppresses H7" — Idefics2 vs LLaVA-Next still
differs along encoder family, image pipeline (no AnyRes vs AnyRes),
fusion projector, and training, in addition to LM. The two-Mistral
clustering is suggestive but multi-axis-confounded, same caveat as the
M6 r6 main result.

### Implication for hypotheses

- **H-encoder-saturation** (architecture-level): cross-stim confirmed at
  the 5th model point. PMR mid-band of LLaVA-Next holds on M8a + M8d;
  photo collapse hits all 5 models.
- **H7** (label-selects-regime): unsaturated-only on LLaVA-1.5 was the
  cleanest signal in the project. LLaVA-Next removes that cleanness —
  the architectural switch breaks H7 even when PMR has measurement
  headroom (M8d PMR 0.625 is well below ceiling, yet H7 ≈ 0).
- **H-LM-modulation**: still suggested only. Not promoted.

### Cross-stim reproducer

```bash
# M8d
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8d.py --stimulus-dir inputs/m8d_qwen_<ts>
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8d_label_free.py --stimulus-dir inputs/m8d_qwen_<ts>
# M8c
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8c.py --stimulus-dir inputs/m8c_photos_<ts>
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8c_label_free.py --stimulus-dir inputs/m8c_photos_<ts>

# M9 audit re-runs the 4-model × 3-stim table with LLaVA-Next added
uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit
```

## Artifacts

- `outputs/encoder_swap_llava_next_m8a_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8d_<ts>/predictions.{jsonl,parquet,csv}` (cross-stim addendum)
- `outputs/encoder_swap_llava_next_m8d_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8c_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8c_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_vision_activations/*.safetensors` (400 stim × 4 layers)
- `outputs/encoder_swap_llava_next_m8a_probe/{layer_sweep,by_object_level}.csv`
- `outputs/encoder_swap_llava_next_m8a_probe_stim_y/layer_sweep_stim_y_*.csv`
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
![encoder_chain_5model](../figures/encoder_chain_5model.png)
