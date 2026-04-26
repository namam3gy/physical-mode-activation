---
section: §4.10
date: 2026-04-25
status: complete (initial release)
scope: Qwen2.5-VL only — paper appendix figure infrastructure
---

# §4.10 — Attention visualization UI

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **M3** — ST2 vision-encoder probing — encoder AUC ≈ 1.0 trivially separates factorial axes ("boomerang").
> - **M4** — ST3 LM logit lens / per-layer probes — LM AUC plateaus at ~0.95 at visual-token positions from L5.
> - **M5b** — ST4 Phase 3 (SIP + activation patching + SAE feature decomposition) — deferred / optional.
> - **M6** — ST5 cross-model sweep — see M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next).
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).

## Purpose

Qualitative complement to the per-layer probe AUC numbers in M3 / M4 /
M6: attention heatmaps show **where** the LM looks when it commits to a
generation, layer-by-layer. The probes tell us *what information* is
present in encoder representations; attention maps suggest *what the
model attends to* before responding.

## Capture infrastructure

Required a small change to `src/physical_mode/models/vlm_runner.py`:
when `capture_lm_attentions=True`, the model is loaded with
`attn_implementation="eager"` instead of `"sdpa"`. SDPA does not return
attention weights, so a SDPA capture produces empty `lm_attn_*`
tensors silently. The toggle is automatic — set the config flag and
the runner does the right thing.

Capture cost on Qwen2.5-VL-7B (M8a stim, limit=20 stim × 3 labels):
~30 s wall time, ~7 MB per safetensors file (28 heads × 390 q × 390 k
attentions at 4 captured layers, fp16).

## Saved tensors

Each `outputs/attention_viz_qwen_<ts>/activations/<sample_id>.safetensors`:

| key                  | shape          | dtype         |
|----------------------|----------------|---------------|
| `lm_attn_5/15/20/25` | (28, 390, 390) | torch.float16 |
| `lm_hidden_5/15/20/25`| (324, 3584)   | torch.bfloat16|
| `visual_token_mask`  | (390,)         | torch.uint8   |
| `input_ids`          | (390,)         | torch.int64   |

Visual tokens occupy positions 37–360 (contiguous) — 324 tokens =
18×18 patch grid for the 256×256 M8a stim.

## Notebook structure (`notebooks/attention_viz.ipynb`)

6 sections + reproduction:
1. **Load capture + inspect shapes** — `load_capture(sample_id)` helper.
2. **Per-layer heatmap (mean over heads)** — last input token →
   visual grid, side-by-side across the 4 captured layers.
3. **Overlay on original image** — upsample the 18×18 grid to image
   resolution, alpha-blend with the stimulus.
4. **Physics-mode vs abstract-mode comparison** — picks one PMR=1
   exemplar and one PMR=0 exemplar (label=ball arm).
5. **Per-head fine structure** — all 28 heads of a single layer in a
   grid (4 rows × 7 cols).
6. **Aggregate** — visual-token attention entropy by layer × PMR
   across the captured subset.

Section 6 is the closest thing to a quantitative claim: if PMR=1 stim
have **lower** attention entropy at later layers than PMR=0 stim, that
would suggest "physics-mode commitment correlates with focused visual
attention". Initial run shows the bias direction; n=20 stim per arm is
too small for significance testing.

## What this gives the paper

A qualitative figure for the appendix that:
- Confirms the model attends to relevant patches (e.g., the object
  silhouette + the ground plane) at later layers
- Shows head specialization (per-head heatmaps reveal that a few heads
  carry most of the visual binding, consistent with mech-interp work
  on sparse attention)
- Provides a hook for future work on attention-knockout / SIP / SAE
  decomposition (M5b)

## Cross-model extension (added same day)

LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 captured at the same M8a
subset (limit=20, layers 5/15/20/25). Total disk cost ~2.5 GB
(LLaVA-Next dominates at ~480 MB / file due to AnyRes 5-tile attention).

### Visual token counts differ across architectures

| Model       | Visual tokens | Grid layout       | Encoder        |
|-------------|--------------:|-------------------|----------------|
| Qwen2.5-VL  | 324           | 18×18 (1 tile)    | SigLIP         |
| LLaVA-1.5   | 576           | 24×24 (1 tile)    | CLIP-ViT-L     |
| LLaVA-Next  | 2928          | 5-tile AnyRes     | CLIP-ViT-L     |
| Idefics2    | 320           | non-square split  | SigLIP-SO400M  |
| InternVL3   | 256           | 16×16 (1 tile)    | InternViT      |

![Cross-model attention to visual tokens](../figures/session_attention_cross_model.png)

*Figure*: per-layer fraction of last-token attention going to visual tokens
across the 5 captured VLMs on the same M8a stim. Dashed lines = uniform
baseline (n_visual / seq_len, 79–98%). All 5 models attend ≪ baseline,
peaking at mid-layers (15 or 20).

### Cross-model fraction of last-token attention on visual tokens

| Model       | layer 5 | layer 15 | layer 20 | layer 25 | n_visual / seq_len |
|-------------|--------:|---------:|---------:|---------:|-------------------:|
| Qwen2.5-VL  | 0.030   | 0.044    | **0.146**| 0.102    | 0.831              |
| LLaVA-1.5   | 0.049   | 0.097    | **0.143**| 0.059    | 0.901              |
| LLaVA-Next  | 0.187   | **0.206**| 0.085    | 0.090    | 0.976              |
| Idefics2    | 0.106   | **0.256**| 0.161    | 0.060    | 0.823              |
| InternVL3   | 0.033   | 0.036    | **0.169**| 0.048    | 0.793              |

**Key finding**: all 5 VLMs attend to visual tokens at only **3–26%**
even though visual tokens occupy 79–98% of the input sequence. The
LM's last-token attention is dominated by recent prompt + system tokens.
Visual attention peaks at mid-layers (15 or 20) for every model — same
layer band where M4 observed label-physics margin development.

This is consistent with the architecture-level reframe: encoder output
is uniform (stim-y AUC = 1.0), but the LM only "looks" at it briefly,
at mid-layers, allocating most attention bandwidth to the linguistic
context. The **architecture differences shape how strongly the brief
visual peek translates into PMR commitment** — not whether the visual
information is present.

### Heatmap overlays (square-grid models)

Section 8 of the notebook shows side-by-side overlays for the 3
square-grid models (Qwen 18×18 / LLaVA-1.5 24×24 / InternVL3 16×16) at
layer 20 on the same stim. Visual attention concentrates on the
object silhouette + ground plane regions for the saturated models.

LLaVA-Next AnyRes (5 tiles) and Idefics2 (non-square split) are skipped
in the overlay viz — their attention is naturally a multi-tile
structure that needs per-tile breakdown rather than a single grid.

## Limitations

1. **Limited stim subset (n=20)**. The capture covers a
   representative slice but is not statistically powered.
2. **Attention is only one slice of the mechanism**. Activation
   patching (M5b) or SAE decomposition is needed to make causal
   claims about what features attention is reading.
3. **Eager attention is slower than SDPA** — fine for a one-time
   capture, but do not enable in production runs.
4. **LLaVA-Next + Idefics2 overlays not implemented** — multi-tile
   structure needs per-tile decomposition, not in this initial round.
5. **Layer 25 means different things across models**: late-late for
   Qwen/InternVL3 (28 layers total), early-late for LLaVA-Next/Idefics2
   (32 layers) and LLaVA-1.5 (32 layers). Cross-model layer comparison
   is informal — depths are not normalized.

## Reproducer

```bash
# Capture (5 models, M8a Qwen stim, ~30s each on H200)
for cfg in attention_viz_qwen attention_viz_llava attention_viz_llava_next \
           attention_viz_idefics2 attention_viz_internvl3; do
    uv run python scripts/02_run_inference.py \
        --config configs/$cfg.py \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# View notebook
uv run jupyter lab notebooks/attention_viz.ipynb
```

## Artifacts

- `configs/attention_viz_{qwen,llava,llava_next,idefics2,internvl3}.py` —
  capture configs (each: limit=20, capture_lm_attentions=True,
  layers=(5,15,20,25)).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation`
  auto-switch to `"eager"` when capturing attentions.
- `outputs/attention_viz_<model>_<ts>/` — predictions + 20 ×
  safetensors capture files (sizes: Qwen 7 MB, LLaVA-1.5 21 MB,
  LLaVA-Next 480 MB, Idefics2 10 MB, InternVL3 5 MB per file).
- `notebooks/attention_viz.ipynb` — 8-section interactive viz notebook
  (sections 1-6: Qwen single-model; sections 7-8: 5-model cross-model).
- `docs/insights/sec4_10_attention_viz.md` (this doc, + ko).
