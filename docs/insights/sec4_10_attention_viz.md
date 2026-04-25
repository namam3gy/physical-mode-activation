---
section: §4.10
date: 2026-04-25
status: complete (initial release)
scope: Qwen2.5-VL only — paper appendix figure infrastructure
---

# §4.10 — Attention visualization UI

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

## Limitations

1. **Single model only**. Extending to LLaVA-1.5 / LLaVA-Next /
   Idefics2 / InternVL3 would multiply the disk cost (each capture
   ~7 MB × 60 records × 5 models = ~2 GB).
2. **Limited stim subset (n=20)**. The capture covers a
   representative slice but is not statistically powered.
3. **Attention is only one slice of the mechanism**. Activation
   patching (M5b) or SAE decomposition is needed to make causal
   claims about what features attention is reading.
4. **Eager attention is slower than SDPA** — fine for a one-time
   capture, but do not enable in production runs.

## Reproducer

```bash
# Capture (uses M8a Qwen stim, ~30s)
uv run python scripts/02_run_inference.py \
    --config configs/attention_viz_qwen.py \
    --stimulus-dir inputs/m8a_qwen_<ts>

# View notebook
uv run jupyter lab notebooks/attention_viz.ipynb
```

## Artifacts

- `configs/attention_viz_qwen.py` — capture config (limit=20,
  capture_lm_attentions=True, layers=(5,15,20,25)).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation`
  auto-switch to `"eager"` when capturing attentions.
- `outputs/attention_viz_qwen_<ts>/` — predictions + 20 ×
  safetensors capture files.
- `notebooks/attention_viz.ipynb` — interactive viz notebook.
- `docs/insights/sec4_10_attention_viz.md` (this doc, + ko).
