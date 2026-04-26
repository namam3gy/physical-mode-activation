---
section: §4.6
date: 2026-04-26
status: design — approved 2026-04-26 (user "그대로 진행해")
---

# §4.6 — Counterfactual stimulus generation via VTI reverse (design)

## Goal

Generate an "adversarial physics-mode stimulus" by gradient-ascent on
the M5a `v_L10` direction. **Synthesize an image (perturbation of an
abstract baseline) that humans still call abstract but the VLM reads
as physics-mode** (high PMR). This is the counterfactual to the M5a
result: M5a showed `+α · v_L10` injected at the LM activation level
flips abstract responses to physics-mode; §4.6 asks whether we can
reach the same activation-space target from the *image* side, and
what such an image looks like.

If the synthesized image is human-imperceptibly-different from the
baseline yet flips PMR, that is direct evidence of a **shortcut
interpretation** in Qwen2.5-VL (cf. encoder-saturation / label-prior
findings).

## Background

- **M5a (`outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/`)**:
  `v_L10 ∈ ℝ^3584` is the unit direction of `mean(h_L10 | PMR=1) −
  mean(h_L10 | PMR=0)` over the M2 factorial (forced-choice arm).
  L10 α=40 flips 10/10 D → B uniformly across (line, textured) ×
  (ball, circle) at moderate baseline.
- **M5a-ext**: `+α · v_L10` selects kinetic regime, `−α · v_L10`
  selects static regime — `v_L10` is a *regime axis within physics-
  mode*, not a binary on-off switch.
- §4.6 has been "PRIORITY 5 next" since 2026-04-25. SAE-based variant
  (originally Sub-task 1) is deferred — SAE training is M5b stretch.
  This spec uses `v_L10` only.

## Scope

In scope:
- Pixel-space gradient ascent (Approach A: optimize over post-processor
  `pixel_values` tensor).
- Single direction `v_L10` (existing, validated).
- Baseline stim: `line/blank/none × circle` (M8a abstract anchor with
  lowest PMR — highest room to flip; 10 seeds available).
- Bounded (`‖δ‖_∞ ≤ ε`) + unconstrained ablation.
- Quantitative PMR pre/post + qualitative figure + control (random
  direction).

Out of scope:
- SAE feature ascent (M5b stretch).
- Approach B (raw RGB pixel-space + processor reimplementation) —
  fallback only.
- Approach C (activation-space "virtual stim") — fallback only.
- Other layers' steering vectors (`v_L15` / `v_L20` / `v_L25`); follow-
  up if `v_L10` does not converge.
- Cross-model extension; this is Qwen2.5-VL-only (M5a is Qwen-only).

## Approach (A — recommended)

Pixel-space gradient ascent on the *post-processor* `pixel_values`
tensor (shape `(1, T_patches, 1176)` for Qwen2.5-VL with the existing
M2 stim resolution).

```
PIL stim → processor → pixel_values_0  (one-time, no grad)
  ↓
δ = leaf tensor, same shape as pixel_values_0 (init zero, requires_grad)
  ↓
pixel_values = pixel_values_0 + δ
  ↓ (Qwen vision tower + projector + LM 0..10)
h_L10[visual_tokens]                    (with grad)
  ↓
loss = -⟨mean(h_L10[visual_tokens]), v_L10_unit⟩
  ↓ backward
δ.grad
  ↓ Adam step
δ ← δ - lr · δ.grad   (gradient ascent on projection ⇔ descent on −proj)
  ↓ (optional) clip to ‖δ‖_∞ ≤ ε
```

Loop for N steps (default 200), recording projection per step. After
convergence, reconstruct the image from `(pixel_values_0 + δ)` for
human-eye viz (un-flatten patches + denormalize), and run a fresh
PMR inference on the reconstructed PIL image as causal sanity check.

Key design choices:
1. **`pixel_values` leaf tensor** — bypasses the non-differentiable
   PIL-side `processor()` call. We snapshot a single `pixel_values_0`
   per stim (forward through the processor once, with gradient
   disabled) and then optimize a delta in this normalized-and-
   patch-flattened space.
2. **Loss target** — mean projection over visual tokens. Aligns with
   how M5a computed `v_L10` (mean-pooled hidden states). Per-token
   targeting could be more aggressive but harder to interpret.
3. **L∞-bounded mode** — `δ ← clip(δ, [−ε, ε])` after each step. ε
   sweep `{0.05, 0.1, 0.2}` (in normalized pixel-value space, where
   raw pixel values are roughly in `[−2.0, 2.0]` after SigLIP
   normalization). The question "does the model see physics through
   imperceptible noise?" is answered with small ε.
4. **Unconstrained mode** — same loop without clipping. Reveals what
   the model considers "maximal physics" raw — likely visually messy
   but informative.
5. **Random-direction control** — same loop with `v_random` (random
   unit vector in 3584-d) as target instead of `v_L10`. Confirms the
   effect is `v_L10`-specific, not a generic image-perturbation
   artifact. n=3 controls.
6. **Causal sanity check** — after gradient ascent, run the inference
   pipeline on the reconstructed image (not on `pixel_values + δ` —
   we round-trip through the image to confirm the perturbation
   survives reconstruction). Measure PMR delta vs baseline.

### Why not Approach B (PIL-bypass + raw RGB tensor)

Qwen2.5-VL's processor does dynamic resolution + 28-pixel patch
padding + spatial merge. Reimplementing this differentiably is a
high-bug-risk surgery. Approach A sidesteps the issue entirely by
optimizing post-processor.

### Why not Approach C (activation-space virtual stim)

C optimizes over the vision-encoder *output* — bypassing the encoder
entirely. Faster + simpler, but the "stimulus" is then a token grid,
not an image. The §4.6 claim ("adversarial *image*") loses force.
Reserve C as a fallback if A fails (e.g., if the vision tower turns
out to be non-differentiable in PyTorch's autograd graph — unlikely
for SigLIP but possible if Qwen's spatial merge has a non-
differentiable cast or no_grad block).

## Architecture

### Components

1. **`src/physical_mode/synthesis/counterfactual.py`** (new module):
   - `pixel_values_from_pil(pil, processor, device) -> Tensor`
     — one-shot forward through processor; returns the
     `pixel_values` leaf-eligible tensor.
   - `forward_to_layer_h(model, processor, inputs, layer)
     -> Tensor` — re-implements the L0 → L<layer> forward pass with
     gradients enabled. Stops at L<layer> (no need to compute the
     full LM forward).
   - `reconstruct_pil(pixel_values, processor) -> PIL.Image`
     — un-patch + de-normalize back to a viewable image.
   - `gradient_ascent(model, pil, v_unit, layer, n_steps, lr, eps,
     mode='bounded') -> dict` — main loop. Returns updated
     pixel_values, projection trajectory, reconstructed PIL.
2. **`scripts/sec4_6_counterfactual_stim.py`** (new driver):
   - Args: `--baseline-dir`, `--steering-vectors`, `--layer`,
     `--n-seeds`, `--n-steps`, `--lr`, `--eps`, `--mode {bounded,
     unconstrained}`, `--output-dir`.
   - Runs the loop on each baseline stim (default 5 seeds), saves
     per-seed reconstructed image + projection trajectory + final
     pixel_values delta.
   - Re-runs PMR inference on the reconstructed images and writes
     a results CSV (sample_id, seed, mode, eps, baseline_pmr,
     synthesized_pmr, projection_trajectory).
3. **`scripts/sec4_6_summarize.py`** (analysis):
   - Aggregate PMR pre/post per (mode, eps), bootstrap CI.
   - Generate the 4-panel figure (baseline / ε=0.05 / ε=0.1 /
     unconstrained) for one canonical seed.
   - Generate projection trajectory plot.
   - Random-direction control comparison.
4. **`notebooks/sec4_6_counterfactual_stim.ipynb`**: walkthrough
   reproduction (per project rule 7 — milestone-level reproducibility,
   though §4.6 is a §-level extension; precedent is mixed: most §4.x
   skip notebooks but §4.6 is large enough to warrant one).

### Data flow

```
inputs/mvp_full_*/images/line_blank_none_fall_{000..004}.png
                    │
                    ▼ (PIL load)
      ┌─────────────────────────────┐
      │  Approach A loop (per seed) │
      │  - processor forward 1×     │
      │  - leaf δ + gradient ascent │
      │  - 200 steps                │
      └──────────┬──────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  reconstructed      projection
  image (PIL)        trajectory
       │                   │
       ▼                   ▼
  inference         npz / parquet
  pipeline                │
       │                   │
       ▼                   ▼
  predictions       summarize.py
       │                   │
       └────────┬──────────┘
                ▼
        figures + CSV + insight doc
```

### Differentiability check (pre-implementation)

Before writing the optimizer, the implementation plan must:
1. Confirm the Qwen2.5-VL forward path from `pixel_values` to
   `h_L<layer>` is fully differentiable (no `@torch.no_grad`, no
   non-differentiable ops). Likely points of failure: (a) any
   `torch.compile` or `torch.jit` regions, (b) `torch.argmax` /
   `topk` in the spatial merge, (c) cast-to-bfloat16 with
   `enable_grad=False`.
2. If non-differentiable: localize the op, decide whether to bypass
   (Approach C) or surgically replace.

This must be verified in iteration 1 of implementation BEFORE writing
the optimizer loop. A 30-line smoke script that does
`pixel_values.requires_grad_(True); h = forward_to_L10(...);
h.sum().backward(); print(pixel_values.grad)` decides Approach A vs
fallback.

### Error handling

1. **Optimization divergence** — if projection trajectory plateaus
   below baseline-mean projection: fail gracefully, log "did not
   converge", skip viz (don't ship a misleading figure).
2. **Reconstructed-image PMR ≤ baseline PMR** — gradient-ascent
   succeeded in pixel-values space but the round-trip through PIL
   destroyed the signal. Log this failure mode prominently — it's a
   meaningful finding (rules out adversarial transfer). If it
   happens, retry with the un-rounded pixel_values directly via the
   inference path (skip PIL reconstruction).
3. **Vision tower non-differentiable** — fall back to Approach C.

### Testing

- Unit test: `pixel_values_from_pil` round-trip (load → forward →
  reconstruct → load again, check ≤ 1e-6 max abs diff modulo
  uint8-rounding).
- Smoke test: 1 seed × 10 steps × bounded mode → projection
  trajectory monotonically increases (or fail).
- Integration test: random-direction control should not flip PMR
  significantly (Δ within ±0.1).

## Success criteria

Required:
1. Synthesized images (5 seeds × bounded mode at ε=0.1) flip PMR from
   baseline 0.18 to ≥ 0.5 (mean of 5 seeds, ≥ 3/5 individual flips).
2. Random-direction control: Δ PMR ≤ 0.1 (no flip — confirms
   `v_L10`-specificity).
3. Projection trajectory monotonically increases over the 200-step
   loop (or any time-step decrease is < 5% of total).

Stretch:
1. Bounded mode at ε=0.05 (smaller, less perceptible) still flips ≥
   2/5 seeds.
2. Unconstrained mode produces visually-interpretable physics-cue-
   like patterns (e.g., gradients, shadows).

Failure modes (each documented in the insight doc):
- A1: Pixel-values gradient ascent works but PIL round-trip breaks
  (PMR baseline-level after reconstruction). Documents: "the model
  reads physics through specific pixel-value patterns, but those
  patterns don't survive PIL/uint8 round-trip" — interesting null.
- A2: Vision tower not differentiable. Switch to Approach C, document
  the bypass.
- A3: Even unconstrained ascent fails to lift projection. v_L10
  unreachable from pixel space — documents shortcut may be in the LM
  itself, not the encoder; fits encoder-saturation finding.

Each is a publishable result, just not the headline.

## Sub-tasks (high-level — full plan via writing-plans skill)

1. **Differentiability smoke** (~1 hr) — verify backward pass through
   Qwen vision tower → projector → LM 0..10.
2. **`counterfactual.py` module** (~3 hrs) — pixel-values forward,
   gradient ascent loop, reconstruction.
3. **`sec4_6_counterfactual_stim.py` driver** (~2 hrs) — 5 seeds ×
   3 ε × 2 modes × N steps.
4. **`sec4_6_summarize.py` + figure** (~2 hrs) — aggregate, viz.
5. **`docs/insights/sec4_6_counterfactual_stim.md` + KO** (~1 hr).
6. **Notebook reproduction** (~1 hr) — milestone-level.
7. **Roadmap update + commit** (~30 min).

Total estimate: 10-11 hrs across 4-5 implementation iterations.

## Risks

1. **Differentiability** (high): Qwen's vision tower may have non-
   differentiable parts. Mitigation: smoke test as iteration 1, fall
   back to Approach C documented here.
2. **Memory** (mid): full backward through Qwen vision + projector +
   LM 0..10 may exceed 80 GB on H200 with bf16. Mitigation: use
   `torch.utils.checkpoint` for the vision tower; if still OOM,
   switch to Approach C.
3. **PIL round-trip degradation** (low-mid): uint8 quantization of the
   reconstructed image may destroy the perturbation. Mitigation:
   document as a failure mode (still informative); consider keeping
   pixel_values directly for the final PMR measurement (bypassing
   PIL).
4. **Single-model claim** (low): §4.6 is Qwen-only because v_L10 is
   Qwen-only. Cross-model variant is out of scope (would need to
   re-derive `v_L10` per model first — hours of capture work).

## Open questions for user review

1. Does the user prefer the **Approach A bound-and-unconstrained sweep**
   as scoped, or just **bounded ε=0.1 single-mode** as a smaller MVP?
2. Should the random-direction control be **n=3 or n=5**?
3. Should the deliverable include a **reproduction notebook** (project
   rule 7 mentions for milestone-level; §4.x usually skip)?
4. **Time budget**: is 10-11 hrs across 4-5 iterations acceptable, or
   should we trim (e.g., drop unconstrained ablation, drop n_seeds to
   3) to ~6 hrs?

## Status

- **2026-04-26 18:24 UTC** — Spec drafted in autonomous mode.
  Defaults: M8a abstract baseline (line/blank/none × circle, 5 seeds),
  Approach A pixel-values gradient ascent, bounded ε ∈ {0.05, 0.1, 0.2}
  + unconstrained, random-direction control n=3, full deliverable
  (script + driver + summarize + notebook + insight doc + KO).
- **2026-04-26 — approved by user** ("그대로 진행해"). All open
  questions defaulted to the recommended answers. Proceeding to
  writing-plans skill for detailed implementation plan.
