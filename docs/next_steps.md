# Next steps — code-level entry points for remaining sub-tasks

Companion to `references/roadmap.md`. The roadmap is the milestone-level view;
this file is the per-module plug-in detail. Keep up to date as work progresses.

Many sub-tasks here have already been delivered (M3, M4, M5a). The list below
describes what's left.

## Sub-task 4 Phase 3 — SIP + activation patching + SAE (M5b, optional)

**Goal** (`references/project.md` §2.5). Move the M5a steering finding from
"a direction exists at L10" to "specific layer × head combinations are causally
necessary". Identify the smallest set of components whose activation patching
restores or destroys physics-mode behavior.

**Entry points**:
- `src/physical_mode/probing/steering.py` — extend with patching helpers.
- new `src/physical_mode/probing/patching.py` — clean/corrupted forward
  replay.
- new `scripts/07_sip_patching.py`.

**What's needed**:
1. Construct Semantic Image Pairs from the M2 factorial. For each pair the
   image differs along exactly one axis (e.g., `shaded_ground_none` vs
   `line_ground_none` on the same seed). Emit `sip_manifest.parquet` with
   columns `(clean_id, corrupted_id, differing_axis)`.
2. Re-capture activations with `capture_lm_attentions=True` on the SIP
   subset (~120 stimuli × both members of each pair). New disk: ~15 GB.
3. **Activation patching**: forward both clean and corrupted, then replay
   corrupted with selected layers' visual-token activations replaced by the
   clean tensor. Measure the change in PMR-positive token probability
   ("indirect effect"). Use raw PyTorch hooks; nnsight is fine but not
   required.
4. **Attention knockout**: zero out specific (head, layer) attention from
   visual tokens to the last position. Measure PMR delta on the test stimuli.
5. **SAE** (stretch): train a SAE on the M3 vision-encoder activations
   following Pach et al. 2025. Identify monosemantic "cast_shadow", "ground",
   "shading" features. Intervene by clamping individual SAE features and
   measure behavioral PMR shift.

**Success criteria**:
- A single (layer, head) range whose attention knockout drops PMR by ≥ 10 pp
  on the SIP test set.
- An SAE feature whose ablation reproduces the M5a L10 D→B flip without an
  explicit steering vector.

## Sub-task 5 — Cross-model sweep (M6)

**Goal** (`references/project.md` §2.6). Replicate M1-M5 on LLaVA-1.5-7B,
LLaVA-Next-7B, InternVL2-8B, and (stretch) Qwen2-VL-7B. Test whether the
boomerang pattern (M3), the LM late-layer peak (M4), and the L10 steering
sweet spot (M5a) are universal or Qwen-specific.

**Entry points**:
- `configs/cross_model.py` (new).
- `scripts/02_run_inference.py` — extend to iterate over a model list.

**What's needed**:
1. Add `system_prompt_override: str | None` to `EvalConfig` for Gavrikov
   et al. 2024-style prompt steering ("treat this as an abstract geometric
   shape" vs "treat this as a physical object subject to gravity").
2. Iterate inference over a list of model_ids; outputs go to
   `outputs/cross_model_<model>_<ts>/`.
3. For each model, run abridged ST3 (LM probe at ~5 layers) and ST4 (M5a
   steering at the analog of L10).
4. Disk budget: ~60 GB total model downloads (LLaVA-1.5-7B ~13 GB,
   LLaVA-Next-7B ~14 GB, InternVL2-8B ~16 GB, Qwen2-VL-7B ~15 GB).

**Success criteria**:
- `H-boomerang` confirmed on at least 2 additional models (encoder AUC
  high, behavioral PMR varies).
- The "L10 sweet spot" is replicated, OR the per-model sweet-spot indices
  cluster around 30-40 % of LM depth.

## Stimulus extensions (any round)

- **Photorealistic axis A level 5**: add `src/physical_mode/stimuli/diffusion.py`
  that calls FLUX.1-schnell (pattern in
  `vlm_anchroing/scripts/generate_irrelevant_number_images.py`) to render
  `textured_photo` variants. Filter via a CLIP-similarity threshold to
  "a ball" vs "a circle drawing". Critical for testing whether the M3
  encoder AUC = 1.0 finding is an artifact of the programmatic stimulus
  ceiling or generalizes.
- **Blender 3D**: render controlled spheres with exact light directions for
  shape-from-shading experiments. Less urgent; do only if reviewers ask.
- **Block-stack as separate path** (`references/roadmap.md` §4.1): add
  `block_stack` to MVP-full's `object_levels`. Tests "physical object ≠ ball"
  axis.

## Behavioral / methodological extensions

- **Label-free prompt** (`references/roadmap.md` §4.9): `prompts.py`
  `open_no_label` variant that asks "What do you see? What might happen
  next?" without the {label} slot. Disables H2 prior and gives M4
  switching-layer metric a chance to be meaningful.
- **Reverse prompting** (`references/roadmap.md` §4.2): attach `"abstract
  diagram"` label to a textured-ground-both stimulus. Counterfactual for H4.
- **Per-label regime annotation** (`references/roadmap.md` §4.11): zero-shot
  classify open-ended responses into 5 categories (gravity-fall / gravity-roll
  / orbital / inertial / static). Tests H7 quantitatively.

## Human baseline (optional, M7)

Sample 50 stimuli from the MVP-full set, collect 20 Prolific responses per
stimulus on the same open-ended prompt, compute human PMR. Report
human-vs-VLM alignment as a secondary headline figure. Required only for
the "ambitious" version of the paper (NeurIPS scope).
