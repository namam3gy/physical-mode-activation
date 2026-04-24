# Next steps — Sub-tasks 2-5 (not implemented this round)

Keep this file up to date as work progresses. Each section is a concrete
plug-in point; the scaffolding already exists.

## Sub-task 2 — Vision-encoder probing

**Goal** (research_plan.md §2.3). Train linear probes on vision-encoder
activations against behavioral PMR labels. Target: show that the encoder
already separates "physical-like" from "geometric-like" inputs even when
the LLM doesn't pick it up — the "encoder-decoder boomerang" claim.

**Entry point**: `src/physical_mode/probing/vision.py` (empty).

**What's needed**:
1. Extend `PhysModeVLM.capture()` to also record vision-encoder layer
   activations. Qwen2.5-VL's vision tower is under `model.visual`; for
   Qwen2-VL under `model.visual.blocks[i]`; for LLaVA under
   `model.vision_tower.vision_model.encoder.layers[i]`. Write a
   `resolve_vision_layer_path(model)` helper that returns the right list
   given the loaded model instance.
2. Run the MVP-full config with `capture_vision_layers=(5, 10, 15, 20, 22)`
   set (add the field to `EvalConfig`). Activations land in
   `outputs/<run_id>/activations/<sample_id>.safetensors` under keys
   `vision_hidden_<layer>`.
3. In `probing/vision.py`, write:
   - `load_dataset(run_dir) -> (X: n_samples × d, y: PMR labels)` by
     joining captured activations with `predictions_scored.parquet`.
   - `train_probes(X_per_layer, y)` with `LogisticRegression(max_iter=500)`,
     5-fold stratified CV; report AUC per layer.
   - Optional: head-wise probes per Gandelsman et al. 2024 — split each
     layer's attention output by head and probe each 128-dim slice.
4. Report plot: AUC-per-layer × object_level — the boomerang appears as
   high AUC on `shaded`/`textured` layers diverging from the flatter
   behavioral PMR curve.

## Sub-task 3 — LLM backbone probing + logit lens

**Goal** (research_plan.md §2.4). Find the *layer* at which physics-mode
emerges inside the LM. Target: match Neo et al. 2024 finding that
object-specific features crystallize in layers 15-24 of LLaVA-1.5.

**Entry point**: `src/physical_mode/probing/lm.py` (empty).

**What's needed**:
1. MVP-full run already captures `lm_hidden_<layer>` at the visual-token
   positions. Use those directly.
2. **Per-layer probes**: same approach as Sub-task 2, but on the LM hidden
   states. Train a probe at each captured layer on (a) PMR binary, (b)
   "next-motion verb" 5-way (from a hand-annotated slice of the predictions).
3. **Logit lens**: for each captured layer, apply `model.lm_head` to the
   hidden state at the last visual-token position and read out top-k token
   logits. Track the layer-wise logit trajectory of:
   - Physics verbs: `fall`, `roll`, `bounce`, `drop`
   - Geometry nouns: `circle`, `shape`, `line`, `drawing`
   The layer where physics-verb logits first exceed geometry-noun logits is
   the "switching layer". Expected to depend on object_level.

## Sub-task 4 — Causal localization (NOTICE / VTI / SAE)

**Goal** (research_plan.md §2.5). Move from correlation to causation.

**What's needed**:
1. **Semantic Image Pairs**: from the factorial, pick pairs that differ in
   exactly one axis level (e.g., `shaded_ground_none` vs `line_ground_none`
   on the same seed). Emit a `sip_manifest.parquet` listing
   (clean_id, corrupted_id, differing_axis).
2. **Activation patching**: pass both images through the model, capture all
   layers, then replay the corrupted image's forward pass with selected
   layers' activations replaced by the clean image's activations. Measure
   change in PMR probability. Implementation uses forward-hook replacement;
   nnsight or raw PyTorch hooks both work.
3. **VTI steering vectors**: `v = mean(h_clean) - mean(h_corrupted)` per
   layer. At test time add `alpha * v` to the residual stream at a chosen
   layer and rerun generation. Measure PMR shift on *unseen* stimuli.
4. **SAE features** (stretch): train a sparse autoencoder on the captured
   vision-encoder activations and identify monosemantic "shading" /
   "ground-plane" features per Pach et al. 2025.

## Sub-task 5 — Cross-model sweep + prompt steering

**Goal** (research_plan.md §2.6). Replicate across LLaVA-1.5-7B,
LLaVA-Next-7B, InternVL2-8B, Qwen2-VL-7B and compare prompt steering
("treat this as abstract" vs "treat this as physical").

**What's needed**:
1. Add a new `EvalConfig` field `system_prompt_override: str | None` so a
   config can swap the default system prompt for a steering variant.
2. Copy `configs/mvp_full.py` → `configs/cross_model.py` with a list of
   `model_id`s and a loop over them in `scripts/02_run_inference.py`.
3. Only changes needed to the code are in the `02_run_inference.py` driver;
   `PhysModeVLM` already handles arbitrary HF model ids.

## Stimulus extensions (any round)

- **Photorealistic axis A level 5**: add `diffusion.py` that calls
  FLUX.1-schnell (pattern in
  `vlm_anchroing/scripts/generate_irrelevant_number_images.py`) to render
  `textured_photo` variants. Filter via a CLIP-similarity threshold to
  "a ball" vs "a circle drawing".
- **Blender 3D**: render controlled spheres with exact light directions
  for shape-from-shading experiments.

## Human baseline (optional)

Sample 50 stimuli from the MVP-full set, collect 20 Prolific responses per
stimulus on the same open-ended prompt, compute human PMR. Report
human-vs-VLM alignment as a secondary headline figure.
