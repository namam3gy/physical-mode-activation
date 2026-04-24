# M3 — Vision encoder probing (2026-04-24)

- **Command**: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 --output-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations --layers 3,7,11,15,19,23,27,31`
- **Capture time**: ~70 s (10 it/s forward-only) + model load 20 s
- **Disk**: 12 GB (480 stimuli × 8 layers × (1296 tokens × 1280 dim × 2 bytes))
- **Probes**: sklearn LogisticRegression, StratifiedKFold (5), mean-pool token axis. Code: `src/physical_mode/probing/vision.py`
- **Outputs**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_vision/*.csv`
- **Deep dive**: `docs/insights/m3_encoder_boomerang.md`.

## Headline: the encoder-decoder boomerang is real and **saturated**

**Stimulus-property probes (AUC by layer × target)**: the vision encoder linearly separates **every** factorial axis at AUC=1.00 from layer 3 onward:

| target | layer 3 | 15 | 31 |
|---|---|---|---|
| y_bg_ground (bg != blank) | 1.00 | 1.00 | 1.00 |
| y_bg_scene | 1.00 | 1.00 | 1.00 |
| y_obj_3d (shaded/textured) | 1.00 | 1.00 | 1.00 |
| y_obj_textured | 1.00 | 1.00 | 1.00 |
| y_cue_any | 1.00 | 1.00 | 1.00 |
| y_cue_shadow | 1.00 | 1.00 | 1.00 |
| y_cue_arrow | 1.00 | 1.00 | 1.00 |

→ **Any downstream system with linear access to these features could recover every factorial axis perfectly.** The encoder has zero information bottleneck on the stimulus descriptor.

Meanwhile, forced-choice **behavioral PMR** on the same stimuli:

| axis | level | beh. PMR |
|---|---|---|
| bg | blank / ground / scene | 0.51 / 0.71 / 0.77 |
| object | line / filled / shaded / textured | 0.58 / 0.65 / 0.71 / 0.71 |
| cue | none / cast_shadow / motion_arrow / both | 0.28 / 0.49 / 0.93 / 0.95 |

The **LM is imperfectly sensitive** to properties the encoder perfectly encodes. The gap is entirely downstream of the visual features.

## Controlled probe: the no-cue subset

Restrict to `cue_level=none` (120 stimuli, no red-arrow shortcut, no cast shadow). Train encoder probe on behavioral forced-choice PMR.

| layer | encoder AUC | behavioural PMR |
|---|---|---|
| 3 | 0.793 | 0.28 |
| 11 | 0.852 | 0.28 |
| **19** | **0.890** | 0.28 |
| 27 | 0.852 | 0.28 |
| 31 | 0.859 | 0.28 |

The encoder's layer-19 activations determine which "no-cue" stimuli the LM will call physics-mode with **AUC 0.89** — yet only **28 %** of those stimuli actually trigger physics-mode in behavior. Read as calibration: *"encoder knows which cells trigger physics-mode, but LM only lets through a fraction of those."*

## Per-object-level encoder AUC vs behavior (forced-choice)

| object_level | encoder AUC @ L31 | behavioural PMR | gap |
|---|---|---|---|
| line | 0.944 | 0.583 | **+0.361** |
| filled | 0.950 | 0.647 | +0.303 |
| shaded | 0.943 | 0.711 | +0.232 |
| textured | 0.952 | 0.714 | +0.238 |

Boomerang gap is **largest for the most abstract object** — consistent with H4 (the language-prior-vs-vision tension worsens with abstraction).

## Methodological caveat

Programmatic stimuli make encoder AUC 1.0 trivially attainable (a simple mean-pooled representation is enough). The sharpness of the 1.0 vs behavioral-PMR gap is evidence that **when the information is unambiguously present, the LM still fails to route it into physical-mode behavior** — but on photographic stimuli the encoder side may itself be imperfect, and the effect size on real-world inputs needs cross-validation. M6 (cross-model) and the axis A photorealistic extension will address this.

## Hypothesis scorecard post-M3

| H | status after M2 | status after M3 | change |
|---|---|---|---|
| **H-boomerang** | candidate (project doc §1.4) | **supported (saturated evidence)** | encoder AUC 1.0 on every axis; behavioral PMR 0.28-0.95 |
| H4 (open-forced gap) | supported | **supported + mechanism** | Per-object-level encoder AUC ~ constant (~0.95) while behavior varies (0.58-0.71); gap concentrated in LM |
| H6 (shadow standalone) | supported (revised) | **supported** | Encoder AUC=1.0 on y_cue_shadow → information full; LM uses it partially (0.49 PMR) |

## Unlocks

- **Sub-task 3 (M4) now has full machinery ready**: LM hidden states at 5 layers (from M2) + vision hidden states at 8 layers (from M3). Logit lens on LM hidden states at visual-token positions is the next natural figure.
- **Sub-task 4 (M5, activation patching)** prereq: need attention capture. Flip `capture_lm_attentions=True` and rerun mini-batch.
- **Photorealistic stimulus extension** (additional idea) will test whether the boomerang survives when encoder AUC is not saturated.
