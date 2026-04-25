"""§4.10 — Small attention-capture config for the visualization UI.

Captures LM hidden states + attentions on a representative subset of
M8a Qwen stimuli. The viz notebook (`notebooks/attention_viz.ipynb`)
loads these and renders per-layer × per-head attention heatmaps over
the visual-token grid.

Design choice — minimal subset:
- Qwen2.5-VL-7B-Instruct (saturated; project canonical model)
- Same M8a factorial as the labeled run, with limit=20 — covers a
  small mix of physics-mode and abstract-mode cells
- 4 representative LM layers: 5 (early), 15 (mid), 20 (late-mid), 25 (late)
- Both labeled (ball/circle/planet) and label-free arms — but limit=20
  applies to the manifest, so we'll get the first 20 stim × 4 labels
  ≈ 80 inferences

Capture cost: ~30 seconds inference + ~80 safetensors files of
attention tensors (each ~3MB for 28-layer × 28-head Qwen at typical
sequence length).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="attention_viz_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("circle", "square", "triangle", "hexagon", "polygon"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall",),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=(5, 15, 20, 25),
    capture_lm_attentions=True,
    random_seed=42,
    limit=20,  # representative subset, not the full 400
)
