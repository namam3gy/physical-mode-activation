"""C6-alt-A (2026-05-01) — M8d subtype/style label extension on idefics2-8b.

Same 3 shapes (car/person/bird) as M8d, but uses 5 labels per shape
(physical / abstract / exotic / **subtype** / **style**) to test whether
sub-category labels ("sedan", "eagle") behave the same as broad-category
labels ("car", "bird"), and whether "style" labels ("cartoon", "sketch")
suppress PMR like "silhouette" does.

Inference: 3 shapes × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds = 480 stim
× 5 labels × 1 open prompt = 2400 inferences.
"""
from __future__ import annotations
from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_subtype_idefics2",
    model_id="HuggingFaceM4/idefics2-8b",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic", "subtype", "style"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
