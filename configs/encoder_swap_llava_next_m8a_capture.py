"""§4.6 cross-model — LLaVA-Next-Mistral M8a-stim run with LM captures.

Re-runs the M6 r6 LLaVA-Next M8a behavioral protocol with
`capture_lm_layers=(5, 10, 15, 20, 25)` enabled. M2-derived v_L was
class-imbalanced (n_neg=9); M8a has n_neg=279 (~23%), clean.

Inference: 1200 + 400 capture calls. ~30-50 min on H200.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_llava_next_m8a_capture",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
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
    capture_lm_layers=(5, 10, 15, 20, 25),
    capture_lm_attentions=False,
    random_seed=42,
)
