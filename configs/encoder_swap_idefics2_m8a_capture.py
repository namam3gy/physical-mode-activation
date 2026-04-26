"""§4.6 cross-model — Idefics2 M8a-stim run with LM activation captures.

Re-runs the §4.5 Idefics2 M8a behavioral protocol with
`capture_lm_layers=(5, 10, 15, 20, 25)` enabled, so per-model v_L
extraction (for §4.6 cross-model proper test) is possible from M8a
stim. M2-derived v_L was class-imbalanced (n_neg=5); M8a has
n_neg=204 (~17%), much cleaner.

Inference: 5 shapes × 4 obj × 2 bg × 2 cue × 1 event × 5 seed × 3
labels = 1200 inferences + 400 once-per-stimulus capture calls.
Expected wall-clock on H200: ~30-50 min.
Disk: ~10-15 GB safetensors (LM hidden states only, bf16).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_idefics2_m8a_capture",
    model_id="HuggingFaceM4/idefics2-8b",
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
