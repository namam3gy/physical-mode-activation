"""M6 round-1 cross-model — LLaVA-1.5-7B labeled run.

Re-runs the M2 factorial on the same stimuli with LLaVA-1.5-7B-hf. Primary
goal: test whether the M4b H2 revision (`ball ≈ no-label`, `circle =
suppressor`) replicates beyond Qwen2.5-VL.

Forced-choice is **excluded** in this round: a smoke test on LLaVA-1.5
showed that the model returns "A" for every (image, label) FC combination
(12/12 cells), making FC PMR uninformative. Open prompts give diverse
label/image-sensitive responses suitable for cross-model H2 + H7 testing.

Matched to M2 / M4b for paired comparability:
- factorial: 4 obj × 3 bg × 4 cue × 1 event × 10 seeds = 480 stimuli
- temperature: 0.7, top_p: 0.95
- labels: ball, circle, planet
- max_new_tokens: 96

Inference size: 480 stim × 3 labels × 1 prompt = 1440 inferences.
Expected wall-clock on H200: ~25-35 min.

Stimuli reused via `--stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3`.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground", "scene"),
        cue_levels=("none", "cast_shadow", "motion_arrow", "both"),
        event_templates=("fall",),
        seeds_per_cell=10,
    ),
    labels=("circle", "ball", "planet"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
