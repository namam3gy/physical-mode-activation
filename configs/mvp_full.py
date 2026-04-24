"""MVP-full config — Sub-task 1 main run with pilot-informed revisions.

Changes vs pilot (see references/roadmap.md §3 M2 and docs/insights/m1_pilot.md §6):
  1. Axis C redesigned to decompose the saturating `arrow_shadow` cue:
     none / cast_shadow / motion_arrow / both (4 levels).
  2. Wind cue removed from the factorial (invisible to Qwen2.5-VL per §3.4).
  3. Axis D expanded to {"circle", "ball", "planet"} to quantify H2/H4.
  4. Event template reduced to {"fall"} — pilot showed fall≈horizontal in PMR.
  5. T=0.7 + 10 seeds/cell so RC is informative (H3-adjacent).
  6. LM hidden-state capture at 5 layers for Sub-task 3 reuse.
     Attention capture deferred to Sub-task 4.

Factorial size: 4 obj × 3 bg × 4 cue × 10 seeds × 1 event = 480 stimuli.
With 3 labels × 2 prompt variants → 2880 inferences.
Expected wall-clock on H200: ~1.5-2h including activation capture.
Disk cost (hidden states only, bf16): ~4-6 GB.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="mvp_full",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
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
    prompt_variants=("open", "forced_choice"),
    # Hidden states at 5 sampled layers — spans early/mid/late of Qwen2.5-VL-7B's
    # 28-layer LM. Sub-task 3 (logit lens) reuses these directly.
    capture_lm_layers=(5, 10, 15, 20, 25),
    capture_lm_attentions=False,  # flip to True for Sub-task 4 patching
    random_seed=42,
)
