"""M-Add6 Pillar B fill-in — Pixtral 12B (Mistral-NeMo + 400M Pixtral encoder) on M8a stim.

6th non-Qwen data point for the encoder-saturation chain. Pixtral is a unique
non-CLIP non-SigLIP encoder (custom 400M-param vision tower trained jointly
with Mistral-NeMo-12B), so it tests whether non-CLIP saturation is family-
specific or generalizes.

Inference: 400 stim × 1 prompt × 1 label = 400 inferences (label-free arm).
Wall-clock on H200 (single GPU, bf16): ~3-5 min if all goes well; first run
may need ~5 min to download.

Note: Pixtral may require trust_remote_code or a custom processor. If
AutoModelForImageTextToText doesn't pick it up, the chain logs and skips.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m_add6_pixtral_m8a",
    model_id="mistral-community/pixtral-12b",
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
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
