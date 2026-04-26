"""§4.3 ext — Qwen2.5-VL with Japanese labels (ボール/円/惑星) on M8a circle stim.

Japanese mirror of `configs/sec4_3_korean_labels.py`. Tests whether the
multilingual label-prior pattern observed for Korean (cross-label
ordering preserved 4/5 models, magnitude bottlenecked by LM Korean
fluency) generalizes to Japanese — a different language with different
LM SFT coverage profiles per model.

Labels:
- ボール (booru) = ball (Katakana, common loanword)
- 円 (en) = circle (Kanji, also Chinese hanzi)
- 惑星 (wakusei) = planet (compound Kanji, lower training-data frequency
  than 'planet')

Same OPEN_TEMPLATE prompt as Korean run; English question with Japanese
label inserted. n_inferences = 80 × 3 = 240 on the M8a circle subset.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="sec4_3_japanese_labels_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("circle",),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall",),
        seeds_per_cell=5,
    ),
    labels=("ボール", "円", "惑星"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
