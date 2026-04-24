# Architecture — Physical-Mode Activation

**Audience**: future Claude Code sessions working on this project. Read this
before touching code. The canonical research spec is `references/project.md`
(English; Korean translation at `references/project_ko.md`); this doc is the
*implementation* contract that translates §2.2 of that plan into code.

## One-paragraph summary

We render a controlled factorial of programmatic images (circles and block
stacks rendered with varying abstraction, background, and context cues), query
an open-source VLM with both open-ended and forced-choice next-state-prediction
prompts, and score the responses for three behavioral metrics (PMR, GAR, RC).
Hidden states and attentions can be captured during inference so that layer-wise
probes (Sub-task 2) and logit-lens / causal-patching analyses
(Sub-tasks 3-4) can be added later without re-running inference.

## Module map

```
src/physical_mode/
├── config.py         # EvalConfig + FactorialSpec + StimulusRow dataclasses
├── utils.py          # seed, timestamp, config_hash, WORKSPACE constant
├── stimuli/
│   ├── primitives.py # PIL drawers: circles, shading, ground, wind, arrow, shadow
│   ├── scenes.py     # compose(row) -> PIL.Image
│   └── generate.py   # write inputs/<run_id>/{images/, manifest.parquet}
├── models/
│   └── vlm_runner.py # PhysModeVLM — AutoModelForImageTextToText + hidden-state capture
├── inference/
│   ├── prompts.py    # open-ended + forced-choice templates, label-parameterized
│   └── run.py        # main loop: stream predictions.jsonl + save activations
├── metrics/
│   ├── lexicons.py   # PHYSICS_VERB_STEMS, DOWN_DIRECTION_PHRASES, ABSTRACT_MARKERS
│   └── pmr.py        # score_pmr / score_gar / score_rc / summarize
└── probing/          # Sub-tasks 2-4 modules
    ├── vision.py     # ST2 — vision-encoder linear probes
    ├── lm.py         # ST3 — LM logit lens + per-layer probes
    └── steering.py   # ST4 — VTI steering vectors
```

Scripts at `scripts/0{1,...,6}_*.py` are thin argparse wrappers around the
library. Configs at `configs/*.py` are Python files exposing a `CONFIG = EvalConfig(...)`
binding; scripts load them via `importlib.util.spec_from_file_location` so
configs can reference the in-tree types without going through a separate
registry.

## Factorial axes (from `references/project.md` §2.2, reduced)

| Axis | Code | Levels |
|---|---|---|
| A — object abstraction | `object_level` | `line` · `filled` · `shaded` · `textured` · `block_stack` |
| B — background | `bg_level` | `blank` · `ground` · `scene` |
| C — context cue | `cue_level` | `none` · `cast_shadow` · `motion_arrow` · `both` (legacy: `wind`, `arrow_shadow`) |
| D — object label (prompt-time) | `label` | `circle` · `ball` · `planet` · `shape` · `object` |
| E — scene consistency | (future) | not manipulated in this round |

Five event templates (`fall`, `horizontal`, `hover`, `wall_bounce`, `roll_slope`)
control the object's on-canvas position; the first two are used by `pilot.py`
and `mvp_full.py`.

## Data flow

```
FactorialSpec.iter()
  → StimulusRow (sample_id, factor levels, seed)
  → render_scene(row) → PIL.Image
  → inputs/<run_id>/images/<sample_id>.png + manifest.parquet
       │
       ▼
PhysModeVLM(model_id, ...)
  for each (stimulus × label × prompt_variant):
    .generate(image, rendered_prompt, choice_tokens) → {raw_text, token_info, option_logits}
  for each stimulus (once):
    .capture(image, rendered_prompt) → hidden_states + attentions
  ↓
outputs/<run_id>/
  ├── predictions.jsonl    (streamed per-inference)
  ├── predictions.parquet  (flat, for analysis)
  ├── predictions.csv
  ├── activations/<sample_id>.safetensors  (only if capture_lm_layers is set)
  └── run_meta.json
       │
       ▼
score_rows → pmr, gar, hold_still, abstract_reject
summarize  → summary_{overall, by_object_level, by_bg_level, ...}.csv
response_consistency → response_consistency.csv
       │
       ▼ (probing follow-ups)
probing.vision   → vision-encoder layer probe AUC
probing.lm       → LM-layer probe AUC, logit lens trajectories
probing.steering → VTI direction extraction + residual-stream injection
```

## Key design choices

1. **Generic `AutoModelForImageTextToText`**, not `Qwen2_5_VLForConditionalGeneration`.
   Matches the `vlm_anchroing/src/vlm_anchor/models.py:39-55` pattern. Switching to
   LLaVA-1.5 or InternVL2 in a later round is a config-only change.
2. **Config as Python, not YAML.** Configs parameterize a dataclass with typed
   literals; YAML would either lose the type info or require a schema. The
   eval-sufficiency pattern (`EvalConfig(...)` constructed in a driver) is
   adopted directly.
3. **Streamed JSONL output.** `predictions.jsonl` is flushed per inference so
   a crash 3 hours into a 6-hour run doesn't wipe the completed rows.
   Parquet / CSV are materialized once at the end.
4. **Activation capture is optional and *prompt-agnostic*.** When enabled, the
   capture forward pass uses the `open` prompt with `labels[0]` — this keeps
   every capture comparable across stimuli regardless of which inference
   prompt triggered physics-mode. For MVP-scale runs, disk cost is
   ~7 MB per stimulus for 5 layers × bf16, so ~7 GB for a 1k-stimulus run.
5. **PMR scoring is deliberately dumb.** Stem-prefix matching + abstract-reject
   veto. False negatives are expected and must be caught by expanding
   `lexicons.PHYSICS_VERB_STEMS` — see `docs/scoring_rubric.md`.
6. **No model-in-the-loop tests in CI.** `tests/` only covers pure-Python
   stimulus determinism and PMR scoring. Running a VLM smoke is done manually
   via `scripts/02_run_inference.py --limit 5`.

## Commands cheat sheet

```bash
# First time.
uv sync

# Generate stimuli.
uv run python scripts/01_generate_stimuli.py --config configs/pilot.py

# Smoke inference on 5 stimuli.
uv run python scripts/02_run_inference.py --config configs/pilot.py --limit 5

# Full pilot (~30-60 min on H200).
uv run python scripts/02_run_inference.py --config configs/pilot.py

# Score + summarize.
uv run python scripts/03_score_and_summarize.py --run-dir outputs/pilot_<ts>_<hash>

# Vision-encoder activation capture (M3).
uv run python scripts/04_capture_vision.py --stimulus-dir inputs/<run> --output-dir outputs/<run>/vision_activations --layers 3,7,11,15,19,23,27,31

# LM logit lens + per-layer probes (M4).
uv run python scripts/05_lm_probing.py --run-dir outputs/<run>

# VTI steering causal intervention (M5).
uv run python scripts/06_vti_steering.py --run-dir outputs/<run> --stimulus-dir inputs/<run> --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40

# Tests.
uv run python -m pytest
```

## What's next (see `docs/next_steps.md` and `references/roadmap.md` §3)

- M5b: SIP activation patching + SAE feature decomposition.
- M6: cross-model sweep (LLaVA-1.5 / LLaVA-Next / InternVL2).
- Photorealistic / 3D stimuli (axis A level 5, Blender).
- Human baselines (ROADMAP §4).
