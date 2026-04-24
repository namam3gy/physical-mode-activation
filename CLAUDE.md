# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

**Read `references/roadmap.md` at the project root FIRST.** It is the single
source of truth for "what milestone we're on, what's next, what hypotheses
have been tested, what ideas are still open." Update it when a milestone
completes or a new hypothesis/idea surfaces.

Sub-task 1 MVP, ST2 vision-encoder probing, ST3 LM logit lens, and ST4
Phase-1+2 VTI steering all complete (M0 through M5a — see ROADMAP §3).
Package at `src/physical_mode/`, entry scripts at `scripts/0{1..6}_*.py`,
configs at `configs/{pilot,mvp_full}.py`, tests at `tests/`. Read
`docs/architecture.md` for the implementation contract and
`references/project.md` for the original scientific motivation.

ST4 Phase 3 (SIP activation patching + SAE) and ST5 (cross-model sweep) are
the next milestones. See `docs/next_steps.md` for code-level plug-in points
and `references/roadmap.md` §3 for milestone-level framing.

## Repository layout

```
references/    project.md, roadmap.md (+ Korean *_ko.md translations)
docs/
  figures/     — embedded figure assets (PNG)
  insights/    — per-milestone deep-dive markdown (m1_pilot, m3_..., m4_..., m5_...)
  experiments/ — per-milestone run logs with raw numbers (m1_pilot, m2_..., m3_..., m4_..., m5_...)
  architecture.md, stimulus_spec.md, scoring_rubric.md, next_steps.md
configs/       Python config files (pilot.py, mvp_full.py)
scripts/       Argparse runners (01..06)
src/           physical_mode package
tests/         pytest suite (no model-in-the-loop)
notebooks/     demo.ipynb walkthrough
```

All `*.md` files except CLAUDE.md and README.md have `*_ko.md` Korean
translations. English is canonical; if translations drift, the English
version wins.

## Research intent (summary)

Canonical spec: `references/project.md`. One-sentence: at what visual-cue
threshold does an open-source VLM stop processing an abstract shape (circle)
as geometry and start processing it as a physical object (ball)?

- **Target model (this round)**: Qwen2.5-VL-7B-Instruct.
- **Five sub-tasks**: PhysCue (behavioral thresholds), vision-encoder
  probing, LLM logit-lens / layer-wise probing, causal patching / SIP / VTI
  / SAE, cross-model + prompt-steering. M5b (Phase 3 of ST4) and ST5 are
  the remaining milestones.
- **Metrics**: PMR (physics-mode priming rate), GAR (gravity-align rate),
  RC (response consistency). Definitions in `docs/scoring_rubric.md`.
- **Venue framing**: EMNLP long primary, NeurIPS stretch.

## Commands

```bash
uv sync                                   # install deps (uses pytorch-cu130 index)

# Unit tests (no model required).
uv run python -m pytest

# Generate stimuli for a config.
uv run python scripts/01_generate_stimuli.py --config configs/pilot.py

# Smoke: 5-stimulus end-to-end with the 15 GB Qwen2.5-VL-7B download on first run.
uv run python scripts/02_run_inference.py --config configs/pilot.py --limit 5

# Full pilot (~30-60 min on H200).
uv run python scripts/02_run_inference.py --config configs/pilot.py

# Score + summarize.
uv run python scripts/03_score_and_summarize.py --run-dir outputs/<run_id>

# Vision-encoder activation capture (M3).
uv run python scripts/04_capture_vision.py --stimulus-dir inputs/<run> --output-dir outputs/<run>/vision_activations --layers 3,7,11,15,19,23,27,31

# LM logit lens + per-layer probes (M4).
uv run python scripts/05_lm_probing.py --run-dir outputs/<run>

# VTI steering causal intervention (M5).
uv run python scripts/06_vti_steering.py --run-dir outputs/<run> --stimulus-dir inputs/<run> --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40

# Walkthrough notebook with plots + demo inferences.
uv run jupyter lab notebooks/demo.ipynb
```

## Key conventions

- Configs are Python files exposing `CONFIG = EvalConfig(...)`, not YAML.
  Loaded via `importlib.util` in scripts.
- Outputs always triple: `predictions.jsonl` (streamed, crash-safe) +
  `predictions.parquet` (flat) + `predictions.csv`.
- `inputs/` and `outputs/` are gitignored. Activation `.safetensors` are
  big — gitignored too.
- VLM loading uses generic `AutoModelForImageTextToText` + `AutoProcessor`
  so the same `PhysModeVLM` class works across Qwen/LLaVA/InternVL families.
  Do not hardcode `Qwen2_5_VLForConditionalGeneration` — it breaks the
  cross-model plan for Sub-task 5.
- After each real run, append a per-run entry to
  `docs/experiments/m{N}_<slug>.md` (English canonical; mirror in `_ko.md`).
  Major milestone completions also get a `docs/insights/m{N}_<slug>.md`
  deep dive — see `references/roadmap.md` §5 for the convention.
