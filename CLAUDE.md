# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Sub-task 1 MVP implemented. Package at `src/physical_mode/`, entry scripts at
`scripts/0{1,2,3}_*.py`, configs at `configs/{pilot,mvp_full}.py`, tests at
`tests/`. Read `docs/00_architecture.md` **first** — it is the implementation
contract and names every module. Read `research_plan.md` (Korean) for the
scientific motivation.

Sub-tasks 2-5 (probing, logit lens, causal patching, multi-model sweep) are
scaffolded but not implemented. See `docs/04_next_steps.md` for concrete
plug-in points.

This project sits alongside `vlm_anchroing/`, `eval_sufficiency/`, and
`agent_orchestration/` under `/mnt/ddn/prod-runs/thyun.park/src/`. The parent
`../CLAUDE.md` covers workspace-wide conventions (each subproject is its own
git repo with its own `.venv/` and `uv.lock`; always `uv run` from inside the
project dir).

## Research intent (summary)

Canonical spec: `research_plan.md`. One-sentence: at what visual-cue threshold
does an open-source VLM stop processing an abstract shape (circle) as geometry
and start processing it as a physical object (ball)?

- **Target model (this round)**: Qwen2.5-VL-7B-Instruct. Matches
  `eval_sufficiency/`'s proven setup; plan §2.6 lists LLaVA-1.5, LLaVA-Next,
  Qwen2-VL, InternVL2 for the multi-model round.
- **Five sub-tasks**: PhysCue (behavioral thresholds), vision-encoder probing,
  LLM logit-lens / layer-wise probing, causal patching / SIP / VTI / SAE,
  cross-model + prompt-steering. This round covers only the first.
- **Metrics**: PMR (physics-mode priming rate), GAR (gravity-align rate),
  RC (response consistency). Definitions in `docs/02_scoring_rubric.md`.
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
- Append a new entry to `docs/03_run_log.md` after each real run, with
  the exact command, wall-clock, and headline PMR-by-object_level table.
