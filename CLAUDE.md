# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Scaffold only. `uv init` created `main.py`, `pyproject.toml`, `README.md` (empty),
and `.python-version` (3.11). No implementation exists yet. Dependencies in
`pyproject.toml` is currently `[]` — add packages via `uv add <pkg>` rather
than editing the file by hand.

This project sits alongside `vlm_anchroing/`, `eval_sufficiency/`, and
`agent_orchestration/` under `/mnt/ddn/prod-runs/thyun.park/src/`. The parent
`../CLAUDE.md` covers workspace-wide conventions (each subproject is its own
git repo with its own `.venv/` and `uv.lock`; always `uv run` from inside the
project dir). Do not reuse sibling projects' venvs here.

## Research intent

The canonical spec is `research_plan.md` (Korean). **Read it before writing
code** — it is the source of truth for the experimental design, hypotheses,
and venue framing. Short English summary so future Claude instances can
orient quickly:

- **Research question**: at what visual-cue threshold does an open-source VLM
  stop processing an abstract shape (circle) as geometry and start processing
  it as a physical object (ball)?
- **Target models**: LLaVA-1.5 / LLaVA-Next, Qwen2-VL, InternVL2 (the same
  open-source trio as Pixels-to-Principles, Ballout et al. 2025).
- **Five sub-tasks** (§2 of the plan):
  1. `PhysCue` — controlled factorial stimulus set (abstraction × background ×
     cue × label × scene-consistency) + next-state-prediction prompts.
     Behavioral metrics: PMR (physics-mode priming rate), GAR (gravity-align
     rate), RC (response consistency).
  2. Vision-encoder probing (CLIP-ViT / SigLIP / InternViT) — layer- and
     head-wise linear probes on PMR labels; Gandelsman-style head
     decomposition; Pach et al. SAE features.
  3. LLM-backbone layer-wise emergence — logit lens + per-layer probes at
     visual-token positions following Neo et al. 2024.
  4. Causal localization — Semantic Image Pairs (NOTICE recipe) + activation
     patching / attention knockout / VTI-style steering vectors / SAE
     interventions.
  5. Cross-model + prompt-steering generalization (Gavrikov et al. 2024 style).
- **Headline claims being set up**: S-shaped switching curve across
  abstraction axis; encoder-decoder "boomerang" (probe AUC high while
  behavioral PMR lags); a small number of causally necessary "physics heads";
  a residual-stream steering direction that forces physics-mode on line
  drawings.
- **Venue framing**: EMNLP long (grounding angle) as primary, NeurIPS main
  (interpretability angle) as stretch. See §3.1.

When the user asks for a "minimum viable" vs "ambitious" scope, map to
§2.7 of the plan (subset of axes + single model for MVP; full 5 subtasks +
SAE + 5-model sweep for the ambitious version).

## Commands

```bash
uv sync                          # create .venv and install
uv add <pkg>                     # add a dependency
uv run python main.py            # run the current stub
uv run python -m pytest          # no tests exist yet
```
