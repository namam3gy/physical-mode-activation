# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project rules (load every session)

These are the hard constraints set by the user — follow them automatically without being asked each time.

1. **Use Context7 MCP for library / API documentation, code generation, setup or configuration steps**, even when the user does not explicitly ask. Anytime you would have answered from training data about a library API (transformers, sklearn, torch, diffusers, sentencepiece, etc.), call Context7 first to get current docs.
2. **Always run Python through `uv`** (`uv run python ...`, `uv run python -m pytest`, `uv add <pkg>`). Never call bare `python` or `pip` from this project.
3. **Adhere to the project file structure** (see "Repository layout" below). New artifacts go into the matching directory; do not invent ad-hoc top-level folders.
4. **`references/` is the staging area for materials useful during work** — papers, raw spec PDFs, downloaded datasets schemas, etc. `project.md` and `roadmap.md` always live here.
5. **`references/project.md` is the research plan; `references/roadmap.md` is the living execution doc derived from it.** At every step, refer to both. Update `roadmap.md` (status table, hypothesis scorecard, additional ideas, change log) whenever a milestone completes or a finding contradicts a hypothesis. Treat `project.md` as read-only; major spec revisions go into the roadmap.
6. **Bilingual file requirement**: every file in `references/{project,roadmap}.md` and `docs/insights/*.md` MUST have a paired Korean translation `*_ko.md`. English is canonical; translations follow English. Other docs (`docs/architecture.md`, `docs/experiments/*.md`, etc.) may be bilingual but are not required to be.
7. **After completing hypothesis validation or implementation, write a reproduction notebook** at `notebooks/<slug>.ipynb` that runs cell-by-cell and reproduces the result. Use the existing `notebooks/demo.ipynb` (M1) as the template.
8. **Work in English; speak to the user in Korean.** All committed code, comments, docstrings, commit messages, and English markdown files are English. All terminal-visible messages to the user (chat replies, status updates, summaries) are Korean. Mid-sentence English technical terms are fine.

## Status

**Read `references/roadmap.md` at the project root FIRST.** It is the single
source of truth for "what milestone we're on, what's next, what hypotheses
have been tested, what ideas are still open." Update it when a milestone
completes or a new hypothesis/idea surfaces.

Sub-task 1 MVP, ST2 vision-encoder probing, ST3 LM logit lens, ST4
Phase-1+2 VTI steering, M5a-ext VTI follow-ups, M4b label-free H2
null test, M6 round 1 (LLaVA-1.5-7B cross-model), and M4c FC
label-free all complete (M0 through M6 r1 + M4c — see ROADMAP §3).
Key recent findings (2026-04-25):
- M5a-ext: `v_L10` is a **regime axis within physics-mode** (+α →
  A/kinetic, −α → B/static, baseline D below |α| threshold).
- M4b + M6 r1 + M4c: H2 unified under the **visual-saturation
  hypothesis** — language prior is positive across labels; Qwen's
  PMR(_nolabel) ≈ 0.95 saturation masks it (M4b sees only `circle`
  suppression; M4c FC adds a planet-suppression artefact from the
  gravity-centric option set), while LLaVA-1.5's lower visual prior
  recovers the original H2 (`ball +47.5 pp` paired delta vs no-label).
- LLaVA-1.5 FC pathology: `first_letter` = `A` for 477/480 stimuli even
  with re-templated label-free options. Confirmed model-level bias.

Package at `src/physical_mode/`, entry scripts at `scripts/0{1..6}_*.py`,
configs at `configs/{pilot,mvp_full,label_free,cross_model_llava{,_label_free},fc_label_free_{qwen,llava}}.py`,
tests at `tests/`. Read `docs/architecture.md` for the implementation
contract and `references/project.md` for the original scientific
motivation.

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

Bilingual `*_ko.md` Korean translations are **required** for `references/{project,roadmap}.md` and every file under `docs/insights/`. They are also currently provided as a courtesy for the rest of `docs/*.md` and `docs/experiments/*.md`, but those are not required by the project rules; English is canonical regardless.

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

# Walkthrough notebooks (cell-by-cell reproductions).
uv run jupyter lab notebooks/demo.ipynb               # M1 pilot + general pipeline tour
uv run jupyter lab notebooks/m5_vti_steering.ipynb    # M5a VTI causal steering reproduction
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
