# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project rules (load every session)

These are the hard constraints set by the user — follow them automatically without being asked each time.

1. **Use Context7 MCP for library / API documentation, code generation, setup or configuration steps**, even when the user does not explicitly ask. Anytime you would have answered from training data about a library API (transformers, sklearn, torch, diffusers, sentencepiece, etc.), call Context7 first to get current docs.
2. **Always run Python through `uv`** (`uv run python ...`, `uv run python -m pytest`, `uv add <pkg>`). Never call bare `python` or `pip` from this project.
3. **Adhere to the project file structure** (see "Repository layout" below). New artifacts go into the matching directory; do not invent ad-hoc top-level folders.
4. **`references/` is the staging area for materials useful during work** — papers, raw spec PDFs, downloaded datasets schemas, etc. `project.md` and `roadmap.md` always live here.
5. **`references/project.md` is the research plan; `references/roadmap.md` is the living execution doc derived from it.** At every step, refer to both. Update `roadmap.md` (status table, hypothesis scorecard, additional ideas, change log) whenever a milestone completes or a finding contradicts a hypothesis. Treat `project.md` as read-only; major spec revisions go into the roadmap.
6. **After completing hypothesis validation or implementation, write a reproduction notebook** at `notebooks/<slug>.ipynb` that runs cell-by-cell and reproduces the result. Use the existing `notebooks/demo.ipynb` (M1) as the template.
7. **Work in English; speak to the user in Korean.** All committed code, comments, docstrings, commit messages, and markdown files are English. All terminal-visible messages to the user (chat replies, status updates, summaries) are Korean. Mid-sentence English technical terms are fine.

## Status

**Read `references/roadmap.md` at the project root FIRST.** It is the single
source of truth for "what milestone we're on, what's next, what hypotheses
have been tested, what ideas are still open." Update it when a milestone
completes or a new hypothesis/idea surfaces.

**Headline status (2026-04-28)**: M0–M9 + §4.2/4.3/4.5/4.6/4.7/4.8/4.10/4.11
all complete. **M5b SAE intervention (full causal chain)** + **§4.6 5-model
n=10 layer sweep** + **§4.8 Qwen 7B-vs-32B PMR scaling** finished in the
2026-04-27 → 2026-04-28 overnight chain. **2026-04-28 afternoon-evening
suite**: §4.6 Idefics2 deeper-layer disambiguation (L26-L31) + cross-model
SAE training (4 non-Qwen models) + **scorer regression audit (R1)** + **M4
LM logit-lens cross-model** (5-model probe AUC) + **M5a runtime steering
cross-model** (3 of 4 testable models flip 10/10 — Idefics2 perceiver-
resampler 가설 forward/inverse dissociation 으로 refined) + **M5b SAE
intervention cross-model (round 2 — actually-consumed layer per model)**:
**3 of 5 models break PMR cleanly** (Qwen k=20, Idefics2 k=160, InternVL3
k=160 — random all 1.0); **2 LLaVA models NULL** at any k ≤ 160 (encoder
cluster). The 5-model M8a chain is locked (Qwen / LLaVA-1.5 / LLaVA-Next /
Idefics2 / InternVL3). See ROADMAP §1.3 hypothesis scorecard + §3 status
table for details.

Key recent findings (2026-04-27 → 04-28):
- **M5b SAE intervention (Qwen, full causal chain)**: ~20 vision-encoder
  physics-cue features (top-20 ablation breaks PMR 0/20; mass-matched
  random retains 20/20). Cohen's d re-ranking surfaces a tighter top-set
  (idx 1674 / mass 32.7) than delta-ranking did. Triangulation with M5b
  SIP + per-head knockout: **L9 MLP constructs the physics-mode
  commitment, L10 reads it out via redundant attention.**
- **§4.6 5-model n=10 layer sweep + Idefics2 9-layer disambiguation**:
  pixel-encodability is **architecture-conditional**. Qwen broad
  (5 shortcut layers ≥ 80 %), LLaVA-Next (L20+L25 100 %), LLaVA-1.5
  (L25 only, 40 % at n=10), **Idefics2 anomaly resolved (2026-04-28)**:
  9-layer test (L5-L31, 16-97 % depth) yields 0 clean shortcuts at any
  depth despite v_L projection ascending cleanly — **wrong-relative-
  depth falsified; perceiver-resampler is the leading remaining
  candidate** (controlled projector-swap test out of scope, since
  Idefics2 differs from MLP-projector models on encoder + projector +
  AnyRes simultaneously). InternVL3 protocol-saturated (baseline=1.0).
  Random controls 1/250 in aggregate. **H-shortcut framing**:
  pixel-encodability is architecture-conditional with **a projector-
  design candidate (MLP vs perceiver) emerging as the leading
  remaining axis** — encoder saturation alone is ruled out, projector
  isolation deferred.
- **§4.8 PMR scaling (Qwen 7B vs 32B on M2 open prompt)**: aggregate
  PMR 0.926 ≈ 7B 0.931 — **scale doesn't help PMR aggregate** (MechBench-
  style). But the per-cell `cue=none` PMR drops 8.6 pp (0.797 → 0.711)
  and `abstract_reject` jumps 35× (0.002 → 0.065): **scaling helps on
  the 5 % of cells where the cue is weakest**, exactly where the
  visual-prior under-weighting hypothesis predicts headroom exists.
  Label gap halves (`ball − circle` +0.071 → +0.010) — H2 weakened,
  not eliminated.
- **M4 LM logit-lens cross-model (2026-04-28)**: 5-model × 5-layer
  probe AUC reuses existing M2 cross-model captures (no new inference).
  AUC ladder: **Idefics2 0.995** > Qwen 0.96 > LLaVA-Next 0.79 > LLaVA-
  1.5 0.76 > InternVL3 untestable. Idefics2 LM AUC > vision AUC (0.93)
  → perceiver-resampler does NOT strip information. H-encoder-saturation
  gains 2nd downstream signature (LM probe AUC ladder).
- **M5a runtime steering cross-model (2026-04-28)**: 3 of 4 testable
  models flip PMR 10/10 — Qwen L10 α=40 + LLaVA-Next L20 α=10 / L25
  α=15 + **Idefics2 L25 α=20** ("The tip of the arrow will hit the
  center of the circle."). LLaVA-1.5 L25 0/10 (encoder bottleneck).
  **§4.6 perceiver-resampler 가설 refined**: Idefics2 LM has the
  signal (M4 0.995) + forward-hook works (M5a 10/10) + pixel-space
  inverse blocked (§4.6 0/9) → perceiver-resampler removes
  **pixel-space gradient routability**, not LM-side information.
  Causal localization (paper 기여 2) extended Qwen-only → 3-model
- **M5b SAE intervention cross-model round 2 (2026-04-28 evening)**:
  per-model SAE retrain at the **actually-consumed** vision-encoder
  layer (LLaVA `vision_feature_layer=-2` → L22; Idefics2
  `last_hidden_state` → L26; InternVL3 `-1` → L23 already correct).
  **3 of 5 models break PMR cleanly**: Qwen k=20 (0.4 % of features),
  Idefics2 k=160 (3.5 %), InternVL3 k=160 (3.9 %). **LLaVA-1.5 +
  LLaVA-Next NULL** at any k ≤ 160 — encoder-side SAE features absent
  or too distributed in CLIP cluster. Random controls all 1.0
  (specificity confirmed). Effect concentration tracks M3 vision
  probe AUC ladder — second downstream signature of H-encoder-
  saturation. **LLaVA-Next M5a positive (10/10 LM-side flip) +
  M5b NULL** → physics-mode commitment routes through LM, not
  encoder, in the LLaVA family. Insight:
  `docs/insights/m5b_sae_intervention_cross_model.md`.
  cross-model.

Package at `src/physical_mode/`, entry scripts at `scripts/0{1..6}_*.py`
+ `scripts/sec4_6_*_layer_sweep_unified.py` + `scripts/sae_*.py`,
configs at `configs/` (incl. **m2_qwen_32b**, m8a/c/d/e variants,
cross_model_{llava,llava_next,idefics2,internvl3}{,_capture,_label_free}),
tests at `tests/`. Read `docs/architecture.md`
for the implementation contract and `references/project.md` for the
original scientific motivation.

Next priorities (per `references/roadmap.md`):
- **M7** — paper draft (EMNLP long primary) + Prolific human baseline (20 raters × 50 stim).
- **§4.6 follow-ups** — InternVL3 alternative-baseline exploration (no in-distribution abstract-baseline cell exists under the §4.6 "circle" prompt). ~~Idefics2 deeper-layer test~~ ✅ done 2026-04-28 — perceiver-resampler is leading remaining candidate (not isolated; controlled projector-swap test out of scope).
- **§4.8 follow-up** — Qwen 72B on M2 to extend the scaling curve (predicted to land near 32B).
- **§4.4** — Michotte 2-frame causality (needs 2-image prompt support).
- **M5b follow-ups** — post-projection SAE. ~~Cross-model SAE intervention runs~~ ✅ done 2026-04-28 evening (3 of 5 models break PMR cleanly; LLaVA family NULL — encoder-vs-LM mechanism dissociation locked).

**Retired from paper scope (2026-04-28)**:
- **ST5 explicit prompt-steering** (Gavrikov-style "treat as abstract / treat as physical") — retired with reframe: prompt-variation axis is covered by §4.3 KO/JA labels + M4b label-free + M4c FC label-free + open vs forced-choice. Reviewers can be referred to those sections rather than executing the explicit Gavrikov instantiation.

See `docs/next_steps.md` for code-level plug-in points.

## Repository layout

```
references/    project.md, roadmap.md
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
  `docs/experiments/m{N}_<slug>.md`. Major milestone completions also
  get a `docs/insights/m{N}_<slug>.md` deep dive — see
  `references/roadmap.md` §5 for the convention.
