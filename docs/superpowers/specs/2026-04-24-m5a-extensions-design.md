# M5a extensions — design spec

**Date**: 2026-04-24
**Status**: draft → awaiting implementation plan
**Owner**: namam3gy (with Claude Code)
**Roadmap linkage**: extends M5a (ROADMAP §3 "M5a"); not a new milestone — a follow-up strengthening the M5a causal claim before committing to M5b/M6.

## 1. Motivation

M5a established that **injecting `α=40 · v_L10` into Qwen2.5-VL-7B's LM
layer 10 residual stream flips 10/10 responses on `line+blank+none` stimuli
from "D: abstract" to "B: stays still"** (see `docs/insights/m5_vti_steering.md`).
Two sharp questions remain from that work's §7 "Limitations":

- **Is the direction bidirectional?** M5a tested only positive α on an
  abstract baseline. If `-α · v_L10` injected at a physics-mode baseline
  (`textured+ground+both`) flips physics→abstract, the direction is a true
  bidirectional "object-ness" axis. If it doesn't, M5a's finding is a
  one-way gating effect rather than a reversible concept vector.
- **Does the flip target depend on the label?** M5a used `label="circle"`
  and observed D → B flips (stationary physical object), not D → A (falls).
  H7 ("label selects physics regime") predicts that with `label="ball"`,
  the same L10 α=40 injection should shift the flip distribution toward A
  (falls), directly demonstrating the steering-direction × label interaction.

Both experiments are cheap (≈1.5 min each on H200) and reuse all M5a
infrastructure. Their outcomes — combined — substantially harden M5a's
causal claim from "a direction exists" to "a bidirectional, label-composable
direction governs abstract-vs-physical classification".

## 2. Experiments

### Experiment 1 — Negative α sweep (bidirectionality)

| field | value |
|---|---|
| Test subset | `textured/ground/both`, all 10 seeds, 1 event_template (`fall`) |
| Label | `circle` (mirror of M5a's choice — keeps the test symmetric and removes the "ball" prior as a confound) |
| Prompt variant | `forced_choice` |
| Steering layer | **L10 only** (M5a's sole effective layer) |
| α values | `0, -5, -10, -20, -40` |
| Temperature | `0.0` |
| Total inferences | 10 × 1 × 5 = **50** |
| Wall-clock | ≈1.5 min |
| Output dir | `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` |

**Prediction**:
- α=0 baseline: first-letter distribution concentrated on A or B (physics-mode answers).
- α=-40: if direction is bidirectional, distribution shifts toward D (abstract rejection).
- If no shift: direction is one-way — positive α activates "object-ness", negative α does nothing.

### Experiment 2 — Label swap (regime selection under intervention)

| field | value |
|---|---|
| Test subset | `line/blank/none`, all 10 seeds, 1 event_template (same as M5a) |
| Label | **`ball`** (the variable being swapped vs M5a's `circle`) |
| Prompt variant | `forced_choice` |
| Steering layer | **L10 only** |
| α values | `0, 5, 10, 20, 40` |
| Temperature | `0.0` |
| Total inferences | 10 × 1 × 5 = **50** |
| Wall-clock | ≈1.5 min |
| Output dir | `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` |

**Prediction**:
- α=0 baseline: per M2, `ball+line+blank+none` already has PMR≈0.85, so first-letter should favor A or B even without intervention.
- α=40: if the direction is purely "object-ness" and regime is label-driven, the flip target at ball+α=40 should favor **A (falls)** more than M5a's circle+α=40 did (which was 10/10 B).
- A shift from "10/10 B" (M5a) → "≥ 5/10 A" (this) would be strong H7 evidence.

## 3. Implementation

### 3.1 Script changes (single-script extension)

**File**: `scripts/06_vti_steering.py`.

Add:
- CLI flag `--output-subdir <name>`: when set, results go to
  `<run-dir>/steering_experiments/<name>/` instead of the bare
  `steering_experiments/`. Default unset = M5a-compatible behavior (writes
  to `steering_experiments/`).
- First-letter extraction: add a small helper that regex-matches the first
  `A|B|C|D` token in `raw_text` (allowing leading whitespace and optional
  `)` or `:` after) and stores it as a new `first_letter` column.
- Summary: alongside the existing PMR pivot, print a first-letter pivot
  (layer × α → counts in {A, B, C, D, other}). Save as
  `first_letter_by_layer_alpha.csv`.
- `run_meta.json`: add `output_subdir` field.

Everything else unchanged — same forward-hook logic, same PMR scorer, same
steering-vector loader.

**No changes** to `src/physical_mode/probing/steering.py` (steering vectors
are already persisted from M5a at
`probing_steering/steering_vectors.npz`).

### 3.2 Execution

Two script invocations (sequential; model load cost ~30 s, shared if
wrapped, but separate runs for cleanliness):

```bash
# Experiment 1 — negative α on physics-mode baseline
uv run python scripts/06_vti_steering.py \
    --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset textured/ground/both \
    --label circle \
    --layers 10 \
    --alphas 0,-5,-10,-20,-40 \
    --output-subdir neg_alpha_textured_ground_both

# Experiment 2 — label=ball on M5a's abstract baseline
uv run python scripts/06_vti_steering.py \
    --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none \
    --label ball \
    --layers 10 \
    --alphas 0,5,10,20,40 \
    --output-subdir ball_line_blank_none
```

### 3.3 Analysis deliverables

After both runs complete:

- **Experiment log**: `docs/experiments/m5a_ext_neg_alpha_and_label.md`
  (+ `_ko.md` Korean translation per project rule 6). Include: both runs'
  first-letter tables, PMR tables, representative sample responses per
  α value, run_meta JSON dumps.
- **Insight doc**: `docs/insights/m5a_ext_bidirection_and_label.md`
  (+ `_ko.md`). Include: interpretation of bidirectionality result,
  interpretation of label × steering interaction, hypothesis-scorecard
  updates (H-regime strengthened or refuted; H-locus unchanged).
- **Reproduction notebook**: `notebooks/m5a_ext_steering.ipynb` per project
  rule 7. Loads both runs, renders tables + one first-letter bar chart.
- **Roadmap update**: §1.3 hypothesis scorecard (H-regime row), §2 status
  table (add a new "M5a-ext" row beneath M5a), §6 change log entry.

## 4. Scope guards

**In scope**:
- Items (4) and (2) from `docs/insights/m5_vti_steering.md` §7.
- Minimal script extension (CLI flag + first-letter helper).
- Experiment logs, insight doc, notebook, roadmap update.

**Out of scope**:
- Items (1) "broader subset" and (3) "fine α sweep" from §7. Defer.
- SAE feature decomposition (M5b).
- Activation patching (M5b).
- SIP pair construction (M5b).
- Cross-model replication (M6).
- Any stimulus regeneration, config changes, or manifest changes.

## 5. Success criteria

- Both experiments execute end-to-end, producing
  `intervention_predictions.parquet` + `first_letter_by_layer_alpha.csv` +
  `run_meta.json` in each output subdir.
- The M5a-baseline output directory (`steering_experiments/` root) is
  unchanged — M5a's original files remain bit-identical.
- First-letter distribution table is clearly readable for both experiments.
- Insight doc makes one of three clear claims per experiment:
  **(a) prediction confirmed** (direction is bidirectional / label shifts regime),
  **(b) prediction refuted** (one-way / label-independent),
  or **(c) mixed** — with the observed evidence cited.
- Hypothesis scorecard is updated accordingly.
- Notebook runs cell-by-cell from a clean kernel.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| First-letter regex mis-parses responses starting with "The circle..." etc. | Test regex on M5a's known raw_text samples (the insight doc has 3 examples). Fall back to `(A|B|C|D)[\s\)\.:,-]` with anchored variants; log `other` category if unmatched. |
| α=0 baseline on `textured+ground+both` doesn't concentrate on a single letter (muddies bidirectionality read) | Accept as finding — report the per-letter counts as-is; interpret any shift away from the baseline distribution as signal. |
| Run-dir contention if someone else re-runs the script without `--output-subdir` | `--output-subdir` defaults to unset (preserves old behavior), so explicit invocations are required. Add a `mode` note in the run_meta. |
| GPU OOM if model is also in RAM from a prior session | Run one experiment at a time; the existing script already tears down after one invocation. |

## 7. Naming note

This work strengthens M5a, so the file prefix is `m5a_ext_*`, **not**
`m5b_*`. M5b remains reserved for the larger SIP + activation patching +
SAE milestone defined in ROADMAP §3.
