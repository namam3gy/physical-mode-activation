# M5a Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden M5a's causal steering claim by running two controlled follow-ups — negative α bidirectionality + label=ball regime swap — and documenting the results.

**Architecture:** Reuse the existing M5a scripts/vectors. Add a small `extract_first_letter` helper in `src/physical_mode/metrics/`. Add a `--output-subdir` CLI flag to `scripts/06_vti_steering.py` so two new runs can write alongside the M5a baseline without overwriting it. Post-run: write experiment log, insight doc, reproduction notebook, and update the roadmap.

**Tech Stack:** Python 3.11, `uv`, PyTorch/Transformers (already loaded via `PhysModeVLM`), pandas, pytest, Jupyter.

---

## Context for the executor

- **Spec**: `docs/superpowers/specs/2026-04-24-m5a-extensions-design.md` (read first).
- **M5a reference**: `docs/insights/m5_vti_steering.md` — numbers and response examples.
- **Project rules** (from `CLAUDE.md`):
  1. Always run Python through `uv` (`uv run python ...`, `uv run python -m pytest`).
  2. Bilingual required for `docs/insights/*.md` and `references/{project,roadmap}.md` — write English first, then Korean `_ko.md`.
  3. After completing hypothesis validation, write a reproduction notebook at `notebooks/<slug>.ipynb`.
  4. Work in English; speak to the user in Korean.
- **Inputs on disk**:
  - Run dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/`
  - Stimulus dir: `inputs/mvp_full_20260424-093926_e9d79da3/` (note different timestamp from run dir)
  - Steering vectors: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz` (already computed in M5a)

## File layout

| File | Action | Responsibility |
|---|---|---|
| `src/physical_mode/metrics/first_letter.py` | Create | Pure helper that returns the first A/B/C/D token (or `"other"`) from a response string. |
| `tests/test_first_letter.py` | Create | Pytest cases covering the canonical response shapes observed in M5a. |
| `scripts/06_vti_steering.py` | Modify | Add `--output-subdir` CLI flag; compute + print + save first-letter pivot; include `output_subdir` in `run_meta.json`. |
| `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` | Generate | Experiment 1 output artifacts. (gitignored) |
| `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` | Generate | Experiment 2 output artifacts. (gitignored) |
| `docs/experiments/m5a_ext_neg_alpha_and_label.md` (+ `_ko.md`) | Create | Per-run log: tables, raw samples, run_meta. |
| `docs/insights/m5a_ext_bidirection_and_label.md` (+ `_ko.md`) | Create | Interpretation + hypothesis-scorecard deltas. |
| `notebooks/m5a_ext_steering.ipynb` | Create | Cell-by-cell reproduction — loads both parquets, prints tables, renders one bar chart. |
| `references/roadmap.md` (+ `_ko.md`) | Modify | §1.3 scorecard, §2 status row, §3 new subsection, §6 change log. |

## Commit strategy

Grouped for coherence; ~6 commits total. `outputs/` is gitignored, so runs don't create commits — only code and doc changes do.

1. `test(metrics): add first-letter extractor with canonical M5a cases` — task 1
2. `feat(steering): --output-subdir flag + first-letter summary` — task 2
3. (no commit — experiments run, outputs gitignored) — tasks 3, 4
4. `docs(experiments): M5a extensions run log (en + ko)` — tasks 5, 6
5. `docs(insights): M5a extensions interpretation (en + ko)` — tasks 7, 8
6. `docs(notebooks): M5a extensions reproduction notebook` — task 9
7. `docs(roadmap): M5a extensions — scorecard + status + changelog (en + ko)` — task 10

---

## Task 1: First-letter extraction helper (TDD)

**Files:**
- Create: `src/physical_mode/metrics/first_letter.py`
- Create: `tests/test_first_letter.py`

### Steps

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_first_letter.py`:

```python
"""Tests for first-letter response parsing — canonical M5a response shapes."""

from __future__ import annotations

import pytest

from physical_mode.metrics.first_letter import extract_first_letter


@pytest.mark.parametrize(
    "text, expected",
    [
        # M5a insight doc §3.3 — baseline and L20 α=40
        ("D — This is an abstract shape and as such, it does not have "
         "physical properties that would allow it to fall, move, or change "
         "in any way.", "D"),
        # M5a insight doc §3.3 — L10 α=40 intervention
        ("B) It stays still. — Justification: The circle in the image "
         "appears to be floating or suspended in space without any external "
         "force acting upon it.", "B"),
        # Plain letter + period
        ("A. The ball falls.", "A"),
        # Letter + colon
        ("C: rolls to the right.", "C"),
        # Leading whitespace tolerated
        ("   B) stays still", "B"),
        # Letter + newline
        ("A\nThe ball falls due to gravity.", "A"),
    ],
)
def test_canonical_forms(text: str, expected: str) -> None:
    assert extract_first_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "The ball falls because of gravity.",  # no leading letter
        "Option A: falls",  # leading "Option" word — strict form requires letter-first
        "ABCD — all options",  # ambiguous — must be followed by boundary
    ],
)
def test_non_matching_returns_other(text) -> None:
    assert extract_first_letter(text) == "other"


def test_lowercase_letter_matches_with_uppercase_result() -> None:
    assert extract_first_letter("b) stays still") == "B"
```

- [ ] **Step 1.2: Run tests to confirm they fail**

Run: `uv run python -m pytest tests/test_first_letter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'physical_mode.metrics.first_letter'`

- [ ] **Step 1.3: Write the minimal implementation**

Create `src/physical_mode/metrics/first_letter.py`:

```python
"""Extract the first A/B/C/D letter from a forced-choice response string.

M5a §3.4 noted that PMR is a noisy signal for steering interventions
(option-text quoting inflates hits). The first-letter distribution is the
cleaner causal signal. Use this helper from the steering-sweep script
and from analysis notebooks.
"""

from __future__ import annotations

import re

_FIRST_LETTER_RE = re.compile(r"^\s*([ABCDabcd])(?=[\s\)\.\:\-—,]|$)")


def extract_first_letter(text: str | None) -> str:
    """Return "A"/"B"/"C"/"D" for the leading choice token, else "other"."""
    if not text:
        return "other"
    m = _FIRST_LETTER_RE.match(text)
    if not m:
        return "other"
    return m.group(1).upper()
```

- [ ] **Step 1.4: Run tests to confirm they pass**

Run: `uv run python -m pytest tests/test_first_letter.py -v`
Expected: all cases PASS.

- [ ] **Step 1.5: Run the full test suite**

Run: `uv run python -m pytest`
Expected: all pre-existing tests still pass, plus the new ones.

- [ ] **Step 1.6: Commit**

```bash
git add src/physical_mode/metrics/first_letter.py tests/test_first_letter.py
git commit -m "$(cat <<'EOF'
test(metrics): add first-letter extractor with canonical M5a cases

Pure helper that returns A/B/C/D (or "other") from forced-choice
response strings. M5a §3.4 established that first-letter distribution
is the cleaner causal signal vs PMR for steering interventions.
Regex anchored to start-of-string + boundary char to avoid matching
mid-word capital letters.
EOF
)"
```

---

## Task 2: Extend `scripts/06_vti_steering.py` with `--output-subdir` and first-letter summary

**Files:**
- Modify: `scripts/06_vti_steering.py`

### Steps

- [ ] **Step 2.1: Add the CLI flag**

In `scripts/06_vti_steering.py`, inside `main()`'s argparse block (currently lines 62-77), add after the `--model-id` line:

```python
    p.add_argument("--output-subdir", default=None,
                   help="subdir under steering_experiments/ for this run "
                   "(keeps M5a's original outputs intact when set)")
```

- [ ] **Step 2.2: Import the first-letter helper**

Add to the imports at the top of `scripts/06_vti_steering.py` (after the existing `physical_mode` imports):

```python
from physical_mode.metrics.first_letter import extract_first_letter
```

- [ ] **Step 2.3: Use `--output-subdir` when resolving the output path**

Replace the current output-dir block (currently around lines 153-156):

```python
    outdir = args.run_dir / "steering_experiments"
    outdir.mkdir(exist_ok=True)
    df.to_parquet(outdir / "intervention_predictions.parquet", index=False)
    df.to_csv(outdir / "intervention_predictions.csv", index=False)
```

with:

```python
    outdir = args.run_dir / "steering_experiments"
    if args.output_subdir:
        outdir = outdir / args.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "intervention_predictions.parquet", index=False)
    df.to_csv(outdir / "intervention_predictions.csv", index=False)
```

- [ ] **Step 2.4: Add first-letter computation + summary**

Immediately after the line `df = score_rows(df)` (currently around line 151), insert:

```python
    df["first_letter"] = df["raw_text"].apply(extract_first_letter)
```

Then, after the PMR pivot block (currently lines 161-164), add a first-letter summary block:

```python
    print()
    print("=" * 72)
    print("First-letter distribution by (layer, alpha)")
    print("=" * 72)
    fl_pivot = (
        df.groupby(["layer", "alpha", "first_letter"])
          .size()
          .unstack("first_letter", fill_value=0)
    )
    print(fl_pivot.to_string())
    fl_pivot.to_csv(outdir / "first_letter_by_layer_alpha.csv")
```

- [ ] **Step 2.5: Include `output_subdir` in `run_meta.json`**

Edit the `run_meta.json` write (currently around lines 181-192) to include the new field. Change:

```python
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "stimulus_dir": str(args.stimulus_dir),
        "test_subset": args.test_subset,
        "label": args.label,
        "prompt_variant": args.prompt_variant,
        "layers": layers,
        "alphas": alphas,
        "temperature": args.temperature,
        "n_stimuli": len(sub),
        "n_total_inferences": len(df),
    }, indent=2))
```

to include `output_subdir`:

```python
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "stimulus_dir": str(args.stimulus_dir),
        "test_subset": args.test_subset,
        "label": args.label,
        "prompt_variant": args.prompt_variant,
        "layers": layers,
        "alphas": alphas,
        "temperature": args.temperature,
        "n_stimuli": len(sub),
        "n_total_inferences": len(df),
        "output_subdir": args.output_subdir,
    }, indent=2))
```

- [ ] **Step 2.6: Smoke-test CLI parsing**

Run:
```bash
uv run python scripts/06_vti_steering.py --help 2>&1 | grep -E "output-subdir|alphas|label"
```
Expected: lines for `--output-subdir`, `--alphas`, `--label` all print with their help strings.

- [ ] **Step 2.7: Commit**

```bash
git add scripts/06_vti_steering.py
git commit -m "$(cat <<'EOF'
feat(steering): --output-subdir flag + first-letter summary

Adds --output-subdir so M5a extension runs (neg α, label=ball) can
write alongside the original M5a baseline output without clobbering
it. Also computes and saves first-letter_by_layer_alpha.csv, which
is the cleaner causal signal per M5a §3.4 (PMR over-counts when the
option text itself is quoted).
EOF
)"
```

---

## Task 3: Run Experiment 1 — negative α sweep on physics-mode baseline

No code changes; just execute + verify. No commit (outputs are gitignored).

### Steps

- [ ] **Step 3.1: Execute the experiment**

Run:
```bash
uv run python scripts/06_vti_steering.py \
    --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset textured/ground/both \
    --label circle \
    --layers 10 \
    --alphas 0,-5,-10,-20,-40 \
    --output-subdir neg_alpha_textured_ground_both
```

Expected wall-clock: ≈1.5–3 min (model load ≈ 30–60 s, 50 inferences).
Expected output: tqdm progress bar counts to 50 (10 stim × 1 layer × 5 α).

- [ ] **Step 3.2: Verify output files exist**

Run:
```bash
ls outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/
```
Expected: `intervention_predictions.parquet`, `intervention_predictions.csv`, `pmr_by_layer_alpha.csv`, `first_letter_by_layer_alpha.csv`, `run_meta.json`.

- [ ] **Step 3.3: Capture the summary tables**

Read the stdout from Step 3.1 (or re-open the CSVs) and record:
- The PMR pivot at (layer=10, α ∈ {0, -5, -10, -20, -40}).
- The first-letter pivot at the same cells.
- One representative `raw_text` per α value.

These numbers feed the experiment log in Task 5. Save them to a scratch note (not committed).

---

## Task 4: Run Experiment 2 — label=ball on abstract baseline

No code changes; just execute + verify. No commit.

### Steps

- [ ] **Step 4.1: Execute the experiment**

Run:
```bash
uv run python scripts/06_vti_steering.py \
    --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none \
    --label ball \
    --layers 10 \
    --alphas 0,5,10,20,40 \
    --output-subdir ball_line_blank_none
```

Expected wall-clock: ≈1.5–3 min.
Expected output: tqdm to 50.

- [ ] **Step 4.2: Verify output files exist**

Run:
```bash
ls outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/
```
Expected: same five files as Step 3.2.

- [ ] **Step 4.3: Capture the summary tables**

Same as Step 3.3 — record PMR pivot, first-letter pivot, and one representative `raw_text` per α. Save to scratch.

- [ ] **Step 4.4: Confirm M5a baseline is untouched**

Run:
```bash
ls -la outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/intervention_predictions.parquet
```
Expected: the file exists at the old path, mtime unchanged from April 24.

---

## Task 5: Write English experiment log

**Files:**
- Create: `docs/experiments/m5a_ext_neg_alpha_and_label.md`

### Steps

- [ ] **Step 5.1: Write the experiment log**

Create `docs/experiments/m5a_ext_neg_alpha_and_label.md` with the following structure. Fill the tables and response samples from the scratch notes in Tasks 3 and 4.

```markdown
# M5a Extensions — Run Log

Two follow-up experiments to M5a, executed 2026-04-24 on the same
`outputs/mvp_full_20260424-094103_8ae1fa3d` run directory.

Design spec: `docs/superpowers/specs/2026-04-24-m5a-extensions-design.md`.
Parent milestone: `docs/experiments/m5_vti_steering.md` (if present) / `docs/insights/m5_vti_steering.md`.

## Experiment 1 — Negative α on physics-mode baseline

**Question**: does injecting `-α · v_L10` at a physics-mode baseline flip it toward "abstract"?

**Setup**:
- Test subset: `textured/ground/both`, 10 seeds, event=`fall`.
- Label: `circle`.
- Steering layer: 10.
- α: `0, -5, -10, -20, -40`.
- Prompt variant: `forced_choice`, T=0.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/`.

### PMR by (layer, α)

<table from Task 3 stdout / pmr_by_layer_alpha.csv>

### First-letter distribution by (layer, α)

<table from first_letter_by_layer_alpha.csv>

### Representative responses

<one raw_text sample per α from the parquet>

## Experiment 2 — Label=ball on abstract baseline

**Question**: does swapping label `circle` → `ball` shift the L10 α=40 flip target from B ("stays still") to A ("falls")?

**Setup**:
- Test subset: `line/blank/none`, 10 seeds, event=`fall`.
- Label: `ball`.
- Steering layer: 10.
- α: `0, 5, 10, 20, 40`.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/`.

### PMR by (layer, α)

<table>

### First-letter distribution by (layer, α)

<table>

### Representative responses

<samples>

## Cross-check vs M5a

- M5a `line/blank/none × circle × L10 α=40`: **10/10 → B** (from `docs/insights/m5_vti_steering.md` §3.2).
- M5a baseline α=0 untouched (verified in Task 4.4).

## Raw artifacts

- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` — Exp 1 parquet, CSVs, run_meta.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` — Exp 2.
```

- [ ] **Step 5.2: Verify markdown renders**

Run: `uv run python -c "import pathlib; t = pathlib.Path('docs/experiments/m5a_ext_neg_alpha_and_label.md').read_text(); print(t[:200])"`
Expected: the first 200 chars print correctly.

- [ ] **Step 5.3: Write the Korean translation**

Create `docs/experiments/m5a_ext_neg_alpha_and_label_ko.md` — mirror the English structure section-for-section. All technical terms stay English inside Korean sentences (H-regime, first-letter, steering vector, etc. are fine as-is). Numbers/tables are identical to the English version.

- [ ] **Step 5.4: Commit**

```bash
git add docs/experiments/m5a_ext_neg_alpha_and_label.md \
        docs/experiments/m5a_ext_neg_alpha_and_label_ko.md
git commit -m "$(cat <<'EOF'
docs(experiments): M5a extensions run log (en + ko)

Logs both neg α and label=ball run outputs: PMR pivots, first-letter
distributions, representative raw_text samples, artifact paths. Both
runs targeted outputs/mvp_full_20260424-094103_8ae1fa3d.
EOF
)"
```

---

## Task 6: (merged into Task 5's commit — see Step 5.3)

*(Kept numbered for plan-completion-count clarity. Continue with Task 7.)*

---

## Task 7: Write English insight doc

**Files:**
- Create: `docs/insights/m5a_ext_bidirection_and_label.md`

### Steps

- [ ] **Step 7.1: Write the insight doc**

Create `docs/insights/m5a_ext_bidirection_and_label.md`. Use the following outline. The **interpretation** sections depend on the actual results from Tasks 3–4 — write them based on what the numbers show, selecting one of the three outcome sketches per section.

```markdown
# M5a Extensions — Bidirectionality & Label Interaction

Follow-up to `m5_vti_steering.md`. Addresses two §7 limitations:
negative-α bidirectionality and label × steering-direction interaction.

Raw numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.
Implementation: `scripts/06_vti_steering.py` (extended), `src/physical_mode/metrics/first_letter.py`.

## 1. One-line summary

<one sentence stating the headline result — e.g. "The L10 direction is
(bidirectional | one-way); label swap (does | does not) shift the flip
target.">

## 2. Bidirectionality test (Exp 1)

### 2.1 Setup (link to run log §Exp 1).

### 2.2 Result — first-letter distribution

<table from experiment doc>

### 2.3 Interpretation

Pick ONE of these, grounded in the actual first-letter numbers:

- **(confirmed)**: if α=-40 shifts the distribution toward D, the direction
  is bidirectional — +v drives "abstract → physical", -v drives "physical
  → abstract". Write: "negative-α at L10 does flip the sign; M5a's
  direction is a bidirectional object-ness axis in the residual stream."
- **(refuted)**: if α=-40 stays at the baseline, the direction is a one-way
  gate — positive injection turns "physical-mode" on but removal of
  "physical-mode" does not project through -v. Write: "the direction
  activates but does not suppress; M5a's direction is a one-way
  activation, not a bidirectional concept axis."
- **(mixed)**: if partial shift, describe the exact fraction and note
  that the result supports a weaker claim.

## 3. Label × steering (Exp 2)

### 3.1 Setup (link to run log §Exp 2).

### 3.2 Result — first-letter distribution

<table>

### 3.3 Interpretation

- **(H7 confirmed)**: if α=40 flips to majority A (≥ 5/10), the label
  composes with the steering direction: the same "object-ness" vector
  at L10 routes to the "falls" regime when the token prior selects
  that regime.
- **(H7 refuted)**: if α=40 still flips to B (≥ 5/10), the L10 direction
  dominates the regime choice regardless of label — H7 label-selects-
  regime does not hold under intervention.
- **(mixed)**: describe.

## 4. Hypothesis scorecard update

| H | Pre-M5a-ext | Post-M5a-ext | Change |
|---|---|---|---|
| H-boomerang | causal support | <same / strengthened> | <why> |
| H-locus | supported (early-mid) | <same> | — |
| H-regime | candidate | <confirmed / refuted / mixed> | <Exp 2 result> |
| (new) H-direction-bidirectional | — | <candidate / confirmed / refuted> | <Exp 1 result> |

## 5. Paper implications

<1-2 paragraphs: how this affects Figure 6 from M5a, whether it enables a
new figure, whether M5b/M6 priority shifts.>

## 6. Limitations still open

- α=40 is still magic (finer sweep deferred).
- Broader stimulus subset not tested (deferred).
- L10 only; other layers at extreme α not retested.
- Does not touch SAE / patching / cross-model — those are M5b/M6.
```

- [ ] **Step 7.2: Write the Korean translation**

Create `docs/insights/m5a_ext_bidirection_and_label_ko.md` — mirror structure. English technical terms stay English inside Korean sentences.

- [ ] **Step 7.3: Commit**

```bash
git add docs/insights/m5a_ext_bidirection_and_label.md \
        docs/insights/m5a_ext_bidirection_and_label_ko.md
git commit -m "$(cat <<'EOF'
docs(insights): M5a extensions interpretation (en + ko)

Interprets the two M5a-ext experiments: bidirectionality of the L10
direction and the label × steering interaction. Updates H-regime and
adds H-direction-bidirectional. Bilingual per project rule 6.
EOF
)"
```

---

## Task 8: (merged into Task 7's commit — see Step 7.2)

*(Kept numbered for plan-completion-count clarity.)*

---

## Task 9: Reproduction notebook

**Files:**
- Create: `notebooks/m5a_ext_steering.ipynb`

### Steps

- [ ] **Step 9.1: Create the notebook**

Create `notebooks/m5a_ext_steering.ipynb` with the following cells. Follow the style of `notebooks/m5_vti_steering.ipynb` (markdown intro cells, code cells numbered for clarity, outputs pre-populated after running).

**Cell 1 (markdown)** — title + purpose:
```markdown
# M5a extensions — reproduction

Loads the two M5a extension experiments (neg α on `textured/ground/both` and label=ball on `line/blank/none`) and prints the key tables. Depends on M5a outputs already being present at `outputs/mvp_full_20260424-094103_8ae1fa3d/`.

- Design spec: `docs/superpowers/specs/2026-04-24-m5a-extensions-design.md`
- Run log: `docs/experiments/m5a_ext_neg_alpha_and_label.md`
- Insight: `docs/insights/m5a_ext_bidirection_and_label.md`
```

**Cell 2 (code)** — imports + paths:
```python
from pathlib import Path
import pandas as pd

from physical_mode.metrics.first_letter import extract_first_letter

RUN = Path("outputs/mvp_full_20260424-094103_8ae1fa3d")
EXP1 = RUN / "steering_experiments" / "neg_alpha_textured_ground_both"
EXP2 = RUN / "steering_experiments" / "ball_line_blank_none"
```

**Cell 3 (markdown)** — "## Experiment 1 — negative α"

**Cell 4 (code)** — load + first-letter pivot for Exp 1:
```python
df1 = pd.read_parquet(EXP1 / "intervention_predictions.parquet")
print("n =", len(df1))
fl1 = (
    df1.groupby(["layer", "alpha", "first_letter"])
       .size()
       .unstack("first_letter", fill_value=0)
)
fl1
```

**Cell 5 (code)** — one representative response per α:
```python
for a in sorted(df1["alpha"].unique()):
    r = df1[df1["alpha"] == a].iloc[0]
    text = r["raw_text"].replace("\n", " ")[:180]
    print(f"α={a:>6}: first={r['first_letter']}  :: {text}")
```

**Cell 6 (markdown)** — "## Experiment 2 — label=ball"

**Cell 7 (code)** — same pattern for Exp 2:
```python
df2 = pd.read_parquet(EXP2 / "intervention_predictions.parquet")
print("n =", len(df2))
fl2 = (
    df2.groupby(["layer", "alpha", "first_letter"])
       .size()
       .unstack("first_letter", fill_value=0)
)
fl2
```

**Cell 8 (code)** — sample responses for Exp 2:
```python
for a in sorted(df2["alpha"].unique()):
    r = df2[df2["alpha"] == a].iloc[0]
    text = r["raw_text"].replace("\n", " ")[:180]
    print(f"α={a:>6}: first={r['first_letter']}  :: {text}")
```

**Cell 9 (markdown)** — "## Compare with M5a baseline"

**Cell 10 (code)** — side-by-side:
```python
base = pd.read_parquet(RUN / "steering_experiments" / "intervention_predictions.parquet")
base10 = base[(base["layer"] == 10)].copy()
base10["first_letter"] = base10["raw_text"].apply(extract_first_letter)
m5a_pivot = (
    base10.groupby(["alpha", "first_letter"])
          .size()
          .unstack("first_letter", fill_value=0)
)
print("M5a baseline (circle × line/blank/none × L10):")
m5a_pivot
```

- [ ] **Step 9.2: Execute the notebook cell-by-cell**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/m5a_ext_steering.ipynb \
    --output notebooks/m5a_ext_steering.ipynb
```
Expected: no cell errors; cell outputs written in-place.

- [ ] **Step 9.3: Commit**

```bash
git add notebooks/m5a_ext_steering.ipynb
git commit -m "$(cat <<'EOF'
docs(notebooks): M5a extensions reproduction notebook

Cell-by-cell reproduction of the two M5a-ext experiments. Loads
the parquets, prints first-letter pivots and sample raw_text per α,
and shows a side-by-side with the M5a baseline.
EOF
)"
```

---

## Task 10: Update roadmap — scorecard, status, changelog

**Files:**
- Modify: `references/roadmap.md`
- Modify: `references/roadmap_ko.md`

### Steps

- [ ] **Step 10.1: Update `references/roadmap.md` §1.3 — add/revise H-regime row and add H-direction-bidirectional row (if confirmed)**

In §1.3 "Hypothesis scorecard", locate the H-regime row and update its Status and Evidence columns based on Exp 2's result. Examples:

- If H7 confirmed: `supported → causally supported (M5a-ext Exp 2)` + new evidence sentence.
- If refuted: `candidate → refuted (M5a-ext Exp 2)` + sentence.
- If mixed: `candidate → mixed (M5a-ext Exp 2)` + sentence.

After the H-regime row, add (if Exp 1 supports it):
```markdown
| **H-direction-bidirectional** (M5a-ext-derived) | The L10 object-ness direction acts as a bidirectional axis — `+α·v` flips abstract → physical, `-α·v` flips physical → abstract. | <candidate / supported / refuted> | M5a-ext Exp 1: L10 α=-40 on `textured/ground/both` × circle → <fraction> flip to D. |
```

- [ ] **Step 10.2: Update §2 — add "M5a-ext" status row**

In the milestone overview table (§2), add immediately after the M5a row:

```markdown
| M5a-ext | **ST4 Phase 1+2 extensions** | Neg α bidirectionality + label=ball regime swap | ✅ | 2026-04-24 |
```

- [ ] **Step 10.3: Update §3 — add M5a-ext subsection**

After the existing `### M5a — ST4 Phase 1+2 VTI steering ✅` section, insert:

```markdown
### M5a-ext — Bidirectionality + label interaction ✅ (2026-04-24)

Run: two invocations of `scripts/06_vti_steering.py` with `--output-subdir`.

Output:
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/`
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/`

Deep dive: `docs/insights/m5a_ext_bidirection_and_label.md`.
Run log: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.

**Key results** (fill from insight doc §2 and §3):
- Exp 1: <one-sentence bidirectionality verdict>.
- Exp 2: <one-sentence label × steering verdict>.

**Hypothesis updates**: <list scorecard deltas>.
```

- [ ] **Step 10.4: Update §6 — change log**

Append a new row to the change log table:

```markdown
| 2026-04-24 | M5a extensions: neg α (Exp 1) + label=ball (Exp 2) runs. <Exp 1 one-liner>. <Exp 2 one-liner>. Scorecard: H-regime <status>, H-direction-bidirectional <status>. | (this commit) |
```

- [ ] **Step 10.5: Mirror changes into `references/roadmap_ko.md`**

Translate each of Steps 10.1–10.4's changes into the Korean file. Section numbering is identical.

- [ ] **Step 10.6: Verify both roadmaps parse as markdown**

Run:
```bash
uv run python -c "
import pathlib
for p in ['references/roadmap.md', 'references/roadmap_ko.md']:
    t = pathlib.Path(p).read_text()
    assert '| M5a-ext ' in t, f'{p}: missing M5a-ext row'
    assert '### M5a-ext' in t, f'{p}: missing M5a-ext subsection'
    print(f'{p}: OK ({len(t)} chars)')
"
```
Expected: both files OK.

- [ ] **Step 10.7: Commit**

```bash
git add references/roadmap.md references/roadmap_ko.md
git commit -m "$(cat <<'EOF'
docs(roadmap): M5a extensions — scorecard + status + changelog (en + ko)

Adds M5a-ext row under §2 milestone table, updates §1.3 scorecard
(H-regime, H-direction-bidirectional), adds §3 M5a-ext subsection,
appends §6 change log. Bilingual per project rule 6.
EOF
)"
```

---

## Self-review checklist (for the executor)

Before declaring the plan complete:

- [ ] `uv run python -m pytest` all-green (including new first-letter tests).
- [ ] Two new output subdirs exist and contain `intervention_predictions.parquet`, `first_letter_by_layer_alpha.csv`, `run_meta.json`.
- [ ] M5a's original `steering_experiments/intervention_predictions.parquet` mtime unchanged.
- [ ] `docs/experiments/m5a_ext_*` + `docs/insights/m5a_ext_*` both exist in English + `_ko.md`.
- [ ] `notebooks/m5a_ext_steering.ipynb` executed cell-by-cell without errors.
- [ ] `references/roadmap.md` + `_ko.md` contain the M5a-ext row, subsection, and change log entry.
- [ ] `git log --oneline -8` shows the expected commit sequence (test → feat → docs×3 → docs×2).
- [ ] `git status` is clean.
