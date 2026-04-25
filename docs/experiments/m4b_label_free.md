# M4b Label-Free Prompt — Run Log

Executes ROADMAP §4.9 ("label-free prompt as H2 null-hypothesis test") and
attempts the M4 follow-up of producing a non-degenerate switching-layer metric.

Run date: 2026-04-25.

## Setup

- Config: `configs/label_free.py` (new).
- Stimuli: reused M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/` (480
  stimuli: 4 obj × 3 bg × 4 cue × 1 event × 10 seeds).
- Model / generation: same as M2 — Qwen2.5-VL-7B-Instruct, bf16, T=0.7,
  top_p=0.95, `max_new_tokens=96`.
- Prompt: `open_no_label` variant (new), text is
  `"What do you see in the image? What might happen next? Answer in one short sentence."`.
- Labels: `("_nolabel",)` sentinel, single iteration (no label × variant combinatorics).
- Activation capture: `lm_hidden_{5,10,15,20,25}` under the `open_no_label`
  prompt (matches M2 layer set for paired M4 re-run).
- Output: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions +
  480 activation safetensors (5 layers each). Wall-clock ≈ 13 min.

## Behavioral results

### Overall

| metric | value |
|---|---|
| n | 480 |
| PMR | 0.948 |
| hold_still | 0.127 |
| abstract_reject | 0.002 |
| GAR | 0.728 |

### Paired PMR delta vs M2 open prompt (same (obj, bg, cue, seed))

| label in M2 | mean delta `PMR(label) − PMR(_nolabel)` | n |
|---|---|---|
| ball   | **+0.006** | 480 |
| circle | **−0.065** | 480 |
| planet | **+0.006** | 480 |

- The "ball" and "planet" labels add nothing to PMR vs the no-label baseline.
- The "circle" label actively **suppresses** PMR by ~6.5 pp on average.
- This flips the M2 framing of "ball raises PMR" — the delta is actually
  "circle lowers PMR" (ball ≈ no-label = visual default).

### Per-object-level PMR — open prompt

| object | ball (M2) | circle (M2) | planet (M2) | _nolabel (LF) | ball − _nolabel | circle − _nolabel |
|---|---|---|---|---|---|---|
| line     | 0.950 | 0.850 | — | 0.942 | +0.008 | −0.092 |
| filled   | 0.950 | 0.892 | — | 0.933 | +0.017 | −0.042 |
| shaded   | 0.933 | 0.892 | — | 0.942 | −0.008 | −0.050 |
| textured | 0.983 | 0.900 | — | 0.975 | +0.008 | −0.075 |

The `circle` suppression is strongest on `line` (−9.2 pp) and weakest on
`filled` (−4.2 pp). Pattern: the more abstract the image, the more the
`circle` label can override visual cues.

### Per-cue-level PMR — open prompt

| cue          | ball  | circle | planet | _nolabel | circle − _nolabel |
|---|---|---|---|---|---|
| both         | 0.992 | 0.992 | 0.992 | 0.983 | +0.008 |
| cast_shadow  | 1.000 | 0.850 | 0.958 | 0.967 | −0.117 |
| motion_arrow | 1.000 | 1.000 | 0.992 | 1.000 | 0.000  |
| none         | 0.825 | 0.692 | 0.875 | 0.842 | −0.150 |

`motion_arrow` fully overrides the `circle` label suppression (+0.000).
With no cue at all, circle suppresses by 15 pp — the largest suppression
across all cells.

### Most informative cell — `line/blank/none` (fully abstract image)

| label    | PMR  | hold_still |
|---|---|---|
| _nolabel | 0.40 | 0.20 |
| ball     | 0.40 | 0.60 |
| circle   | 0.10 | 1.00 |
| planet   | 0.70 | 0.30 |

- `_nolabel` and `ball`: same PMR (0.40); `ball` shifts regime toward "stays"
  (hold_still 0.20 → 0.60), without raising the total physics-mode rate.
- `circle`: PMR drops 30 pp; every single response holds still.
- `planet`: PMR rises 30 pp — this is the **only** cell where a label
  genuinely *increases* physics-mode over the visual baseline. Planet label
  brings orbital physics prior that the image alone would not trigger.

## LM probing results (M4 re-run)

Script: `uv run python scripts/05_lm_probing.py --run-dir outputs/label_free_<ts> --sources open_no_label`.

### Physics margin by (layer, object_level)

Label-free:

| layer | filled | line | shaded | textured |
|---|---|---|---|---|
| 5  | 0.08 | 0.09 | 0.12 | 0.15 |
| 10 | 0.29 | 0.27 | 0.33 | 0.38 |
| 15 | 0.38 | 0.35 | 0.44 | 0.49 |
| 20 | 0.89 | 0.87 | 0.97 | 1.05 |
| 25 | 3.94 | 3.76 | 4.29 | 4.35 |

M2 (ball/circle/planet-labeled): **identical to the above** (max diff = 0.0).

### Methodological finding — visual-token captures are prompt-independent

Activations at visual-token positions match bit-for-bit between the M2
run (with label) and the label-free run, confirmed by loading and diffing
`outputs/*/activations/line_blank_none_fall_000.safetensors` for each
layer 5/10/15/20/25 — max absolute diff = 0.0. Only `input_ids` and
`visual_token_mask` (both prompt-length artefacts) differ.

This is a structural property of Qwen2.5-VL's chat template: image tokens
precede the question text in the user message, so causal attention prevents
the label text from affecting visual-token hidden states. The capture at
visual-token positions therefore does not observe any label contribution.

Implication: the M4 switching-layer (all 480 samples at L5) is **not**
evidence that the LM commits to physics-mode from L5 regardless of label —
it is evidence that the vision encoder + early LM layers encode a
physics-biased representation **from image alone**, and the label's
behavioral effect is localized downstream of visual-token positions (last
answer-token position or later), not in the captured region.

### Per-layer PMR probe AUC (open_no_label source)

| layer | AUC mean | AUC std | accuracy | n_pos | n_neg |
|---|---|---|---|---|---|
| 5  | 0.937 | 0.036 | 0.956 | 455 | 25 |
| 10 | 0.955 | 0.016 | 0.956 | 455 | 25 |
| 15 | 0.951 | 0.037 | 0.956 | 455 | 25 |
| 20 | 0.952 | 0.027 | 0.956 | 455 | 25 |
| 25 | 0.958 | 0.027 | 0.956 | 455 | 25 |

Note: the n_pos/n_neg split (455/25) is heavily imbalanced because the
label-free overall PMR = 0.948 — very few y=0 samples. The AUC range
0.94-0.96 here reflects the imbalanced problem, not a mechanistic change
vs M2's 0.94-0.95 on the balanced (312/168) `forced_choice` source.

## Raw artifacts

- `outputs/label_free_20260425-031430_315c5318/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/label_free_20260425-031430_315c5318/activations/*.safetensors` — 480 files, same visual-token hidden states as M2.
- `outputs/label_free_20260425-031430_315c5318/probing_lm/` — layer-sweep + logit-lens outputs.
- `outputs/label_free_20260425-031430_315c5318/summary_*.csv` — factor-level rollups.

## Sample raw responses (T=0.7, `open_no_label` prompt)

- `line/blank/none_000`: "The image shows a simple outline of a circle, and it is unlikely to change as it appears to be a static image."
- `textured/blank/none_000`: "The image shows a bowling ball with holes, and it's likely to roll if thrown or pushed."
- `line/blank/both_000`: "The circle is likely to fall towards the oval shape due to gravity."
- `textured/ground/both_000`: "The ball is about to roll down the incline toward the ground." (typical)

The model identifies the object visually (circle / bowling ball) and applies
physics based on image content + cues, with no prompt-side label required.
