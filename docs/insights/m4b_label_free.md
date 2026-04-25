# M4b — Label-Free Prompt as H2 Null-Hypothesis Test

Executes ROADMAP §4.9 ("label-free prompt as H2 null-hypothesis test"), with
a byproduct of revisiting M4's switching-layer degeneracy.

Raw numbers: `docs/experiments/m4b_label_free.md`.
Code: `configs/label_free.py`, `src/physical_mode/inference/prompts.py`
(`open_no_label` variant), `src/physical_mode/probing/lm.py`
(generic `pmr_source`).

## 1. One-line summary

H2 as originally stated was a **misattribution**: the +15 pp "ball vs
circle" gap observed in M2 is not a ball-enhancement effect but a **circle
suppression** effect. Paired comparison against a label-free baseline shows
`ball ≈ no-label` (+0.6 pp) and `circle − no-label = −6.5 pp`. Without a
label prompt, Qwen2.5-VL's visual system already defaults to physics-mode
for M2's stimuli; the "ball" label is redundant, and the "circle" label is
the only one that actively overrides that default.

## 2. Pairing design

M2 ran `(image × label × open)` and `(image × label × forced_choice)` over
{circle, ball, planet}. The label-free run re-exposes the identical images
(M2 stimulus dir) under a prompt that contains no label token:

> "What do you see in the image? What might happen next? Answer in one short sentence."

Same T=0.7, same 10 seeds/cell → the paired comparison on 480 matched
`(obj, bg, cue, seed)` tuples is stochastic-to-stochastic on the same
seeds, so the PMR deltas are noise-filtered estimates of the per-label
contribution at a fixed image.

The design intentionally excludes forced-choice for this round: the FC
options A-D contain "It" with no antecedent, which requires a different
wording to render without a label. Deferred.

## 3. Behavioral finding — the language-prior attribution flip

### 3.1 Paired deltas

| M2 label | `PMR(label) − PMR(_nolabel)` (mean over 480 pairs) |
|---|---|
| ball   | **+0.006** |
| planet | **+0.006** |
| circle | **−0.065** |

- `ball` and `planet` are statistically indistinguishable from no-label at
  the overall level. The model does not *need* the `ball` token to enter
  physics-mode on M2's stimuli — visual content alone is sufficient.
- `circle` reduces PMR by ~6.5 pp on average. The "circle" token is an
  **override token**: it semantically forces an abstract-geometry reading
  even when the image would otherwise support physics.

### 3.2 Cell-level structure

Two axes modulate the circle-suppression:

- **Object abstraction**: `circle` suppresses `line` by 9.2 pp vs `filled`
  by 4.2 pp. The more abstract the image, the stronger the label override.
  This is the symmetric dual of H4 (higher abstraction → larger language
  prior gap): the language prior's *direction* reverses under the label
  change — physical prior (ball) aligns with image, abstract prior
  (circle) opposes image, and the more abstract the image the more room
  the circle override has to pull PMR down.
- **Cue strength**: `motion_arrow` makes the circle-label suppression go
  to 0.000. The arrow is a visual signal strong enough to override the
  label override. `none` (no cue) gives the largest suppression (−15 pp),
  because the label is the *only* text-side signal.

### 3.3 The fully abstract cell — `line/blank/none`

Because M2 ran 3 labels and label-free adds a 4th condition, we get a
4-point view of the most ambiguous cell:

| label    | PMR  | hold_still |
|---|---|---|
| _nolabel | 0.40 | 0.20 |
| ball     | 0.40 | 0.60 |
| circle   | 0.10 | 1.00 |
| planet   | 0.70 | 0.30 |

Three distinct label effects, visible only at this cell because visual
ambiguity maximally exposes the language prior:

- `ball` leaves PMR unchanged but swaps the regime — responses no longer
  fall (kinetic) but stay still (static). The "ball" label routes to
  physics-mode but redirects the regime toward static interpretation on an
  abstract image. This is the same pattern M5a-ext Exp 3 observed at the
  VTI-steering side (static regime at moderate baselines).
- `circle` triggers outright abstract-override: PMR 0.10, hold_still 1.00.
- `planet` is the **only label that raises PMR above the no-label
  baseline** — and it does so by +30 pp. The "planet" prior is genuinely
  additive over visual content, consistent with its low GAR (0.32 overall)
  representing orbital physics that the image alone cannot evoke.

### 3.4 H2 rewrite

The revised H2 reads:

> **Labels do not uniformly raise PMR over a visual-only baseline. On
> M2's programmatic stimuli, `ball` is visually redundant (≈ no-label),
> `circle` is an abstract override (−6.5 pp vs no-label, amplified at
> more abstract images), and `planet` mildly adds orbital-physics prior
> visible only on abstract images.**

This preserves the original H2 direction (language matters) but
redistributes the per-label contributions: the M2 "ball > circle" gap is
now `circle < visual-default`, not `ball > visual-default`.

## 4. Mechanistic finding — visual-token captures do not observe the label

Re-running M4 (`scripts/05_lm_probing.py --sources open_no_label`) on the
label-free activations reproduces the same physics-margin table and the
same collapsed switching-layer (L5 for all 480 samples) as M2. A bit-for-bit
activation diff between the two runs at layers {5, 10, 15, 20, 25}
confirms: **the visual-token hidden states are identical** regardless of
prompt. Only `input_ids` and `visual_token_mask` differ in length.

This is a structural consequence of the Qwen2.5-VL chat template: image
tokens precede the question text in the user message, so under causal
attention the question text (including the label) does not propagate back
into visual-token positions. The M4 logit lens and per-layer PMR probe, as
currently specified, capture the image-only side of the information flow
and cannot measure label contribution.

Implications:

- The M4 claim "switching layer is at L5 regardless of condition" is true
  but underdetermined — it reflects a structural feature of the capture
  point, not a convergent behavior across prompts.
- To localize the label effect inside the LM, the probe must capture at
  text positions *after* the label token (e.g., the last question-text
  token or the start-of-answer token). Adding a second capture hook at a
  late-prompt position would give per-prompt hidden states that actually
  differ.
- Conversely, the constancy of the image-side hidden states across prompts
  is independent behavioral evidence for H-boomerang (vision encoder
  encodes physics-vs-abstract from pixels alone; the LM's behavioral
  modulation by label is all downstream of visual tokens).

## 5. Hypothesis scorecard update

| H | Pre-M4b | Post-M4b |
|---|---|---|
| **H2** (language prior raises PMR) | quantified | **revised** — ball ≈ no-label, planet ≈ no-label (except on abstract images), circle = suppressor. Per-label contributions flip: "language prior" is asymmetric, primarily a *negative* effect driven by `circle`. |
| **H-boomerang** | supported + causal | **reinforced** — visual-token hidden states are prompt-independent, so the physics-bias present there from L5 is image-only. |
| **H-locus** | supported (early-mid L10) | **unchanged** — label's behavioral effect localizes to text positions *after* the image tokens (not captured here); consistent with M5a's L10-centric intervention being effective on the image-preceding trajectory. |
| **H4** (abstraction → gap) | supported, extended | **refined** — label override strength (`circle − no-label`) is larger on more abstract images, which is the image-side dual of M4's language-prior-per-abstraction finding. |

## 6. Paper implications

- The paper should lead with the **corrected H2**: replace "ball label
  substantially raises PMR" with "circle label substantially lowers PMR
  below the visual-default baseline". This changes the story from "VLMs
  need language to activate physics" to "VLMs default to physics on
  physical-looking stimuli; only an explicit abstract label (`circle`)
  actively opposes that default."
- The `line/blank/none` 4-label table (§3.3) is the cleanest figure for
  this claim: it localizes each label's contribution orthogonally (regime
  for `ball`, suppression for `circle`, orbit-prior for `planet`).
- Methodologically, the activation-capture diff note (§4) should be
  flagged in the M4 section of the paper — it tempers the original
  switching-layer claim and clarifies what the captured hidden states
  can and cannot measure.

## 7. Limitations still open

- Forced-choice label-free deferred: the FC options ("It falls / stays /
  ...") need a re-written subject to make sense without a label. A
  single-word placeholder like `this` or `the object` would be the next
  design to validate.
- Capture at last-question-token / start-of-answer position would measure
  the label's downstream effect inside the LM. Not done here.
- Single prompt text; a sensitivity audit (e.g., "Describe what is in the
  image..." vs "Is there a ball / object / shape?...") would separate
  wording effects from the core label-vs-no-label contrast.
- Cross-model replication is M6 scope. If LLaVA-1.5 needs `ball` to enter
  physics-mode on the same stimuli, the "visual default" here is
  Qwen-specific.
