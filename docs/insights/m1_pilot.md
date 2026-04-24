# M1 Pilot Insights

Observations and implications from the Sub-task 1 pilot experiment
(Qwen2.5-VL-7B-Instruct, 480 inferences, H200, 2026-04-24). Raw numbers in
`docs/experiments/m1_pilot.md` and
`outputs/pilot_20260424-072418_2c16efb6/`.

In short — we **partially confirmed the behavioral S-curve**, found that the
**language prior is far more dominant than expected**, and discovered a
**structural problem where some stimulus cues are not read at all by the VLM**.

---

## 1. Experimental setup

### 1.1 Prompt templates

We run both **open-ended** and **forced-choice** variants on every stimulus.
The `{label}` slot carries axis D (fixed to `"ball"` in the pilot). Implementation:
`src/physical_mode/inference/prompts.py`.

**Open-ended**

```text
[system]
You are a careful observer of images. When asked what will happen next,
describe the most likely next state or motion in one short sentence.

[user]
The image shows a {label}. Describe what will happen to the {label} in the
next moment, in one short sentence.
```

**Forced-choice** (4-way MCQ)

```text
[system]
You are a careful observer of images. Answer the multiple-choice question
with a single letter A, B, C, or D followed by a brief justification.

[user]
The image shows a {label}. Which option best describes what will happen next?
A) It falls down.
B) It stays still.
C) It moves sideways.
D) This is an abstract shape — nothing physical happens.
Answer with a single letter (A, B, C, or D), then briefly justify.
```

The D option acts as an "abstract escape route". Pilot results show that
whether or not this option is offered is the largest methodological variable
in PMR (see §3.2).

### 1.2 Stimulus examples

512 × 512 px RGB. Every element is seed-deterministic (implementation
`src/physical_mode/stimuli/`, detailed spec `docs/stimulus_spec.md`).

| Condition | Image | Role |
|---|---|---|
| `line + blank + none` | <img src="../figures/01_line_blank_none.png" width="220"> | most abstract — behavioral baseline |
| `line + ground + none` | <img src="../figures/02_line_ground_none.png" width="220"> | abstract circle + a single ground line — minimal pair for the ground effect |
| `shaded + ground + none` | <img src="../figures/03_shaded_ground_none.png" width="220"> | 3D shading + ground — canonical "ball-falls" stimulus |
| `textured + ground + arrow_shadow` | <img src="../figures/04_textured_ground_arrow_shadow.png" width="220"> | maximum cue — textured ball + ground + arrow + shadow |
| `filled + blank + wind` | <img src="../figures/05_filled_blank_wind.png" width="220"> | wind-cue invisible case — the grey arcs to the left aren't read by the VLM |
| `textured + blank + none` | <img src="../figures/06_textured_blank_none.png" width="220"> | 3D texture + no ground — the object-level effect in isolation |

### 1.3 Actual model response examples

The same 6 stimuli, two prompt variants × Qwen2.5-VL-7B (bf16, temperature=0).
PMR scoring per `src/physical_mode/metrics/pmr.py`.

#### (1) `line + blank + none` — most abstract

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will remain stationary unless acted upon by an external force." | 0 |
| forced | "D — The image is an abstract representation of a circle and does not depict a physical object like a ball that can fall, move, or stay still." | 0 |

→ Neither prompt enters physics-mode. Open-ended is a Newton's-first-law
framing; forced-choice is a meta-cognitive abstract recognition. This cell
is the model's behavioral baseline.

#### (2) `line + ground + none` — **the key pair for the ground effect**

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will likely roll down the incline if it is placed on an inclined surface." | **1** |
| forced | "D — The image is an abstract representation and does not depict a physical object like a ball that can fall, move, or stay still." | **0** |

→ **Open-ended PMR = 1, forced-choice PMR = 0.** On the same image, just
swapping the prompt flips physics-mode on / off. This is the cleanest
example of §3.2 "language-prior dominance". Compared to `line/blank/none`,
adding a single ground line shifts the open-ended response from "stationary"
to "rolls" — the minimal pair for the ground effect.

#### (3) `shaded + ground + none` — canonical physics stimulus

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will fall downward due to gravity." | **1** |
| forced | "A) It falls down. — The ball appears to be in free fall above a horizontal line, which typically represents the ground. In the absence of any forces acting on it other than gravity, the ball would naturally fall downward towards the ground." | **1** |

→ Both prompts → **A (falls)**. 3D shading + ground is the easiest condition
for Qwen2.5-VL to recognize as physical.

#### (4) `textured + ground + arrow_shadow` — maximum cue

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will collide with the surface below it." | **1** (GAR=1) |
| forced | "A — The red arrow indicates a downward direction, suggesting that the ball will fall due to gravity." | **1** (GAR=1) |

→ Explicitly cites the arrow ("the red arrow indicates"). This cue is too
strong — it saturates PMR at 1.0, an unmeasurable cell. To be split into
shadow-only / arrow-only in MVP-full.

#### (5) `filled + blank + wind` — wind cue invisible

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will remain stationary as there is no indication of movement or force acting upon it." | **0** |
| forced | "D — The image is an abstract representation and does not depict a physical object in motion or at rest." | **0** |

→ **"no indication of movement or force"** — the model does not read the
wind streaks on the left as airflow. Behaves identically to the `none`
condition. The programmatic wind cue fails as a stimulus.

#### (6) `textured + blank + none` — object-level effect in isolation

| Prompt | Response | PMR |
|---|---|---|
| open | "The ball will likely roll or bounce due to an external force such as gravity or a push." | **1** |
| forced | "D — The image is an abstract representation and does not depict a physical object in motion or at rest." | **0** |

→ Another open-vs-forced split. The textured ball alone elicits a physical
description in open-ended, but with the D option available in forced-choice
the model escapes to "abstract". Same pattern as cell (2) — the visual cue
is weak, the language prior dominates open-ended.

---

## 2. Headline numbers

| Condition | PMR | Interpretation |
|---|---|---|
| **Ground present?** (blank → ground) | 0.488 → **0.854** (+36 pp) | Largest single-factor effect. Aligns with the Kellman / Spelke / Gibson cog-sci prediction that a support plane induces physical-object interpretation. |
| **Abstraction axis A** (line → textured) | 0.575 → **0.808** | Endpoints match H1 (S-curve); the middle two levels (filled 0.658, shaded 0.642) are tied → middle cues give no marginal gain. |
| **Cue arrow_shadow** | **1.000** (saturated) | The arrow is too strong → MVP-full must split into shadow-only / arrow-only. |
| **Cue wind** | 0.513 ≈ none 0.500 | The programmatic wind streaks **are not interpreted as airflow** by the VLM. |
| **Open vs forced-choice** | PMR 0.800 vs 0.542, abstract_reject 0.000 vs 0.450 | In open-ended the model **never** spontaneously calls the stimulus abstract. With option D available in forced-choice, 45 % escape to abstract. |

---

## 3. Insights derived

### 3.1 Background (ground plane) is the strongest single trigger

The `blank` vs `ground` difference (same object, same cue) shifts PMR from
0.49 → 0.85. So **the easiest entry point into physics-mode is the
"ground + object" composition**, not 3D shading or texture.

- Kersten / Mamassian / Knill (1997): cog-sci finding that cast shadows
  attach an object to the ground — same direction.
- Gibson (1979) ecological approach: surface / ground plane / optic flow
  are the primary cues for objecthood.
- **Implication**: Figure 1 candidate for the main paper is **§1.3 cell (1)
  vs cell (2)** minimal pair — "the same line circle is stationary on
  blank, rolls on ground". The paper's one-line copy comes from this.

### 3.2 "Encoder knows, decoder blurts" — language-prior dominance

Open-ended PMR = 0.80, abstract_reject = 0.00. Qwen2.5-VL-7B **never
spontaneously says "this is an abstract shape"** in open-ended mode. On
the same image, when option D ("abstract shape") is available in
forced-choice, 45 % choose it.

§1.3 cells (2) and (6) are the cleanest examples of this gap. Same image,
open PMR = 1, forced PMR = 0.

This is structurally identical to the **"Vision Language Models Are Biased"**
(Vo et al. 2025) pattern from `references/project.md` §1.6: the model can see
the image, but in free-form generation the **language prior** (the word "ball")
overrides the visual evidence.

- **Key insight**: the behavioral signal we wanted to measure ("physics-mode
  priming") moves **far more along the language axis than the image axis**
  in open-ended. The clean behavioral S-curve must be read off the
  **forced-choice subset**.
- **Methodological lesson**: the paper's main behavioral figure should use
  forced-choice PMR. Open-ended PMR is interesting as the gap between
  "upper bound (with language prior)" and "lower bound (with abstract
  escape route)" — that gap itself is a finding.

### 3.3 Abstraction-axis S-curve — partial confirmation

| object_level | PMR | Δ |
|---|---|---|
| line | 0.575 | (base) |
| filled | 0.658 | +0.083 |
| shaded | 0.642 | −0.016 |
| textured | 0.808 | +0.166 |

Endpoints match H1 but **the middle two levels are tied** → the parametric
S-curve isn't clean. Possible causes:

1. **Noise**: n=120 / cell → std error ≈ 4.5 pp, so a ±1 pp swap is
   statistically tolerable.
2. **The middle cues are not distinguished by the VLM**: the difference
   between filled circle and shaded sphere may not propagate meaningfully
   through CLIP / SigLIP encoder to the behavioral output. This is itself
   the direct motivation for Sub-task 2 (vision-encoder probing).

### 3.4 Wind cue is invisible — a stimulus-side defect

The programmatic 5-cluster wind streaks have **almost no effect on PMR**
(wind 0.513 vs none 0.500). §1.3 cell (5) is a concrete example — the model
explicitly denies it ("no indication of movement or force"). Some readings:

- The VLM doesn't recognise small grey arcs as airflow. "Abstract wind
  symbols" are rare in training data.
- A style that's easily read by human subjects is OOD for the VLM.
- → In MVP-full replace with **motion blur** or **dust trail** (photoreal),
  or redesign as a VLM-friendly representation like "trail of spheres
  fading out".

**Research implication**: what cog-sci literature treats as a "strong motion
cue" and what **the VLM actually recognises as motion** are different.
Systematically mapping this gap could itself be an EMNLP-tier result —
framing like "Human-salient physical cues that VLMs miss".

### 3.5 The arrow + shadow cue is **too strong** — an unmeasurable cell

`arrow_shadow` produces PMR 1.000 (saturated). In §1.3 cell (4) the model
explicitly cites the arrow ("the red arrow indicates a downward direction").
We can't see meaningful differences. This cue is really two cues combined:

1. The cast shadow on the ground (Kersten-style ground-attachment cue).
2. The red directional arrow (a direct annotation of motion — almost a label).

To measure (1)'s theoretical effect we have to separate them. MVP-full's
axis C should be redesigned to:
- `none`
- `cast_shadow_only`
- `motion_arrow_only`
- `both`

### 3.6 Event template has no effect — contrary to expectation

`fall` vs `horizontal`: PMR is **identical at 0.67 each**. There was an
implicit assumption that varying the object's on-canvas position would
change behavior, **refuted by the data**.

- Interpretation: the VLM looks at "is there a ground in the image" and
  "is there a cue" before object position. Event template matters to the
  stimulus generator but is barely visible in behavioral output.
- **In MVP-full, downgrade event_template from a factorial axis to a
  controlled variation**. Reinvest the same factorial budget into seeds
  or axis D (label).

### 3.7 Model responses are "systematic, not random"

The 6 cell responses in §1.3 directly demonstrate this:

- blank + no cue: "remain stationary unless acted upon" → Newton's First
  Law framing. The default in the absence of context is **inertia**.
- ground + no cue (shaded or above): "fall downward due to gravity" →
  the presence of ground alone activates the gravity narrative.
- ground + arrow_shadow: "collide with the surface below" → explicitly
  follows the arrow.
- blank + wind: "no indication of movement" → the wind cue isn't
  ignored, it's **not perceived**.
- forced-choice + abstract stimulus: "D — abstract representation, does
  not depict a physical object" → meta-cognitive "this is just a shape"
  utterance.

**What this means**: the model's "errors" are not random hallucinations.
For each cue it responds via **interpretable, reproducible rules**. This
confirms the premise that Sub-tasks 3-4 (layer-wise localization, causal
patching) target **measurable phenomena**. If responses were truly random,
internal-mechanism localization would be impossible.

---

## 4. Methodological lessons

### 4.1 The scoring lexicon is living

The initial `PHYSICS_VERB_STEMS` had `"move"` as a stem, which **missed
"moving"** (prefix match: `"moving"` does not start with `"move"`). Other
natural physics verbs like "continue", "rotate", "glide" were missing.
Discovered when actual pilot textured/ground/horizontal responses fell
out as false negatives.

- **Lesson**: stems should be the shortest common prefix (`"move"` → `"mov"`,
  `"continue"` → `"continu"`). Periodically collect false negatives and
  add to the regression test. 5 regression cases now in
  `tests/test_pmr_scoring.py::PMR_POSITIVE`.

### 4.2 RC is useless at temperature = 0

Deterministic decoding makes every seed in the same cell produce the same
generation → RC = 1.0 for every cell. The Response Consistency metric
promised in `references/project.md` §2.2 was **not measurable** in the pilot.

- **Lesson**: in MVP-full, set `temperature=0.7` and raise `seeds_per_cell`
  to 10-15 to obtain a **meaningful RC distribution**.

### 4.3 Streamed JSONL actually helped

Each inference flushes to JSONL. Even an 8-min pilot crash would have
preserved completed rows. This decision pays off much more for the
several-hour MVP-full run.

### 4.4 The factorial as designed and the factorial that actually moves are different

At design time we assumed all 5 axes (A/B/C/D/events) would be informative.
In reality:

- Big movers: **B (background)** > **prompt_variant** > **A (abstraction)
  endpoints**.
- Barely moving: **C's wind**, **event_template**.
- Too big a mover: **C's arrow_shadow** (saturated).

**Lesson**: MVP-full should reallocate axes based on the pilot numbers.
In particular, split the cue axis more finely and collapse the event axis,
reinvesting that capacity into the label axis (D) and seeds.

---

## 5. Original-hypothesis comparison

Status of the `references/project.md` §2.2 hypotheses against the pilot data:

| Hypothesis | Statement | Pilot result | Status |
|---|---|---|---|
| H1 | PMR rises **S-shaped** along the abstraction axis. 3D shading and ground introduction produce the largest step changes. | Endpoints match (line 0.58 → textured 0.81). Middle two tied. The ground introduction effect was confirmed at +36 pp (not the main effect itself, a separate axis B). | **partial support** |
| H2 | The "ball" label substantially raises PMR even on line drawings → independent contribution of the language prior. | Open-ended PMR 0.80 with the "ball" prompt alone, abstract_reject 0.00. With a forced-choice escape route, half escape to abstract. §1.3 cells (2) and (6) are the evidence. | **strong support** |
| H3 | Scene inconsistency degrades RC. | Not measurable (the pilot didn't include the scene-inconsistency axis + RC was saturated at T=0). | **untested** |

For MVP-full to test H3: `temperature=0.7`, seeds≥10, and include axis E
(scene consistency) at least at 2 levels.

---

## 6. Concrete changes for the next round

Captured partly in `docs/experiments/m1_pilot.md` and `docs/next_steps.md`,
but the firm recommendations grounded in the pilot data:

1. **Redesign axis C**: split into `{none, cast_shadow_only, motion_arrow_only, both}`.
   The current combined `arrow_shadow` produces an unmeasurable cell.
2. **Replace the wind cue**: motion blur or dust trail. Programmatic arcs
   are invisible to the VLM.
3. **Collapse event template**: fix to `fall` (horizontal has identical
   behavior). Reinvest into seeds and labels.
4. **Expand axis D**: `("circle", "ball", "planet")` 3 levels. Quantify
   the H2 language-prior contribution.
5. **Temperature 0.7 + seeds_per_cell ≥ 10**: makes RC measurable.
6. **Enable activation capture**: `capture_lm_layers=(5, 10, 15, 20, 25)`.
   Reused by Sub-task 3 (logit lens).
7. **The headline figure should be the forced-choice subset**. Open-ended
   goes in the appendix as evidence of language-prior dominance.

---

## 7. Implications for the paper narrative

### 7.1 Adjusting the headline-claim candidates

Of the 4 headline-claim candidates noted in the original plan, the two
strongest after the pilot are:

1. **"VLMs see a circle as a ball but don't say so"** (§3.2 open-ended vs
   forced-choice gap) — confirmed in the pilot. 0.80 vs 0.54, abstract_reject
   0.00 vs 0.45. §1.3 cell (2) is the 1-figure version.
2. **"A single ground line activates physics-mode"** — single factor of
   +36 pp. Figure-1-grade minimal-pair demo possible (§1.3 cell (1) vs (2)).

The original headline candidates "S-shaped switching curve" and
"encoder-decoder boomerang" weren't sharp at the pilot scale → push these
to **Sub-task 2 (vision-encoder probing)** to get sharpness.

### 7.2 Venue framing revisited

- **EMNLP (grounding angle)**: the pilot's strong open-vs-forced-choice
  gap + the ground effect directly support the "grounding failure"
  narrative. **EMNLP Interpretability and Analysis of Models for NLP**
  is increasingly the right placement.
- **NeurIPS (interpretability)**: weak until Sub-tasks 2-4 land. But if
  Sub-task 3 (logit lens) shows what we predict, "when does physics-mode
  emerge?" answered at the layer level brings NeurIPS back into play.

### 7.3 Risk R1 (no clean switching) is **partially realised**

The pilot didn't show a clean H1 S-curve. Per the R1 risk fallback:

- **Reframe via cross-axis interaction rather than single axes**: "which
  cue combination produces saturation" — arrow_shadow + ground → PMR=1;
  line + blank + open-ended (ball label) → PMR=0.85 (language prior
  alone). This 2D interaction is the story.
- Human baseline collection becomes increasingly important: if humans
  also show tied PMR at the middle levels on the same stimuli, then H1
  itself was a less-sharp cog-sci hypothesis.

---

## 8. One-line summary

> Qwen2.5-VL-7B **renames a circle a ball as soon as a single ground line
> is drawn**, and **fails to recognize the abstraction even given the word
> "ball" alone**. The visual abstraction-axis S-curve is only visible at
> the endpoints; the wind streak and directional arrow are respectively
> too weak and too strong to use as measurement axes — these four points
> are the basis for the MVP-full design changes.
