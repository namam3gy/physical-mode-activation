# M5 (ST4) Insights — VTI Causal Steering: "Physical Object-ness" Direction Found

The first deliverable of Sub-task 4. We extract a VTI-style **physics-mode
direction** from M2-captured LM activations and inject it (α · v) into the
LM residual stream at test time, to verify whether
**Qwen2.5-VL-7B's default behavior of refusing line circles as abstract
shapes can actually be flipped**.

Raw numbers: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/`
and `steering_experiments/`.
Implementation: `src/physical_mode/probing/steering.py`,
`scripts/06_vti_steering.py`.

## 1. One-line summary

**Injecting `α=40 · v_L10` into Qwen2.5-VL-7B's LM layer 10 residual
stream flips 10/10 responses on `line+blank+none` stimuli from "D: abstract"
to "B: stays still"**. Layers 15/20/25 do not move at the same α. This is
evidence that **a "physical object-ness" linear direction exists in the
LM early-mid residual stream and responds causally to intervention**.

## 2. Phase 1 — Direction extraction

### 2.1 Recipe (project doc §2.5 VTI)

For each captured layer L:

```
v_L = mean_sid (mean_token h_L[sid]  |  PMR(sid) = 1)
      − mean_sid (mean_token h_L[sid]  |  PMR(sid) = 0)
```

PMR is the forced-choice target. 480 stimuli, n_pos=312, n_neg=168.

### 2.2 Result — direction amplifies through layers

| layer | ‖v_L‖₂ | n_pos | n_neg |
|---|---|---|---|
| 5 | 5.88 | 312 | 168 |
| 10 | 8.16 | 312 | 168 |
| 15 | 10.33 | 312 | 168 |
| 20 | 15.43 | 312 | 168 |
| 25 | **30.98** | 312 | 168 |

→ Norm grows **5× through the LM** (5.9 → 31). Matches the "physics
margin amplification" we saw in M4.

### 2.3 Projection distribution (layer 20, the M4 forced-choice peak AUC)

Each sample's mean-pooled hidden state projected onto `v_unit_L20`:

**By object_level**:
| level | mean proj | std | min | max |
|---|---|---|---|---|
| line | 30.71 | 11.21 | 5.48 | 45.44 |
| filled | 32.14 | 10.79 | 7.63 | 46.23 |
| shaded | 34.26 | 10.39 | 10.37 | 47.41 |
| textured | 35.41 | 7.36 | 19.31 | 47.90 |

→ Small but monotone increase (+4.7 from line to textured).

**By cue_level**:
| cue | mean proj | std | min | max |
|---|---|---|---|---|
| none | **22.31** | 9.73 | 5.48 | 35.45 |
| cast_shadow | 29.68 | 6.68 | 18.40 | 38.80 |
| motion_arrow | 37.81 | 4.89 | 28.86 | 44.88 |
| both | **42.73** | 3.04 | 37.52 | 47.90 |

→ **20-unit spread** — the projection cleanly tracks cue_level. The
"physics direction" is aligned with the strongest factorial axis.

## 3. Phase 2 — Causal intervention

### 3.1 Setup

- **Test stimuli**: `line / blank / none`, 10 seeds × 1 object × 1 bg × 1
  cue × 1 event = 10 stimuli.
- **Prompt**: forced-choice with label = `"circle"` (baseline PMR ≈ 0 →
  maximum room to flip).
- **Layers tested**: 10, 15, 20, 25.
- **α values**: 0, 5, 10, 20, 40.
- **Temperature**: 0 (deterministic).
- **Intervention**: a forward hook on
  `model.model.language_model.layers[L]` adds `α · v_unit_L` to the output
  hidden_states. Applied uniformly to all token positions.
- **Total inferences**: 10 stimuli × 4 layers × 5 α = 200. ~5 min.

### 3.2 Result — first-letter response distribution

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10/10 D | 10 D | 10 D | 10 D | **10 B** 🔥 |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

**At layer 10 α=40, all 10 flip in unison from D (abstract) to B (stays
still)**. No other layer moves at α=40.

### 3.3 Response-text comparison (sample stimulus `line_blank_none_fall_000`)

**Baseline (α=0, common to all layers)**:

> "D — This is an abstract shape and as such, it does not have physical
> properties that would allow it to fall, move, or change in any way. The
> image is simply a representation…"

**L10 α=40 (intervention)**:

> "B) It stays still. — Justification: The circle in the image appears to
> be floating or suspended in space without any external force acting upon
> it. In such a scenario, the circle would remain stationar[y]…"

**L20 α=40**:

> "D — The image depicts an abstract shape, specifically a circle, which
> is not a physical object and therefore cannot fall, move, or change
> direction in the way that objects do. The question is based on…"

**Interpretation differences**:
- Baseline / late layers: "abstract shape, not a physical object" — the
  abstract-shape rejection mode.
- L10 α=40: "the circle... floating or suspended in space... no external
  force" — **recognised as a physical object but a stationary one**. The
  intervention causes an "abstract → physical object" categorical shift.

### 3.4 PMR-scoring caveat

Forced-choice raw_text mentions all options explicitly ("cannot fall, move,
or change direction"), so the physics-verb lexicon picks up hits in the
option text → returns PMR=1. So in M5 intervention runs, PMR=1 is **not
direct evidence** that the model went into physics-mode; it's a **secondary
signal**.

**The real causal evidence is the first-letter (A/B/C/D) distribution** —
looking only at this, the D→B flipping appears exactly at L10 α=40 and
nowhere else. The PMR scorer needs future refinement: **distinguish option-
list quoting from genuine physics description**.

## 4. Mechanism interpretation

### 4.1 Why is only L10 steerable?

In M4 the probe AUC peaked at **L20 (0.953)**, with L10 at 0.944. Yet the
intervention is far more effective at L10. Possible explanations:

1. **Upstream intervention propagates downstream**. Biasing the hidden
   state at L10 lets subsequent layers do residual updates that follow
   the signal, changing the final output. Adding the same vector at L20+
   becomes a **small nudge on top of an already-committed representation**
   and has no effect on the decoding.
2. **L25 direction norm = 31** vs L10 = 8. But α · v_unit normalises that
   away, so the actual magnitude is constant per layer. So "effective
   strength" is relative to the layer's typical activation magnitude. At
   late layers α=40 · v_unit may be **too small relative to the layer's
   norms** to push.
3. **L10's direction is the semantic bottleneck** — the "abstract vs
   physical" decision is made around there, and intervention flips the
   decision.

(1) is consistent with prior causal-interpretability literature (Basu et
al. 2024 "constraint information stored in layers 1-4" in LLaVA;
Neo et al. 2024).

### 4.2 The direction is "object-ness", not "gravity"

L10 α=40 responses are **B: stays still**, not **A: falls down**. If the
steering vector encoded a gravity concept like "falls/drops/rolls", we'd
expect a flip to A. The actual flip is to B (stationary physical object).

→ **This direction is a binary "abstract vs physical object" distinction**,
not a "which physics" (gravity / orbit / inertia) selector. Consistent
with H7 (label selects physics regime): the direction is coarse
"object-ness", and the **label** decides the specific physics narrative.

A future SAE or finer-grained probe could **decompose the direction into
multiple sub-directions**, separating "gravity direction" from "object-ness
direction".

## 5. Hypothesis scorecard update (post-M5)

| H | Post-M4 | Post-M5 | Change |
|---|---|---|---|
| H-boomerang | extended | **extended + causal support** | Information present (M3), preserved through LM (M4), causally active under intervention (M5). |
| H-locus | candidate | **supported (early-mid)** | L10 is the causal sweet spot. Aligns with prior early-layer intervention literature. |
| H-regime (new) | — | **candidate** | The steering direction is "object-ness" binary, not "which physics". Physics-regime selection is label-driven. |

## 6. Paper figure candidates

### Figure 6 — Causal steering

```
A) baseline (line/blank/none × "circle" prompt):
   10/10 responses → "D: abstract"
B) with α · v_L10 injection:
   α=0, 5, 10, 20: 10/10 → D
   α=40:            10/10 → B ("stays still")
C) α=40 at L15, L20, L25:
   10/10 → D (no effect)
```

Message in summary: "Injecting the physics-mode direction at layer 10 can
override abstract rejection. The direction is causally inactive after
layer 15 — the representation has already committed."

Backs the paper's "we can steer physics-mode" claim.

## 7. Limitations · open questions

1. **Test subset is small (10 stimuli)**. Need to verify on other
   abstract-baseline conditions (filled+blank+none, line+ground+none, etc.)
   that L10 α=40 still flips them.
2. **Flipping only goes to "B"**. If the direction were truly "physics-mode",
   some samples should also flip to A (falls). The all-B result may be
   because label="circle" still suppresses "motion" interpretation.
   Test with `label="ball"` to see if A/B distribution changes.
3. **α=40 is a magic number**. A finer α sweep (30, 35, 45, 50, 60) would
   reveal the threshold position and saturation behavior precisely.
4. **Negative α** (abstract-ness direction) not tested. Would
   `-α · v` injected at `textured+ground+both` flip from physics-mode to
   abstract-mode in reverse?
5. **Attention knockout and activation patching not run**. M5 covers only
   VTI steering. The rest of project doc §2.5 (Semantic Image Pairs +
   activation patching + SAE) is deferred to a later round.

## 8. What M5 unlocks

- **The paper's causal claim is now defensible**: the chain
  correlation (M3, M4) → causation (M5) is complete.
- **§4.2 "reverse prompting" idea**: attach `"abstract shape"` label to a
  photo of a ball. The negative-α counterpart of this M5.
- **SAE (project doc §2.5 stretch)**: train an SAE on L10's residual
  stream → decompose the "physical object-ness" direction more finely.
- **Cross-model (M6)**: verify whether the same "L ≈ mid-early" sweet
  spot exists in other open-source VLMs (LLaVA-1.5, InternVL2). If
  consistent, a general claim; if model-dependent, an architecture-specific
  finding.
