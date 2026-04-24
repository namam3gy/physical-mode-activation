# M4 (ST3) Insights — LM Logit Lens + Per-Layer Probes

Sub-task 3 is the **internal extension** of the M3 boomerang: if the vision
encoder transmits information perfectly (M3), then **how far** through the
LM does that information survive, and **where** does it leak out before
decoding?

Raw numbers: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.csv` ·
`*.parquet`. Implementation: `src/physical_mode/probing/lm.py`, driver:
`scripts/05_lm_probing.py`.

## 1. One-line summary

**Qwen2.5-VL-7B's LM linearly predicts physics-mode PMR from the
visual-token hidden states at every captured layer (5, 10, 15, 20, 25)
with AUC ≈ 0.94-0.95.** Information passes through the LM almost without
loss — the gating happens at the **token-generation (decoding) step**.

## 2. What we measured

### 2.1 Setup

- Input: M2-captured LM hidden states at `layers=(5, 10, 15, 20, 25)` of
  Qwen2.5-VL-7B-Instruct's 28-layer language model. Per stimulus: 324
  visual tokens × 3584 dim, bf16.
- Probe: `sklearn.LogisticRegression`, 5-fold stratified CV, mean-pool
  across visual tokens, StandardScaler.
- Logit lens: apply `lm_head` (unembedding projection) to the hidden state
  at each layer → track logits for a curated set of token ids. Mean-pooled
  across visual tokens.

### 2.2 Token sets

- **Physics verbs** (15 tokens): `fall`, `falls`, `falling`, `drop`, `drops`,
  `roll`, `rolls`, `rolling`, `bounce`, `slide`, `land`, `tumble`, `move`,
  `moving`, `orbit`.
- **Geometry / static** (10 tokens): `circle`, `shape`, `line`, `drawing`,
  `image`, `figure`, `abstract`, `geometric`, `still`, `static`.
- **Label** (3 tokens): `ball`, `planet`, `object`.

We keep only those that the tokenizer maps to a single sub-token.

## 3. Headline numbers

### 3.1 Per-layer PMR probe (forced-choice target)

| layer | AUC (mean ± std) | accuracy |
|---|---|---|
| 5 | 0.939 ± 0.015 | 0.867 |
| 10 | 0.944 ± 0.006 | 0.881 |
| 15 | 0.947 ± 0.009 | 0.885 |
| **20** | **0.953 ± 0.007** | 0.885 |
| 25 | 0.944 ± 0.009 | 0.877 |

→ **PMR is linearly separable at every layer**. Peak at layer 20 (0.953).
Matches the pattern Neo et al. 2024 reports for LLaVA-1.5 (object-specific
features crystallize in layers 15-24). Qwen2.5-VL-7B (28 LM layers) also
peaks in the mid-to-late range.

Comparison:
- Vision encoder AUC @ L31: **0.944**
- LM hidden @ L20: **0.953**
- LM hidden @ L25: 0.944

→ **Vision encoder and LM hidden states have nearly identical
discriminability**. Information is preserved almost without loss as it
passes through the LM.

### 3.2 Logit lens — the "ball" label primes the LM at L5 already

Mean logits over all 480 stimuli:

| layer | geometry | physics | label |
|---|---|---|---|
| 5 | 0.93 | 1.04 | 1.16 |
| 10 | 1.35 | 1.66 | 1.73 |
| 15 | 2.04 | 2.45 | 2.29 |
| 20 | 3.23 | 4.18 | 4.09 |
| 25 | 11.56 | **15.64** | 13.96 |

**At L5, physics > geometry already (1.04 > 0.93)**. This is because the
"ball" label is in the prompt, priming the LM into physics-mode from the
start.

At layer 25 the physics logit is 15.64 vs geometry 11.56 → **physics
dominance is dramatically amplified at the final layer** (gap 4.0). Read:
the LM's final residual updates strengthen physics narratives like "this
is a falling ball".

### 3.3 Physics margin per object_level

Physics margin = mean(physics logit) − mean(geometry logit):

| layer | filled | line | shaded | textured |
|---|---|---|---|---|
| 5 | 0.08 | 0.09 | 0.12 | 0.15 |
| 10 | 0.29 | 0.27 | 0.33 | 0.38 |
| 15 | 0.38 | 0.35 | 0.44 | 0.49 |
| 20 | 0.89 | 0.87 | 0.97 | 1.05 |
| 25 | 3.94 | 3.76 | 4.29 | 4.35 |

- **From L5 onward the abstract → concrete ordering is monotone** (line /
  filled below shaded / textured, with ~0.01 gap at L5).
- **Maximum gap at L25** (textured 4.35 vs line 3.76 = +0.59).
- But this object-induced gap (0.6) is **far smaller than the label-induced
  shift (which lifts everything by ~4 units)** → **the label has 7× the
  effect of the visual evidence**.

### 3.4 Switching layer (limitation)

Using `switching_layer = earliest layer where max(physics token logits) ≥
max(geometry token logits)`, all 480 samples switch at **layer 5** — i.e.
physics already wins at the lowest captured layer. This metric is
uninformative in the current setup — as long as the "ball" label is in
the prompt, the LM has a physics-mode prior from the start.

**To make it more informative**:

- Capture more densely (0, 1, 2, 3, 4, 5 ...) — there might be a geometry-
  dominance region before L5.
- Re-run with a label-free prompt ("What do you see?") — observe the
  switching position after the label prior is removed.

Both are deferred to M4 follow-up (or §4.9 "label-free prompt").

## 4. Combined interpretation — M3 + M4

**Reversibility through the three-stage pipeline**:

| stage | PMR discriminability |
|---|---|
| Vision encoder (M3, layer 31) | AUC **1.00** on stimulus truth, **0.944** on behavioral PMR |
| LM hidden state @ visual tokens (M4, layer 20) | AUC **0.953** on behavioral PMR |
| Behavioral output (actual decoding) | forced-choice accuracy ≈ 0.66 |

**Information is preserved nearly perfectly from vision encoder
(0.94-1.0) → LM hidden state (0.95)**. The discrete generation step
loses ~28-29 pp of accuracy (against the theoretical correct-prediction
rate that an AUC of 0.95 implies, vs the actual forced-choice accuracy).

The "boomerang" now has a precise location:

```
vision encoder ──(info present)──→ LM early layers ──(info present)──→ LM late layers ──(info present)──→ decoding ──(partial loss)──→ token output
```

**The bottleneck is the decoding step itself**, or more precisely,
somewhere in "LM hidden → logit distribution → token sampling". This
gives ST4 (activation patching) a direct target: clean/corrupted pair
patching at the **final residual update** or the **logit bias** should
flip decoding decisions.

## 5. Hypothesis scorecard update (post-M4)

| H | Prior | Post-M4 | Change |
|---|---|---|---|
| H-boomerang | supported (M3) | **extended** | Information survives the entire LM. Gating happens in decoding. |
| H7 (label = physics regime) | candidate (M2) | **supported** | Logit lens shows the label prior shifts physics-geom margin from L5 onward. 7× the magnitude of the object_level effect. |
| H-locus (new) | — | **candidate** | Bottleneck is at LM final layers + decoding. ST4 activation patching can confirm. |

## 6. Plot ideas worth using as paper figures

### Figure 4 (M4): the trajectory of information through the LM

- X axis: layer (5, 10, 15, 20, 25)
- Y left axis: probe AUC (forced-choice) — ~0.94 across all layers
- Y right axis: physics margin (phys − geom logit) — 0.1 → 4.0 amplification
- Per-object_level line plot

Message: "information survives the LM (AUC flat); the physics narrative
amplifies at the final layer (margin spike); but decoding does not fully
use that amplification (behavior < the upper bound implied by AUC)".

### Figure 5 (M4 + M5 predicted): boomerang localized

Layer-wise probe AUC (vision encoder + LM) vs behavioral PMR, horizontal
bars or cumulative plot. The visual answer to "where does the information
get thrown away".

## 7. What we hand off to M5

- **ST4 (activation patching) direct target**: LM final layers (20-28) +
  decoding head. Why: probe AUC peaks at L20 but behavioral accuracy is
  0.66 → "the LM has the information but doesn't reflect it in output".
  Patching intervention will be most effective there.
- **Action required**: re-capture with `capture_lm_attentions=True` (M2
  doesn't save attentions). Construct SIP pairs from a mini-factorial
  and run patching → Sub-task 4 complete. Disk cost: ~15 GB.

## 8. Limitations

1. **The switching-layer metric is degenerate under label-primed prompts**.
   A label-free ("What do you see?") rerun is needed → naturally connects
   to §4.9.
2. **5-layer snapshot is coarse**. In particular L0-L5 and L25-L27 are
   empty. A dense-capture follow-up (every layer) may be necessary.
3. **Mean-pool over-simplifies**. Per-visual-token-position probing (Neo
   et al. 2024 heatmap style) hasn't been done yet.
4. **Logit lens uses only `lm_head`'s linear projection** — at actual
   token generation, softmax + sampling are added. The stochasticity in
   those final steps may be the cause of the gap between AUC 0.95 and
   behavioral accuracy 0.66 (T=0.7 setting).
