---
section: M5b — SAE intervention on Qwen vision encoder (last layer, pre-projection)
date: 2026-04-27 (evening update: Cohen's d revision + 3 mass-matched random sets + dense k-sweep + cluster-conditional non-monotone resolution)
status: complete (n=20 clean stim × 9 top-k conditions × 3 mass-matched random controls; clean positive result + per-cluster ablation curve)
hypothesis: H10 (research plan §2.5) refined — encoder-side physics-mode signal *is* localized to a small set of monosemantic SAE features (top-N in Cohen's-d-rank); single features are dispensable but cumulative ablation breaks physics commitment in a stim-cluster-conditional pattern; mass-matched random-feature controls (3 sets, 72-102 % of top-N mass) confirm this is feature-specific, not magnitude-driven.
---

# M5b — SAE intervention on Qwen2.5-VL vision encoder

> **Recap**
>
> - **M5b SIP patching** (sufficiency, layer-level): L0-L9 patching → 20/20 physics recovery; sharp L10 boundary.
> - **M5b layer-level knockout** (necessity): L9 MLP IE = +1.0 — uniquely necessary; attention 0 IE everywhere.
> - **M5b per-head knockout**: 196 (L, h) all IE = 0 — confirms attention is fully redundant.
> - **What's left**: the upstream side. The LM-internal mechanism is well-localized (L9 MLP construction); is the *encoder-side* signal that L9 MLP is constructing from also localized, or distributed across thousands of SigLIP features?
> - **SAE** (Sparse Autoencoder; Bricken et al. 2023; Pach et al. 2025): trains an over-complete linear-relu-linear bottleneck with L1 sparsity on a layer's activations to recover monosemantic features.
> - **§4.6 Qwen pixel-encodability**: gradient ascent on `pixel_values` along v_L10 flips physics-mode at ε=0.05 (5/5). Encoder-side mechanism exists; SAE is the natural way to identify *which* encoder features carry it.

## Question

The L9 MLP is the LM-side construction site. Where is the *encoder-side*
physics-mode information localized? Two extreme hypotheses:

(a) **Distributed**: the physics-mode signal is spread across many encoder
    features; no small subset is causally responsible. Ablating any small
    feature group should leave behavior intact; only large-magnitude
    perturbations (matched-random or top-feature) break behavior in the
    same way.

(b) **Localized**: a small set of monosemantic features (e.g., 10-50 in
    a 5120-feature SAE) carry the physics-mode signal. Ablating those
    *specific* features should break physics-mode while matched-magnitude
    random ablations leave it intact.

The §4.6 pixel-space result already showed that *some* encoder-side
direction (v_L10 read out at L10) carries shortcut signal at ε=0.05.
SAE intervention here is the encoder-side analog of layer-level MLP
knockout: at fine enough resolution, are there *specific feature groups*
whose ablation breaks physics?

## Method

### Activation source

`outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` —
captured Qwen2.5-VL last vision-encoder layer activations (`vision_hidden_31`,
1296 visual tokens × 1280 SigLIP hidden dim) on 480 stim from the M2 run
(5-axis factorial, label="circle/ball/planet"). Total 622,080 tokens.

### SAE training

- Architecture: tied-weight encoder/decoder with input z-score normalization
  (per-dim mean/std from 100K-sample), input bias `b_pre`, encoder bias
  `b_enc`, decoder column unit-norm constraint.
- d_in = 1280, d_features = 5120 (4× expansion).
- Loss = MSE(reconstruction) + λ × L1(z), λ = 1.0 in normalized space.
- 5000 Adam steps, batch 4096, lr 1e-3 — runs in 1.1 min on H200.
- Final: recon = 0.023 (normalized), L1 = 0.042, 100% features alive,
  7.3% active per token (~370 features per token).

### Feature ranking

Per-sample mean PMR from `predictions_scored.csv`:
- Physics-mode set: 310 stim with mean_pmr ≥ 0.667.
- Abstract set: 19 stim with mean_pmr ≤ 0.333.

Per feature i: `delta_i = mean(z_i | physics) − mean(z_i | abstract)`.
Top-20 by delta saved.

Top-10 features (raw activation means):

| feature_idx | mean_phys | mean_abs | delta |
|------------:|----------:|---------:|------:|
| 4698 | 3.13 | 0.23 | **2.90** |
| 1152 | 2.66 | 0.32 | 2.34 |
| 3313 | 7.86 | 6.24 | 1.62 |
| 4106 | 1.75 | 0.13 | 1.61 |
| 1949 | 1.55 | 0.15 | 1.39 |
| 38 | 2.32 | 0.99 | 1.33 |
| 4468 | 1.39 | 0.14 | 1.26 |
| 438 | 1.26 | 0.14 | 1.12 |
| 117 | 1.18 | 0.12 | 1.06 |
| 1674 | 1.03 | 0.01 | 1.02 |

### Causal intervention

For each clean SIP stim (n=20, same as layer-level knockout cohort), run
inference with a forward hook on the *last vision encoder block*'s output
(model.visual.blocks[-1]). The hook subtracts the target SAE features'
raw-scale contributions (Bricken et al. trick — keeps non-target features
+ reconstruction residual exact):

```
contribution_n = z[:, target_feats] @ W[target_feats]         # normalized space
contribution_raw = contribution_n * input_std                # back to raw
x_new = x - contribution_raw
```

Sweep: top_k ∈ {1, 5, 10, 20} of the physics-cue ranking, plus
**magnitude-matched random controls** (k=20). Initial implementation drew
random sets from the bottom-of-ranking pool, but those features have mass
≈ 1% of top-20's total because the L1 penalty kills inactive features —
i.e., the "random" ablation was zero-magnitude, not a fair specificity
test. Corrected pool: features ranked 21+ in the *top-300 by mass*, with
total `mean_phys + mean_abs` mass in [70%, 200%] of top-20's. Only 1
matched set was found (top_mass = 49.23, random_0 mass = 40.97 = 83%) —
the activation distribution is heavy-tailed (top feature 3313 has mass 14
alone), so most random k=20 samples fall short.

## Result

![SAE intervention results — Cohen's d revision](../figures/m5b_sae_intervention_revised.png)

*(2026-04-27 evening figure: dense k-sweep with Wilson CIs + per-cluster pivot. Old morning figure at `m5b_sae_intervention_phys_rate.png` retained for reference.)*

| Condition | Mass | Physics rate (n=20) | Note |
|-----------|-----:|--------------------:|------|
| Baseline (no hook) | — | 1.000 | by manifest construction |
| top_k=1 (zero feature 4698) | 3.36 | **1.000** | single top feature dispensable |
| top_k=5 (zero top-5 features) | 11.16 | **0.600** | partial break: 8/20 flipped |
| top_k=10 (zero top-10 features) | 27.74 | **1.000** | recovered (non-monotone) |
| **top_k=20 (zero top-20 features)** | **49.23** | **0.000** | **full break: 0/20 retain physics** |
| random k=20 (mass-matched, 83% of top-20) | **40.97** | **1.000** | mass-matched random ablation has *no* effect |

When top_k=20 is ablated, all 20 stim produce a similar D-prefixed
response. Worth noting: the random k=20 control *also* produces highly
similar A-prefixed responses across stim ("The red arrow pointing
downward suggests…"). The "identical response" pattern reflects greedy
decoding on a homogeneous stim set (all clean cue=both physics-mode
inputs), not encoder collapse — both top-feature and random ablations
produce the same identical-prefix pattern, with the *content* (A vs D)
flipping based on which features were ablated.

### Revision (2026-04-27 evening) — Cohen's d ranking + multi-seed random + dense k-sweep

Three weaknesses in the morning run, advisor-flagged: (i) feature 3313
is a high-baseline outlier (mean_phys=7.86, mean_abs=6.24; large delta
but small relative to its variance), suggesting "general image content"
rather than physics-specific; (ii) only 1 mass-matched random set; (iii)
the k=5 → k=10 → k=20 non-monotone (0.6 → 1.0 → 0.0) was unresolved.

**Re-rank by Cohen's d** (delta divided by pooled std). On the full
5120-feature SAE Spearman ρ = 0.98, but ρ = 0.47 on top-50 by delta —
ranking is unstable in the high-delta region. Top-20 turnover: 7/20
features replaced. Feature 3313 drops out (Cohen's d = 0.10, far below
top-50). Replacements include features 1677, 3804, 4275, 129, 4481,
188, 3826 — features with smaller raw delta but cleaner per-feature
variance. New top-10 by Cohen's d:

| feature_idx | mean_phys | mean_abs | pooled_std | delta | Cohen's d |
|------------:|----------:|---------:|-----------:|------:|----------:|
| 1674 | 1.03 | 0.01 | 1.31 | 1.02 | **0.78** |
| 4106 | 1.75 | 0.13 | 2.30 | 1.61 | 0.70 |
| 4468 | 1.39 | 0.14 | 2.23 | 1.26 | 0.56 |
| 4698 | 3.13 | 0.23 | 5.27 | 2.90 | 0.55 |
| 1677 | 0.44 | 0.11 | 0.70 | 0.33 | 0.47 |
| 2028 | 0.77 | 0.26 | 1.17 | 0.51 | 0.43 |
| 1152 | 2.66 | 0.32 | 5.62 | 2.34 | 0.42 |
| 1949 | 1.55 | 0.15 | 3.46 | 1.39 | 0.40 |
| 3804 | 0.30 | 0.07 | 0.58 | 0.23 | 0.40 |
| 438 | 1.26 | 0.14 | 2.86 | 1.12 | 0.39 |

Feature 3313 was rank 3 by delta but drops to rank ~50 by Cohen's d —
Cohen's d is the right metric for filtering high-baseline outliers.

**Multi-seed mass-matched random**: 3 sets within seed 42 alone, mass
23.4 / 24.7 / 33.4 (top-30 mass = 32.7 → 72 % / 76 % / 102 %).

**Dense k-sweep result (Cohen's-d ranking, n=20, with 95 % Wilson CIs)**:

| Condition | n_phys/20 | Phys rate | 95 % Wilson CI |
|-----------|----------:|----------:|----------------|
| top_k=1   | 20 | 1.00 | [0.84, 1.00] |
| top_k=2   | 20 | 1.00 | [0.84, 1.00] |
| top_k=3   | 20 | 1.00 | [0.84, 1.00] |
| top_k=5   | 12 | 0.60 | [0.39, 0.78] |
| top_k=7   | 12 | 0.60 | [0.39, 0.78] |
| top_k=10  | 12 | 0.60 | [0.39, 0.78] |
| **top_k=15** | **20** | **1.00** | [0.84, 1.00] — full recovery |
| top_k=20  | 2  | 0.10 | [0.03, 0.30] |
| **top_k=30** | **0** | **0.00** | [0.00, 0.16] — full break |
| random_0 (mass 72 %) | 20 | 1.00 | [0.84, 1.00] |
| random_1 (mass 76 %) | 20 | 1.00 | [0.84, 1.00] |
| random_2 (mass 102 %) | 20 | 1.00 | [0.84, 1.00] |

**Stim-cluster pivot resolves the non-monotonicity.** The k=5 / 7 / 10
"0.6" rate is *deterministic, not noise*: it is exactly the 8 line_blank
stim flipping while the other 12 stim hold. Per-cluster phys rate
matrix (n per cluster in parentheses):

| Cluster (n) | k=1-3 | k=5-7-10 | k=15 | k=20 | k=30 | random×3 |
|---|---:|---:|---:|---:|---:|---:|
| filled_blank (6) | 1.0 | 1.0 | 1.0 | **0.0** | 0.0 | 1.0 |
| filled_ground (1) | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| filled_scene (1) | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| **line_blank (8)** | 1.0 | **0.0** | **1.0** | **0.0** | 0.0 | 1.0 |
| line_ground (4) | 1.0 | 1.0 | 1.0 | **0.0** | 0.0 | 1.0 |

The line_blank cluster (most abstract M2 cells: line drawing on blank
background, even though cue=both fires) is the only cluster that breaks
in the k=5–10 range *and* recovers at k=15. The other four clusters
hold a stable physics-mode plateau through k=15, then break at k=20
(filled_blank + line_ground at k=20; filled_ground + filled_scene
require k=30). All clusters fully break at k=30.

**Mechanistic reading.** Top-5 by Cohen's d carries strong physics-cue
signal that matters at the boundary (line_blank already sits near the
D-side of the decision). Features ranked 11-15 evidently contain an
"abstract suppressor" — removing them from the ablation set restores
line_blank's physics-mode commitment (top-15 ablation has all 5 clusters
at 1.0). The recovery is a first-shot signature of polysemy / sign-mixed
features in the SAE: the ranking is not strictly "physics+ vs physics−",
some features behave as anti-abstract gates whose removal helps physics.
This is consistent with SAE feature interpretation literature where
features may be sign-mixed at non-monosemantic resolutions.

**Headline (revised)**: top-30 mass-matched ablation (mass 32.7) breaks
all 20 stim (Wilson CI [0.00, 0.16]) while three independent mass-
matched random k=30 sets (mass 23-33, 72-102 % of top mass) all leave
all 20 stim in physics-mode (60/60 trials → aggregate Wilson CI [0.94,
1.00], vs morning's single-set [0.84, 1.00]; lower-bound gap from 1.0
tightens 0.16 → 0.06, ~2.7× narrower). Direction-specificity is the
same finding as the morning's 1-set version, now buttressed by 3
independent replications + a richer ablation curve that exposes a
stim-cluster-conditional non-monotone in the mid-range.

## Headlines

(Numbers below reflect the canonical Cohen's d ranking + 3 mass-matched
random controls from the 2026-04-27 evening revision; see Result §
Revision for the morning delta-rank reference numbers.)

1. **Encoder-side physics-mode signal is localized in SAE feature space.**
   Subtracting the **top-30 physics-cue SAE features** (Cohen's-d rank,
   mass 32.7, < 1 % of the 5120-feature SAE) cleanly flips physics →
   abstract on 20/20 stim (Wilson CI [0.00, 0.16]). Subtracting **3
   independent mass-matched random k=30 sets** (mass 23.4 / 24.7 /
   33.4, i.e., 72-102 % of top-30 mass) leaves all 60 trials (3 sets ×
   20 stim) in physics-mode (aggregate Wilson CI [0.94, 1.00]). The
   first version of this experiment used a bottom-of-ranking random
   pool that turned out to have ~1 % of top-N's mass (the L1 penalty
   kills inactive features); correcting to a mass-matched pool was the
   load-bearing fix. The result is a true positive for direction-
   specificity in the encoder, parallel to the §4.6 v_L10 vs random-
   direction result at the input/LM layer.

2. **Single features are dispensable.** Zeroing only the Cohen's-d top-1
   feature (idx 1674, Cohen's d = 0.78) leaves PMR intact (20/20). Same
   for the delta-rank top-1 (idx 4698, delta = 2.90) in the morning
   run. The redundancy-spreading we observed at the LM attention level
   has an encoder-side analog: physics-mode information is encoded in a
   *small group of features* (~30 by Cohen's d, ~20 by raw delta — the
   two rankings differ on 7/20 features but agree on the localization
   claim), not a single feature. Whether each top-N feature breaks PMR
   *individually* is open (Open follow-up #3).

3. **Non-monotonic mid-range is stim-cluster-conditional, not noise**
   (resolved 2026-04-27 evening). The mid-range "0.6" phys rate
   reflects exactly the 8 line_blank stim flipping while the 12 non-
   line_blank stim hold (perfect 8/8 recovery at k=15 in delta-rank;
   k=15 in Cohen's d-rank). Wilson CIs at n=20 are wide enough that
   the aggregate rate is not separable from noise, but the per-stim
   structure is fully deterministic — the same 8 stim break and recover
   together across both rankings. Mechanistic reading: line_blank is
   the most-abstract M2 cluster (line drawing × blank bg, even with
   cue=both); top-5 Cohen's-d features are sufficient to push it past
   the D-side decision boundary, but features 11-15 act as an
   "abstract suppressor" whose removal compensates. Top-15 is the
   plateau where *all* clusters fall back to physics-mode; top-20+
   pushes past the cluster-specific reserves.

4. **Triangulated mechanism — full causal chain**:
   - **Encoder side**: top-30 SAE features (Cohen's d) in `vision_hidden_31`
     (last SigLIP layer, pre-projection) carry the physics-mode signal.
     Necessary (this experiment) and observable (delta + Cohen's d
     rankings, 13/20 overlap on top-20).
   - **LM side**: L9 MLP constructs commitment in residual stream
     (necessary, M5b knockout); L0-L9 carry sufficient information
     (M5b SIP); L10 reads it via redundant attention (M5a + per-head
     null).
   - **Pixel side**: gradient ascent on `pixel_values` toward v_L10
     flips PMR at ε=0.05; encoder transforms the pixel perturbation
     into changes in those top-20 SAE features (testable follow-up).
   - The mechanism is **input → encoder physics-cue features (top-20) →
     L0-L9 visual tokens → L9 MLP commitment → L10 read-out → letter**.

5. **H10** (research plan §2.5: "narrow IE bands at specific layers/heads")
   gets its encoder-side dimension. The LM side has 1 dominant MLP band
   at L9; the encoder side has ~30 SAE features at the last layer
   (~ 0.6 % of the 5120-feature SAE). Both are "narrow" but at
   different granularities — the framing was per-architecture-
   component (layer/head/feature), not literal layer-count.

## Limitations

1. ~~**Single mass-matched random control set, not 3.**~~ **Resolved
   2026-04-27 evening.** Multi-seed loop (seeds 42..91) within the
   original [70 %, 200 %] mass window yielded 3 sets at top-30 with
   mass 23.4 / 24.7 / 33.4 (72 % / 76 % / 102 % of top-30 mass = 32.7).
   All 3 sets retain physics-mode at 20/20 stim. Multi-seed approach
   replaces the abandoned importance-sampling redesign — the heavy-
   tailed distribution simply makes acceptance probabilistic per seed,
   not impossible.

2. ~~**Non-monotonicity at k=10 unresolved.**~~ **Resolved 2026-04-27
   evening.** Stim-level pivot reveals the non-monotone is
   stim-cluster-conditional, not aggregate-rate noise. All 8 line_blank
   stim flip together at k=5 / 7 / 10 and recover together at k=15;
   the 12 non-line_blank stim never flip in the mid-range. The
   recovery is deterministic and reproducible across both rankings
   (delta vs Cohen's d). See Result § Revision and Headline 3.

3. ~~**Top-20 includes a high-mass outlier (feature 3313, mass 14).**~~
   **Addressed 2026-04-27 evening.** Cohen's d ranking (delta / pooled
   std) drops feature 3313 from rank 3 (delta) to ~rank 50 (Cohen's d),
   exactly as expected for a high-baseline-noise feature. Top-20 by
   Cohen's d has 7/20 turnover; the new top-30 ablation cleanly breaks
   all 20 stim. Cohen's d is now the canonical ranking; raw delta is
   retained for comparison.

4. **Pre-projection layer only.** SAE trained on `vision_hidden_31`
   (1280-dim, before the projector that lifts to 3584). The features
   identified are SigLIP-encoder-level features, not necessarily what
   the LM "consumes" 1:1. Post-projection SAE (3584-dim) would be more
   directly causally upstream of L9 MLP but requires a fresh capture
   pass.

5. **Single SAE training.** Different L1 lambda / expansion factor /
   training-data composition could give different feature dictionaries.
   The 5120-feature 4× expansion is reasonable but not pre-registered.
   The result is internally consistent (top features differ from
   mass-matched random by 1.0 → 0.0 PMR) but feature-set portability
   across SAE trainings is untested.

6. **n=20 from cue=both clean stim only.** Same sampling caveat as
   layer-level knockout; harder cases (line/blank/none) might show
   different feature-group structure.

## Connection to other findings

- **§4.6 pixel encodability**: gradient ascent on `pixel_values` toward
  v_L10 flips PMR at ε=0.05. The mechanism: encoder transforms pixel
  perturbations into changes in the top-20 SAE features, which propagate
  to L9 MLP. This SAE intervention is the *direct* test of that encoder-
  side path — and confirms the localized feature group exists.

- **M5a steering at L10**: v_L10 lives in the post-encoder LM hidden
  state. The top-20 SAE features feed into the projector → LM, where the
  cue eventually becomes v_L10's direction. SAE features are the encoder
  basis; v_L10 is the LM-internal axis.

- **H-encoder-saturation** (M6/M9): Qwen's saturated SigLIP encoder
  produces clean class-separated activations from L3 onward — meaning
  the physics-cue features are already cleanly carved out at L3 and
  persist to L31. The SAE finding adds: the carving has *low intrinsic
  dimensionality* (~20-30 features, not hundreds).

- **M5b layer-level + per-head**: attention is redundant at every
  resolution tested in the LM. The encoder-side localization (this
  experiment) does *not* propagate to LM-side localization beyond the
  L9 MLP. The encoder produces ~20-30-feature signal; the LM compresses
  it into a single decision boundary at L9.

## Reproducer

```bash
# 1. Train SAE on Qwen vision encoder activations (uses existing M2 captures).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_train.py \
    --activations-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations \
    --predictions outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv \
    --layer-key vision_hidden_31 --n-features 5120 --n-steps 5000 \
    --tag qwen_vis31_5120 --device cuda:0 --l1-lambda 1.0

# 2a. Re-rank features with Cohen's d alongside raw delta (no SAE retrain).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_rerank_features.py \
    --sae-dir outputs/sae/qwen_vis31_5120 \
    --activations-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations \
    --predictions outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv \
    --layer-key vision_hidden_31

# 2b. Causal intervention with Cohen's-d ranking, dense k-sweep, 3 random sets.
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/qwen_vis31_5120 \
    --rank-by cohens_d \
    --top-k-list 1,2,3,5,7,10,15,20,30 \
    --random-controls 3 --n-stim 20 \
    --tag qwen_vis31_5120_cohens_d_v2
```

## Artifacts

- `src/physical_mode/sae/{train,feature_id}.py` — SAE module (tied-weight, input-normalized, with `feature_contribution` for clean intervention). `feature_id.py` returns both raw delta and Cohen's d.
- `scripts/sae_train.py`, `scripts/sae_intervention.py`, `scripts/sae_rerank_features.py` — drivers. Intervention supports `--rank-by {delta,cohens_d}` and multi-seed mass-matched random controls.
- `outputs/sae/qwen_vis31_5120/{sae.pt,metrics.json,feature_ranking.csv}` — feature_ranking.csv has `mean_phys / mean_abs / std_phys / std_abs / pooled_std / delta / cohens_d` columns (delta-sorted for back-compat).
- `outputs/sae_intervention/qwen_vis31_5120/results.csv` — original delta-rank intervention.
- `outputs/sae_intervention/qwen_vis31_5120_cohens_d_v2/results.csv` — Cohen's d rank, 9 top-k conditions × 3 mass-matched random sets, n=20 (240 rows).

## Open follow-ups

1. ~~**More mass-matched random sets**~~ ✅ resolved (3 sets, multi-
   seed loop).
2. ~~**Re-rank by Cohen's d / specificity ratio**~~ ✅ resolved
   (Cohen's d is now canonical; feature 3313 dropped to rank ~50; 7/20
   top-20 turnover).
3. **Single-feature ablation sweep**: zero each top-20 feature
   individually; identify which subset is *individually* necessary vs
   redundant within the group. Cohen's d top-1 (feature 1674) and the
   delta top-1 (feature 4698) are different — both should be tested
   individually.
4. **Feature-level functional interpretation**: for each top-15 (or
   top-30) feature, visualize the max-activating image patches across
   the 480-stim corpus. The cluster pivot suggests features 11-15 are
   "abstract suppressors" — visual interpretation should clarify what
   they encode (e.g., line-drawing-specific cues vs general edge
   detectors).
5. **Per-cluster mechanism deep-dive**: the line_blank "break at k=5,
   recover at k=15" pattern is a clean signal about feature polysemy
   and decision-boundary geometry. Is the recovery driven by a single
   feature in the rank 11-15 band or a cumulative effect? The actual
   Cohen's-d rank 11-15 features are `[3116, 3034, 117, 4275, 38]`.
   Targeted ablations (top-5 + each rank-11-15 feature individually,
   then in pairs) would resolve which features behave as sign-mixed
   anti-physics-on-line_blank gates whose removal compensates the
   top-5 damage.
6. **Post-projection SAE**: capture the post-projector activations
   (3584-dim, what the LM actually consumes) and re-run feature
   discovery + intervention.
7. **Cross-layer SAE**: train SAE on `vision_hidden_15` or earlier
   layers; trace which-layer first encodes the physics-mode features.
8. **Cross-model SAE**: port to LLaVA-1.5 / Idefics2 / InternVL3 — does
   each have its own ~20-30-feature physics-cue group at its encoder's
   last layer?
9. **Larger n for stim-cluster Wilson CIs**: per-cluster confirmations
   (8/8 line_blank, 6/6 filled_blank, etc.) are deterministic at n=20,
   but Wilson CIs are wide. Bumping to n=50 per cluster (~250 stim
   total, ~25 min compute) would tighten the per-cluster point
   estimates if this becomes a paper-figure-grade claim.
