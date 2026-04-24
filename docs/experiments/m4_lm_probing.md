# M4 — LM probing (logit lens + per-layer probes) (2026-04-24)

- **Command**: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`
- **Input**: M2-captured LM hidden states at layers (5, 10, 15, 20, 25), 480 stimuli
- **Wall clock**: ~6 min (logit-lens 480 × 5 layers = 2400 projections + model load)
- **Outputs**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/`
- **Deep dive**: `docs/insights/m4_logit_lens.md`.

## Headline findings

**LM per-layer probe AUC (target: forced-choice PMR)**:

| layer | AUC (mean ± std) |
|---|---|
| 5 | 0.939 ± 0.015 |
| 10 | 0.944 ± 0.006 |
| 15 | 0.947 ± 0.009 |
| **20** | **0.953 ± 0.007** (peak) |
| 25 | 0.944 ± 0.009 |

→ All captured LM layers discriminate PMR with AUC ~0.94-0.95. The peak at layer 20 aligns with Neo et al. 2024's LLaVA claim that object features crystallize in mid-to-late LM layers. **Information survives the LM almost losslessly**; the ~29 pp gap to behavioral forced-choice accuracy (~0.66) must come from the discrete generation step.

**Logit-lens trajectory (mean logit by category)**:

| layer | geometry | physics | label |
|---|---|---|---|
| 5 | 0.93 | 1.04 | 1.16 |
| 10 | 1.35 | 1.66 | 1.73 |
| 15 | 2.04 | 2.45 | 2.29 |
| 20 | 3.23 | 4.18 | 4.09 |
| 25 | 11.56 | **15.64** | 13.96 |

→ Physics > geometry from layer 5 onward — the "ball" label in the prompt primes the LM *before* any residual updates. Final-layer amplification (L20 → L25) boosts the physics margin from 0.9 to 4.0.

**Physics margin by object_level** (phys − geom logit):

| layer | line | filled | shaded | textured |
|---|---|---|---|---|
| 5 | 0.09 | 0.08 | 0.12 | 0.15 |
| 20 | 0.87 | 0.89 | 0.97 | 1.05 |
| 25 | 3.76 | 3.94 | 4.29 | 4.35 |

Object-induced margin: +0.6 at L25 (line → textured). **Label-induced margin: +4.0 (flat bias across all stimuli)**. Label ≈ 7× the object effect inside the LM. Consistent with H7/H4.

**Switching layer** is trivially 5 for all 480 samples when using max-logit-per-category — physics is already ahead at the earliest captured layer because of the "ball" label. This metric is uninformative for label-primed prompts; revisit with a label-free prompt variant.

## Hypothesis update

| H | prior | post-M4 | change |
|---|---|---|---|
| H-boomerang | supported (M3) | **extended** | Information survives the entire LM — the gating happens in decoding |
| H7 (label → regime) | candidate (M2) | **supported** | The label prior shifts the physics margin from L5 onward |
| H-locus (new) | — | **candidate** | Bottleneck is at the LM final layers + decoding head. ST4 patching target. |

## Unlocks

- **Sub-task 4 (M5) target narrowed**: LM layers 20-27 residual stream + decoding head are the intervention priority. Re-capture with `capture_lm_attentions=True` makes SIP patching immediately runnable.
- **§4 ideas update**: 4.9 "label-free prompt" is the direct test that would re-validate the switching-layer metric — priority increased.
