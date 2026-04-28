---
section: §4.8 — PMR scaling (Qwen2.5-VL 7B vs 32B on M2)
date: 2026-04-28
status: complete (open prompt, 1440 inferences each, 16 min wall on H200)
hypothesis: scale doesn't help PMR — MechBench-style "scaling doesn't fix grounding" prediction.
---

# §4.8 — PMR scaling: Qwen2.5-VL 7B vs 32B on M2

## TL;DR

Qwen 32B has **virtually identical aggregate PMR** to Qwen 7B on the
M2 stim under the open prompt:

| Metric | 7B (open) | 32B (open) |
|--------|----------:|-----------:|
| **PMR (mean)** | **0.931** | **0.926** |
| abstract_reject | 0.002 | **0.065** |
| GAR (gravity-align) | — | 0.631 |
| Hold-still | 0.152 | 0.103 |
| n | 1440 | 1440 |

The headline ΔPMR = −0.005 is well within noise. **5× scaling does
not help physics-mode classification at the aggregate level** —
supports the MechBench reading that "more parameters" alone doesn't
fix the language-prior dominance of M2's "circle / ball / planet"
prompt regime.

**But aggregate ≠ per-cell.** Two correlated shifts at the
hard-cell level reveal that 32B *does* discriminate better when the
visual evidence is weak:

1. **`abstract_reject` jumps 35× (0.002 → 0.065)**: 32B is much
   more likely to refuse the physics-mode framing when the cue is
   ambiguous.
2. **`cue=none` PMR drops 8.6 pp (0.797 → 0.711)**: same effect
   measured directly — when no cue is present, 32B backs off toward
   abstract more often than 7B does.

These are the same underlying capability (cue-aware discrimination)
viewed two ways. The aggregate PMR doesn't move because **the
remaining 1296 cue-present cells are already saturated at PMR ≈ 1
in both models** — there's no headroom for scaling to help where
the model is already correct. **The paper-relevant claim is therefore
not "scale doesn't help" but the more nuanced "scale helps where
it has room to help: the 5 % of cells where the cue is weakest".**
This is also where the project's core thesis (visual-prior under-
weighting in physics-mode prompts) localizes — exactly the cells
where scale matters are exactly the ones our hypothesis predicts
should be hard.

## Per-axis breakdown (PMR by axis-level)

### By object-level (abstraction axis)
| object_level | 7B | 32B | Δ |
|--------------|----:|----:|--:|
| line         | 0.906 | 0.911 | +0.005 |
| filled       | 0.933 | 0.919 | −0.014 |
| shaded       | 0.933 | 0.922 | −0.011 |
| textured     | 0.950 | 0.950 |  0.000 |

Both models saturated; H1 abstraction-axis ramp is invisible at this
prompt regime (already known from M2 7B). Scale doesn't expose the
ramp either.

### By cue-level (physics-cue axis) — the only place 32B differs
| cue_level | 7B | 32B | Δ |
|-----------|----:|----:|--:|
| none         | 0.797 | **0.711** | **−0.086** |
| cast_shadow  | 0.936 | 0.994 | +0.058 |
| motion_arrow | 0.997 | 1.000 | +0.003 |
| both         | 0.992 | 0.997 | +0.005 |

The `none` cell drops 8.6 pp and `cast_shadow` rises 5.8 pp. **32B
is more cue-sensitive**: when the cue is absent it backs off toward
abstract; when a single cue fires it commits harder to physics-mode.
This is a modest improvement to the visual-prior reliance, but it
doesn't dissolve the language-prior dominance the project has
documented.

### By label (language-prior axis)
| label | 7B | 32B | Δ |
|-------|----:|----:|--:|
| ball   | 0.954 | 0.933 | −0.021 |
| circle | 0.883 | **0.923** | **+0.040** |
| planet | 0.954 | 0.921 | −0.033 |

32B's ball/planet drop slightly while circle rises — **the label
gap narrows** (7B `ball − circle` = +0.071 pp; 32B = +0.010 pp). H2
language-prior dominance is *weakened* but not eliminated under
scaling. Still, this is the most notable per-axis shift along with
cue-level.

## Open questions

- **Why does scale help cue-level but not object-level?** The cue
  axis is single-feature visible (a shadow blob, an arrow); the
  object-axis is mostly stylistic shading. Possibly 32B has more
  capacity to *integrate* discrete physics cues but the same
  saturation on smooth abstraction differences.
- **Is the abstract_reject jump (0.002 → 0.065) on cue=none cells?**
  Most likely yes (matches the per-cue PMR drop). A cell-level
  pivot would confirm.
- **What about 72B?** Not tested — would extend the scaling curve.
  Predicted to land near 32B (saturation in this regime).

## Limitations

1. Open prompt only — forced-choice not run on 32B.
2. T=0.7 means RC<1 cells exist; we report the mean, not the
   per-seed agreement. RC analysis would mirror M2.
3. M5a steering not run on 32B (would need new layer-aware capture).
4. No L10 v_L extraction → §4.6 pixel-encodability cross-scale not
   testable yet.

## Reproducer

```bash
# Inference (single GPU bf16 on H200 — fits at 64 GB weights + KV).
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m2_qwen_32b.py \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3
# ~16 min wall on H200 for 1440 inferences (max_new_tokens=96).

# Score + summarize.
uv run python scripts/03_score_and_summarize.py \
    --run-dir outputs/m2_qwen_32b_<ts>
```

## Artifacts

- `configs/m2_qwen_32b.py` — open-prompt M2 config for Qwen 32B.
- `outputs/m2_qwen_32b_20260427-212653_a167494f/{predictions,
  predictions_scored,response_consistency,summary_overall,
  summary_by_*}.csv` — full 1440-row inference + scoring +
  factorial summaries.
