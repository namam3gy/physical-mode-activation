# §4.6 — VTI-reverse counterfactual stim, Qwen2.5-VL (run log, 2026-04-26)

## Setup

- **Module**: `src/physical_mode/synthesis/counterfactual.py` (gradient_ascent + reconstruct_pil with inverse permute matching Qwen2VLImageProcessor's forward `(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)`).
- **Model**: Qwen2.5-VL-7B.
- **Approach**: pixel-space gradient ascent on post-processor `pixel_values` (T_patches × 1176 normalized representation), maximizing `<mean(h_L10[visual]), v_L10>`. Bypasses non-differentiable PIL → patch preprocessing while still recovering a viewable RGB via inverse permute + de-norm.
- **Sweep**: 5 baseline circle stim × 7 configs × 200 Adam steps (lr=1e-2) = **35 runs in ~30 min on H200**.

## Commands

```bash
uv run python scripts/sec4_6_counterfactual_stim.py
# → outputs/sec4_6_counterfactual_<ts>/{<config>/<sid>/synthesized.png, trajectory.npy, pixel_values.pt}, manifest.json

uv run python scripts/sec4_6_summarize.py --run-dir outputs/sec4_6_counterfactual_<ts>
# → outputs/.../results.csv, results_aggregated.csv
#   docs/figures/sec4_6_counterfactual_stim_panels.png
#   docs/figures/sec4_6_counterfactual_stim_trajectory.png
```

## Output dir

`outputs/sec4_6_counterfactual_20260426-050343/` (the canonical run, referenced from paper draft).

## Result table

| Config | n flipped (PMR 0→1) | Mean final projection |
|---------------------|--------------------:|----------------------:|
| `bounded_eps0.05` | 5 / 5 | 43.7 |
| `bounded_eps0.1` | 5 / 5 | 100.6 |
| `bounded_eps0.2` | 5 / 5 | 125.9 |
| `unconstrained` | 5 / 5 | 181.1 |
| `control_v_random_*` | 0 / 15 | 73–85 |

## Headlines

1. **5/5 v_L10 flips at ε = 0.05** (pre-registered ≥ 3/5).
2. **0/15 random-direction flips at matched ε = 0.1.** Random reaches comparable projection magnitudes (73-85 vs bounded ε=0.1 at 101) but PMR doesn't flip — directional specificity, not magnitude, controls the regime change.
3. **`v_L10` is encodable in the image** — pixel-space change without runtime steering suffices.

## Scorer fix

Random controls exposed an over-permissive PMR scorer that matched the "mov" stem inside an abstract sentence ("no indication of movement"). Added asymmetric abstract-marker patterns (`remain stationary`, `no indication of mov`, `no indication of motion`). Verified asymmetric: 0/20 v_L10 hits new abstract markers; 14/15 random hits. Headline replicates with the pre-fix scorer.

## Tests

`uv run python -m pytest tests/test_counterfactual.py tests/test_pmr_scoring.py -v` (PMR test suite extended 51 → 54 cases).

## Deep dive

`docs/insights/sec4_6_counterfactual_stim.md`.
