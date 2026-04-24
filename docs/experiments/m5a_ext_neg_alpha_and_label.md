# M5a Extensions — Run Log

Two follow-up experiments to M5a, executed 2026-04-24 on the same
`outputs/mvp_full_20260424-094103_8ae1fa3d` run directory.

Design spec: `docs/superpowers/specs/2026-04-24-m5a-extensions-design.md`.
Parent milestone: `docs/insights/m5_vti_steering.md`.

## Experiment 1 — Negative α on physics-mode baseline

**Question**: does injecting `-α · v_L10` at a physics-mode baseline flip it toward "abstract"?

**Setup**:
- Test subset: `textured/ground/both`, 10 seeds, event=`fall`.
- Label: `circle`.
- Steering layer: 10.
- α: `0, -5, -10, -20, -40`.
- Prompt variant: `forced_choice`, T=0.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/`.
- Wall-clock: ≈58 s (model load + 50 inferences).

### PMR by (layer, α)

| layer | α=0 | α=-5 | α=-10 | α=-20 | α=-40 |
|---|---|---|---|---|---|
| 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### First-letter distribution by (layer, α)

| layer | α | A | other |
|---|---|---|---|
| 10 | 0.0 | 9 | 1 |
| 10 | -5.0 | 10 | 0 |
| 10 | -10.0 | 10 | 0 |
| 10 | -20.0 | 10 | 0 |
| 10 | -40.0 | 10 | 0 |

### Representative responses

| α | first | pmr | raw_text (truncated to 200 chars) |
|---|---|---|---|
| -40 | A | 1 | "A  The image depicts a ball falling towards a flat surface, which is a common representation of an object in motion under gravity. The ball's trajectory suggests it is moving downward..." |
| -20 | A | 1 | "A  Justification: The image depicts a ball above a line with a downward arrow pointing towards it, suggesting motion downwards. Therefore, the most logical conclusion is that the ball will fall down." |
| -10 | A | 1 | "A  The image depicts a ball with a downward arrow pointing towards a surface, suggesting that the ball will fall down if it is released from its current position." |
| -5 | A | 1 | "A  The image depicts a ball with a downward arrow pointing towards a surface, indicating that the ball will fall down due to gravity." |
| 0 | A | 1 | "A  The image depicts a ball falling towards a surface, indicated by the downward arrow. Therefore, the most logical conclusion is that the ball will fall down to the surface." |

## Experiment 2 — Label=ball on abstract baseline

**Question**: does swapping label `circle` → `ball` shift the L10 α=40 flip target from B ("stays still") to A ("falls")?

**Setup**:
- Test subset: `line/blank/none`, 10 seeds, event=`fall`.
- Label: `ball`.
- Steering layer: 10.
- α: `0, 5, 10, 20, 40`.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/`.
- Wall-clock: ≈44 s.

### PMR by (layer, α)

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

### First-letter distribution by (layer, α)

| layer | α | A | D |
|---|---|---|---|
| 10 | 0.0 | 0 | 10 |
| 10 | 5.0 | 0 | 10 |
| 10 | 10.0 | 0 | 10 |
| 10 | 20.0 | 0 | 10 |
| 10 | 40.0 | 10 | 0 |

### Representative responses

| α | first | pmr | raw_text (truncated to 200 chars) |
|---|---|---|---|
| 0 | D | 0 | "D  The image is an abstract representation of a circle and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical moti..." |
| 5 | D | 0 | "D  The image is an abstract representation of a circle and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical moti..." |
| 10 | D | 0 | "D  The image is an abstract representation and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical motion." |
| 20 | D | 0 | "D  The image depicts an abstract shape that does not represent a physical object like a ball. Therefore, it cannot be described in terms of physical motion such as falling, staying still, or moving si..." |
| 40 | A | 1 | "A) It falls down.  Justification: The ball is subject to gravity, which causes it to fall downward. This is a fundamental principle of physics and can be observed in everyday life, such as when a ball..." |

## Cross-check vs M5a

- M5a `line/blank/none × circle × L10 α=40`: **10/10 → B** (from `docs/insights/m5_vti_steering.md` §3.2).
- M5a baseline `steering_experiments/intervention_predictions.parquet` mtime verified as `2026-04-24 11:48:24` — not rewritten by these runs.

## Raw artifacts

- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` — Exp 1 parquet, CSVs, run_meta.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` — Exp 2.

## Subsidiary observations

- Experiment 1 α=0 baseline has 9/10 A + 1/10 other (not fully saturated). All negative α → 10/10 A. So -α does not suppress physics-mode; if anything, it concentrates A even at the "other" edge case.
- Experiment 2 α=0 baseline is 10/10 D — this differs from M2's PMR≈0.85 for `ball+line+blank+none`. Likely caused by the forced-choice prompt template in the steering script differing from M2's inference prompts. (Same pattern was observed in M5a's original circle+line+blank+none baseline: 10/10 D per `docs/insights/m5_vti_steering.md` §3.2.) Noted for future reconciliation.
