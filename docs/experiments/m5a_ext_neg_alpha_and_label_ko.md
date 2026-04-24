# M5a 확장 실험 — 실행 로그

M5a 의 두 후속 실험으로, 2026-04-24 에 동일한
`outputs/mvp_full_20260424-094103_8ae1fa3d` run directory 에서 실행.

Design spec: `docs/superpowers/specs/2026-04-24-m5a-extensions-design.md`.
상위 milestone: `docs/insights/m5_vti_steering_ko.md`.

## 실험 1 — physics-mode baseline 에서 negative α

**질문**: physics-mode baseline 에 `-α · v_L10` 을 주입하면 "abstract" 방향으로 flip 되는가?

**설정**:
- Test subset: `textured/ground/both`, 10 seed, event=`fall`.
- Label: `circle`.
- Steering layer: 10.
- α: `0, -5, -10, -20, -40`.
- Prompt variant: `forced_choice`, T=0.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/`.
- Wall-clock: ≈58 s (model load + 50 inference).

### PMR — (layer, α) 별

| layer | α=0 | α=-5 | α=-10 | α=-20 | α=-40 |
|---|---|---|---|---|---|
| 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### First-letter 분포 — (layer, α) 별

| layer | α | A | other |
|---|---|---|---|
| 10 | 0.0 | 9 | 1 |
| 10 | -5.0 | 10 | 0 |
| 10 | -10.0 | 10 | 0 |
| 10 | -20.0 | 10 | 0 |
| 10 | -40.0 | 10 | 0 |

### 대표 응답

| α | first | pmr | raw_text (200자 truncation) |
|---|---|---|---|
| -40 | A | 1 | "A  The image depicts a ball falling towards a flat surface, which is a common representation of an object in motion under gravity. The ball's trajectory suggests it is moving downward..." |
| -20 | A | 1 | "A  Justification: The image depicts a ball above a line with a downward arrow pointing towards it, suggesting motion downwards. Therefore, the most logical conclusion is that the ball will fall down." |
| -10 | A | 1 | "A  The image depicts a ball with a downward arrow pointing towards a surface, suggesting that the ball will fall down if it is released from its current position." |
| -5 | A | 1 | "A  The image depicts a ball with a downward arrow pointing towards a surface, indicating that the ball will fall down due to gravity." |
| 0 | A | 1 | "A  The image depicts a ball falling towards a surface, indicated by the downward arrow. Therefore, the most logical conclusion is that the ball will fall down to the surface." |

## 실험 2 — abstract baseline 에서 label=ball

**질문**: label `circle` → `ball` 로 교체하면 L10 α=40 flip 목적지가 B ("stays still") 에서 A ("falls") 로 이동하는가?

**설정**:
- Test subset: `line/blank/none`, 10 seed, event=`fall`.
- Label: `ball`.
- Steering layer: 10.
- α: `0, 5, 10, 20, 40`.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/`.
- Wall-clock: ≈44 s.

### PMR — (layer, α) 별

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

### First-letter 분포 — (layer, α) 별

| layer | α | A | D |
|---|---|---|---|
| 10 | 0.0 | 0 | 10 |
| 10 | 5.0 | 0 | 10 |
| 10 | 10.0 | 0 | 10 |
| 10 | 20.0 | 0 | 10 |
| 10 | 40.0 | 10 | 0 |

### 대표 응답

| α | first | pmr | raw_text (200자 truncation) |
|---|---|---|---|
| 0 | D | 0 | "D  The image is an abstract representation of a circle and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical moti..." |
| 5 | D | 0 | "D  The image is an abstract representation of a circle and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical moti..." |
| 10 | D | 0 | "D  The image is an abstract representation and does not depict a physical object like a ball that can fall, move, or stay still. Therefore, it cannot be described in terms of physical motion." |
| 20 | D | 0 | "D  The image depicts an abstract shape that does not represent a physical object like a ball. Therefore, it cannot be described in terms of physical motion such as falling, staying still, or moving si..." |
| 40 | A | 1 | "A) It falls down.  Justification: The ball is subject to gravity, which causes it to fall downward. This is a fundamental principle of physics and can be observed in everyday life, such as when a ball..." |

## M5a 와의 교차 검증

- M5a `line/blank/none × circle × L10 α=40`: **10/10 → B** (`docs/insights/m5_vti_steering_ko.md` §3.2 참조).
- M5a baseline `steering_experiments/intervention_predictions.parquet` mtime 확인: `2026-04-24 11:48:24` — 두 실험 모두 각자의 subdir 에 기록했으므로 재작성 없음.

## 원본 artifacts

- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` — Exp 1 parquet, CSV, run_meta.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` — Exp 2.

## 부수적 관찰

- 실험 1 α=0 baseline 은 9/10 A + 1/10 other (완전 포화 아님). 모든 negative α → 10/10 A. 따라서 -α 는 physics-mode 를 억제하지 않으며, 오히려 "other" edge case 를 A 로 수렴시킴.
- 실험 2 α=0 baseline 은 10/10 D — 이는 M2 의 `ball+line+blank+none` PMR≈0.85 와 다름. Likely caused by the forced-choice prompt template in the steering script differing from M2's inference prompts. (M5a 의 original circle+line+blank+none baseline 에서도 동일 패턴 관찰: `docs/insights/m5_vti_steering_ko.md` §3.2 기준 10/10 D.) 향후 reconciliation 필요로 기록.
