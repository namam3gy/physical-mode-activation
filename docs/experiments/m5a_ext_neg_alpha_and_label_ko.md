# M5a 확장 실험 — 실행 로그

M5a 의 후속 실험으로, 동일한
`outputs/mvp_full_20260424-094103_8ae1fa3d` run directory 에서 실행.

- 실험 1 & 2: 2026-04-24 (ceiling 에서의 bidirectionality 검정 + label swap).
- 실험 3a-3d: 2026-04-25 (moderate baseline 에서 bidirectionality 재검정 —
  실험 1 의 ceiling confound 해결).

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

## 실험 3 — moderate baseline 에서 bidirectionality 재검정

**동기**: 실험 1 은 `textured/ground/both` 를 사용했는데 α=0 baseline 이 이미
9/10 A 로 PMR ceiling 이므로 `-α` 에 의한 physics-mode 억제 효과를 관찰할 여지가
없음. 실험 3 은 `textured/blank/none` (α=0 baseline 이 거의 PMR floor 에 가까울
것으로 예상) 로 이동하고 ±α 전체 구간을 sweep — 두 sign 방향 모두 측정 여유가
있는 조건.

**공통 설정** (4개 sub-run 공통):
- 10 seed/cell, event=`fall`.
- Steering layer: 10.
- α: `-40, -20, -10, -5, 0, 5, 10, 20, 40` (3a, 3b); `-40, -20, -10, -5, 0` (3c, 3d).
- Prompt variant: `forced_choice`, T=0.
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/bidirectional_recheck_<slug>/`.

### 실험 3a — `textured/blank/none × label=ball`

First-letter 분포 — (layer, α) 별:

| α | A | B | D |
|---|---|---|---|
| -40 | 0 | **10** | 0 |
| -20 | 0 | 2 | 8 |
| -10 | 0 | 0 | 10 |
| -5  | 0 | 1 | 9 |
| 0   | 0 | 1 | 9 |
| 5   | 0 | 2 | 8 |
| 10  | 7 | 1 | 2 |
| 20  | **10** | 0 | 0 |
| 40  | **10** | 0 | 0 |

Baseline PMR (α=0): 0.1. Output subdir: `bidirectional_recheck_textured_blank_none_ball/`.

### 실험 3b — `textured/blank/none × label=circle`

| α | A | B | D |
|---|---|---|---|
| -40 | 0 | **10** | 0 |
| -20 | 0 | 0 | 10 |
| -10 | 0 | 0 | 10 |
| -5  | 0 | 0 | 10 |
| 0   | 0 | 0 | 10 |
| 5   | 0 | 0 | 10 |
| 10  | 0 | 0 | 10 |
| 20  | 2 | 0 | 8 |
| 40  | **10** | 0 | 0 |

Baseline PMR (α=0): 0.0. Output subdir: `bidirectional_recheck_textured_blank_none_circle/`.

### 실험 3c — `line/blank/none × label=ball` (negative side 만)

Positive side 는 실험 2 에 있음. Negative sweep:

| α | A | B | D |
|---|---|---|---|
| -40 | 0 | **10** | 0 |
| -20 | 0 | 0 | 10 |
| -10 | 0 | 0 | 10 |
| -5  | 0 | 0 | 10 |
| 0   | 0 | 0 | 10 |

Output subdir: `bidirectional_recheck_line_blank_none_ball/`.

### 실험 3d — `line/blank/none × label=circle` (negative side 만)

Positive side 는 M5a 에 있음 (`line/blank/none × circle × L10 α=40 → 10/10 B`). Negative sweep:

| α | A | B | D |
|---|---|---|---|
| -40 | 0 | **10** | 0 |
| -20 | 0 | 0 | 10 |
| -10 | 0 | 0 | 10 |
| -5  | 0 | 0 | 10 |
| 0   | 0 | 0 | 10 |

Output subdir: `bidirectional_recheck_line_blank_none_circle/`.

### 교차 요약 — |α|=40 (L10, T=0)

| obj × label | α=-40 | α=0 | α=+40 | 출처 |
|---|---|---|---|---|
| line × circle     | 10 B | 10 D | 10 B | 실험 3d + M5a |
| line × ball       | 10 B | 10 D | 10 A | 실험 3c + 실험 2 |
| textured × circle | 10 B | 10 D | 10 A | 실험 3b |
| textured × ball   | 10 B |  9 D + 1 B | 10 A | 실험 3a |

패턴:
- `-α=40` → 4개 (obj × label) 조합 모두 **B** 로 균일하게 flip.
- `+α=40` → image 또는 label 중 하나가 physical signal 을 지닐 때 (textured 또는
  ball) **A**; image 와 label 이 모두 abstract 일 때만 (line + circle) **B**.
- `α=0` → 4개 조건 모두 **D** (baseline forced-choice 는 모든 context 에서
  physics 해석을 거부).

## M5a 와의 교차 검증

- M5a `line/blank/none × circle × L10 α=40`: **10/10 → B** (`docs/insights/m5_vti_steering_ko.md` §3.2 참조).
- M5a baseline `steering_experiments/intervention_predictions.parquet` mtime 확인: `2026-04-24 11:48:24` — 모든 실험이 각자의 subdir 에 기록했으므로 재작성 없음.

## 원본 artifacts

- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/neg_alpha_textured_ground_both/` — 실험 1.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/ball_line_blank_none/` — 실험 2.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/bidirectional_recheck_textured_blank_none_ball/` — 실험 3a.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/bidirectional_recheck_textured_blank_none_circle/` — 실험 3b.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/bidirectional_recheck_line_blank_none_ball/` — 실험 3c.
- `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/bidirectional_recheck_line_blank_none_circle/` — 실험 3d.

## 부수적 관찰

- 실험 1 α=0 baseline 은 9/10 A + 1/10 other (완전 포화 아님). 모든 negative α → 10/10 A. 실험 1 의 flat 응답은 ceiling effect 로 확정 — inherent asymmetry 가 아님 (실험 3 에서 baseline 이 ceiling 을 벗어나면 `-α` 의 효과가 분명히 관찰됨).
- 실험 2 α=0 baseline 은 10/10 D — 이는 M2 의 `ball+line+blank+none` PMR≈0.85 와 다름. Steering script 의 forced-choice prompt template 이 M2 의 inference prompt 와 달라서일 가능성이 큼. (M5a 의 original circle+line+blank+none baseline 에서도 동일 패턴 관찰: `docs/insights/m5_vti_steering_ko.md` §3.2 기준 10/10 D.) 향후 reconciliation 필요로 기록.
- 실험 3 `textured/blank/none × circle` 의 α=+20 은 transition zone 표시 (2 A + 8 D) — |α|=10 (무효) 와 |α|=40 (full flip) 사이의 threshold 는 |α|∈[15, 25] 구간. 더 촘촘한 α sweep 은 deferred.
