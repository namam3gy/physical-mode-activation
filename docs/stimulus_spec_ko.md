# 자극 스펙 — PhysCue (Sub-task 1)

Factorial 축 + 렌더링 선택지의 canonical 정의. 이 파일을 `src/physical_mode/stimuli/`
와 `src/physical_mode/config.py` 와 동기화 유지.

## 캔버스

- 512 × 512 px, RGB, 흰 배경.
- `bg_level ∈ {ground, scene}` 일 때 지면선은 y = 0.78 × canvas = 399 px.
- 객체 기본 반지름: 64 px (지름 128 px).

## 축 A — 객체 추상화

| 수준 | 렌더링 | Cue claim (`references/project.md` §1.3) |
|---|---|---|
| `line` | 3-px 검은 외곽선, 흰 fill | 순수 기하학; 가장 약한 physical cue |
| `filled` | 2-px 외곽선, 균일 회색 fill | 약간의 mass cue; 여전히 기하학적 |
| `shaded` | 좌상단 광원 radial gradient (Ramachandran 1988 light-from-above prior) | 가장 강한 *static* 3D cue |
| `textured` | shaded sphere + 자오선 호 + 7 어두운 점 (축구공 cue) | 프로그램적으로 렌더 가능한 최고 수준 |
| `block_stack` | jitter 적용된 사각형 3 개 stacked | "ball 의미 없는 명백한 physical" 자극 — 원-공 축의 대안 |

`block_stack` 수준은 의도적으로 pilot / MVP-full config 에 미포함 — "물리 객체
≠ ball" 축을 다루는 후속 라운드용으로 예약.

## 축 B — 배경

| 수준 | 렌더링 |
|---|---|
| `blank` | 추가 없음 |
| `ground` | y=399 에 3-px 검은 수평선 한 줄 |
| `scene` | 지면선 + 하늘/지면 음영 + 우측에 작은 ramp obstacle (Gibson 식 surface + optic-flow 호환 support plane) |

## 축 C — 맥락 cue

| 수준 | 렌더링 |
|---|---|
| `none` | 추가 없음 |
| `cast_shadow` | 지면 위 타원 cast shadow (Kersten et al. 1997 ground-attachment cue); bg = blank 라도 렌더 |
| `motion_arrow` | 빨간 방향 화살표; heading 은 event template 에 따라 결정 |
| `both` | cast_shadow + motion_arrow |
| `wind` (legacy) | 객체 한 쪽에 anchor 된 짧은 회색 호 5 cluster — pilot 에서 Qwen2.5-VL 에 invisible 로 확인됨, 재현성용으로 유지 |
| `arrow_shadow` (legacy) | pilot 의 통합 cue; `both` 와 동등 |

## 축 D — 객체 라벨 (프롬프트 시점)

| 수준 | 프롬프트 문구 |
|---|---|
| `circle` | "The image shows a circle. …" |
| `ball` | "The image shows a ball. …" |
| `planet` | "The image shows a planet. …" |
| `shape` | "The image shows a shape. …" |
| `object` | "The image shows an object. …" |

언어 prior 단독으로 physics-mode 를 강제할 수 있는지 통제 (research H2). Pilot
은 단일 라벨 (`ball`); MVP-full 은 `(circle, ball, planet)` 사용.

## Event template

| Event | 객체 위치 | 예상 physical answer |
|---|---|---|
| `fall` | center-top (cx=256, cy=128) | 지면으로 떨어짐 |
| `horizontal` | mid-left (cx=179, cy=230) | 옆으로 이동 (특히 wind cue 와) |
| `hover` | center-upper (cx=256, cy=205) | 공중 정지 (over-attribution 테스트) |
| `wall_bounce` | right-mid (cx=358, cy=230) | 튕겨 나옴 |
| `roll_slope` | left-lower (cx=128, cy=307) | ramp 내려감 |

Pilot 은 `fall` + `horizontal`; MVP-full 은 `fall` 만 (pilot 에서 두 event 의
behavior 동등 확인). 나머지 셋은 후속 라운드용으로 scaffolded.

## Seed discipline

`FactorialSpec.base_seed` (기본 1000) 가 시작 값; 각 (object, bg, cue, event,
variant_index) cell 이 monotonically 증가하는 counter 에서 고유 seed 받음.
동일 factorial spec 은 **항상** 동일 stimuli 를 produce — `tests/test_stimuli_deterministic.py`
에서 검증.
