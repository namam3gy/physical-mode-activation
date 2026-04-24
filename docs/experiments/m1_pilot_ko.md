# M1 — Pilot run (2026-04-24)

- **명령**: `uv run python scripts/02_run_inference.py --config configs/pilot.py`
- **자극 dir**: `inputs/pilot_20260424-072216_308c86fc` (240 stimuli)
- **출력 dir**: `outputs/pilot_20260424-072418_2c16efb6`
- **모델**: `Qwen/Qwen2.5-VL-7B-Instruct` (첫 run 다운로드, bf16, sdpa on H200)
- **Wall clock**: 약 8 분 (첫-run HF 다운로드 ~15 초, 729-shard weight load ~8 초, 480 inference @ ~1.0 it/s)
- **N predictions**: 480 (240 stimuli × 1 label "ball" × 2 prompt variant)
- PMR 두 번 채점: 초기 lexicon 한 번, `move` → `mov` family 패치 후 한 번. 아래
  수치는 최종 `predictions_scored.parquet` 기준.
- **심층**: `docs/insights/m1_pilot_ko.md`.

## 핵심 PMR / GAR

**object_level 별** (axis A — 추상화):

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 120 | 0.575 | 0.333 | 0.325 | 0.667 |
| filled | 120 | 0.658 | 0.333 | 0.225 | 0.867 |
| shaded | 120 | 0.642 | 0.408 | 0.183 | 1.000 |
| textured | 120 | **0.808** | 0.142 | 0.167 | 0.600 |

→ H1 (line → textured monotone S-curve) **부분 지지**: 양 끝점은 예측대로
(line 0.575 < textured 0.808) 이지만 `shaded` 와 `filled` 가 중간에서 사실상
tie. 노이즈 (n=120 per cell → ~4.5 pp std error) 이거나 음영 cue 단독으로는
uniform fill 보다 우세하지 않다는 가능성.

**bg_level 별** (axis B):

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 240 | 0.488 | N/A |
| ground | 240 | **0.854** | 0.783 |

→ **지면 존재가 PMR 을 +36 pp 증가시킴.** 측정한 단일 요인 효과 중 최대;
H3 와 "support plane 이 physical-object 해석을 유발한다" 는 인지과학 예측과
일치.

**cue_level 별** (axis C):

| cue_level | n | pmr |
|---|---|---|
| none | 160 | 0.500 |
| wind | 160 | 0.513 |
| arrow_shadow | 160 | **1.000** |

→ `arrow_shadow` 는 PMR 을 1.0 으로 포화 (trajectory arrow 가 완전한
give-away — 모델이 "공이 갈 방향" 으로 읽음). **wind 마크는 거의 효과
없음** — VLM 이 프로그램적 wind streaks 를 airflow 로 해석하지 않음.
surprise #1 참조.

**prompt_variant 별** (방법론):

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 240 | **0.800** | 0.000 | 0.917 |
| forced_choice | 240 | 0.542 | 0.450 | 0.650 |

→ 옵션 D ("abstract shape") 가 제공되면 모델이 45 % 그것을 사용; open-ended
모드에서는 stimulus 를 *한 번도* 자발적으로 abstract 라고 부르지 않음.
"ball" 라벨의 언어 prior 가 open variant 를 완전 지배. 계측 경고: open-ended
PMR 은 axis D label 효과로 부풀려져 있으므로, behavioral S-curve 는
forced-choice 서브셋에서 읽는 것이 best.

## Surprise / 노트

1. **Wind cue 가 Qwen2.5-VL-7B 에 invisible.** 객체 한 쪽에 anchor 된 15 개
   회색 작은 호가 PMR 을 "no cue" 대비 안 움직임 (0.513 vs 0.500). 더 강한
   visual 고려: 객체 wake 의 blurred motion trail, 또는 perspective 적용된
   particle streaks. MVP-full run 전 `primitives.draw_wind_marks` 를 개선
   하거나 axis C 에서 wind level 을 `motion_blur` / `dust_cloud` 로 교체.
2. **arrow+shadow cue 가 너무 강함.** PMR=1.0 은 측정할 정보가 없다는 뜻.
   MVP-full 에서 axis C 를 `{none, cast_shadow_only, trajectory_arrow_only,
   both}` 로 분리해서 boost 의 얼마나가 shadow 에서 (Kersten/Mamassian 의
   ground-attachment 예측 지지) vs arrow 에서 (순수 directional cue) 오는지
   확인.
3. **Lexicon 튜닝 중요.** 초기 stem set 이 "moving" (because "move" ≠ "moving"
   의 prefix) 과 "continue" 를 놓침, textured cell 에서 ~2 pp 손실. 패치된
   stems 가 `lexicons.py` 에 commit; regression test 추가됨. 향후 lexicon
   편집은 `tests/test_pmr_scoring.py` 를 통해 진행.
4. **temperature=0 에서 모든 seed 가 동일 generation** per (stimulus, prompt).
   따라서 모든 cell 의 RC = 1.0 — T=0 에서는 useful signal 아님. MVP-full
   run 에서는 `temperature=0.7` 설정 + `seeds_per_cell` 증가로 RC 를
   informative 하게 (`references/project.md` §2.2 의 Sub-task 1 metric).
5. **Raw 응답이 sensible** (예: ground cell 에서 "The ball will collide with
   the surface below it"; blank cell 에서 "The ball will remain stationary
   unless acted upon by an external force" — Newton's-first-law framing).
   모델의 오류는 systematic 이지 random 이 아님.

## 다음 actions

- 위 세 fix 를 적용한 MVP-full run (wind → motion trail / dust; cue 축
  분리; temperature 0.7 + 더 많은 seed). 다음 run 전 `configs/mvp_full.py`
  에 spec 명시.
- Sub-task 3 logit-lens 분석을 위한 hidden state 준비를 위해
  `capture_lm_layers = (5, 10, 15, 20, 25)` 활성화.
- MVP-full 실행 전 axis D 를 `("circle", "ball", "planet")` 로 확장해서 H2
  언어 prior 효과 측정.
