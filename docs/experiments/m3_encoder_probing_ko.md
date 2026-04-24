# M3 — Vision encoder probing (2026-04-24)

- **명령**: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 --output-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations --layers 3,7,11,15,19,23,27,31`
- **Capture 시간**: ~70 초 (10 it/s forward-only) + model load 20 초
- **Disk**: 12 GB (480 stimuli × 8 layers × (1296 tokens × 1280 dim × 2 bytes))
- **Probe**: sklearn LogisticRegression, StratifiedKFold (5), token axis 평균 pool. 코드: `src/physical_mode/probing/vision.py`
- **출력**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_vision/*.csv`
- **심층**: `docs/insights/m3_encoder_boomerang_ko.md`.

## 헤드라인: encoder-decoder boomerang 이 real 하고 **포화**

**Stimulus-property probe (layer × target 별 AUC)**: vision encoder 가 layer 3
부터 **모든** factorial 축을 AUC=1.00 으로 선형 분리:

| target | layer 3 | 15 | 31 |
|---|---|---|---|
| y_bg_ground (bg != blank) | 1.00 | 1.00 | 1.00 |
| y_bg_scene | 1.00 | 1.00 | 1.00 |
| y_obj_3d (shaded/textured) | 1.00 | 1.00 | 1.00 |
| y_obj_textured | 1.00 | 1.00 | 1.00 |
| y_cue_any | 1.00 | 1.00 | 1.00 |
| y_cue_shadow | 1.00 | 1.00 | 1.00 |
| y_cue_arrow | 1.00 | 1.00 | 1.00 |

→ **이러한 feature 에 linear access 만 있는 어떤 downstream 시스템도 모든
factorial 축을 완벽히 회복할 수 있다.** Encoder 는 stimulus descriptor 에
대해 정보 bottleneck 이 0.

같은 stimuli 에 대한 forced-choice **behavioral PMR**:

| axis | level | beh. PMR |
|---|---|---|
| bg | blank / ground / scene | 0.51 / 0.71 / 0.77 |
| object | line / filled / shaded / textured | 0.58 / 0.65 / 0.71 / 0.71 |
| cue | none / cast_shadow / motion_arrow / both | 0.28 / 0.49 / 0.93 / 0.95 |

**LM 은 encoder 가 완벽히 인코딩하는 속성에 imperfectly sensitive**. Gap 은
visual feature 의 downstream 에서만 발생.

## Controlled probe: no-cue 서브셋

`cue_level=none` 으로 제한 (120 stimuli, red-arrow shortcut 없음, cast shadow
없음). Encoder probe 를 behavioral forced-choice PMR 에 학습.

| layer | encoder AUC | behavioural PMR |
|---|---|---|
| 3 | 0.793 | 0.28 |
| 11 | 0.852 | 0.28 |
| **19** | **0.890** | 0.28 |
| 27 | 0.852 | 0.28 |
| 31 | 0.859 | 0.28 |

Encoder 의 layer-19 activation 은 어떤 "no-cue" stimuli 가 LM 에서 physics-mode
가 될지 **AUC 0.89** 로 예측 — 하지만 그 stimuli 의 **28 %** 만 실제로
behavior 에서 physics-mode 를 trigger. Calibration 으로 읽으면: *"encoder 는
어느 cell 이 physics-mode 를 trigger 할지 알지만, LM 은 그 일부만 통과시킨다"*.

## Per-object-level encoder AUC vs behavior (forced-choice)

| object_level | encoder AUC @ L31 | behavioural PMR | gap |
|---|---|---|---|
| line | 0.944 | 0.583 | **+0.361** |
| filled | 0.950 | 0.647 | +0.303 |
| shaded | 0.943 | 0.711 | +0.232 |
| textured | 0.952 | 0.714 | +0.238 |

Boomerang gap 이 **가장 추상적인 객체에서 최대** — H4 (언어-prior 대 vision
tension 이 추상도가 높을수록 악화) 와 일관.

## 방법론적 caveat

프로그램적 자극은 encoder AUC 1.0 을 trivially 달성 가능 (단순 mean-pooled
representation 으로 충분). 1.0 vs behavioral-PMR gap 의 sharpness 는 **정보가
명확히 존재할 때도 LM 이 그것을 physical-mode 행동으로 route 하지 못한다** 는
증거 — 하지만 photographic stimulus 에서는 encoder 측 자체가 imperfect 일 수
있고, real-world input 에서의 effect size 는 cross-validation 필요. M6
(cross-model) 과 axis A 의 photorealistic 확장이 이를 다룰 것.

## 가설 스코어카드 post-M3

| H | M2 후 상태 | M3 후 상태 | 변화 |
|---|---|---|---|
| **H-boomerang** | 후보 (project doc §1.4) | **지지 (증거 포화)** | 모든 axis 에서 encoder AUC 1.0; behavioral PMR 0.28-0.95 |
| H4 (open-forced gap) | 지지 | **지지 + mechanism** | Per-object-level encoder AUC ~ constant (~0.95) while behavior varies (0.58-0.71); gap 이 LM 에 집중 |
| H6 (shadow standalone) | 지지 (수정) | **지지** | y_cue_shadow 에 encoder AUC=1.0 → 정보 full; LM 이 부분만 사용 (0.49 PMR) |

## Unlock

- **Sub-task 3 (M4) 가 이제 full machinery 준비됨**: LM hidden state 5 layer
  (M2 에서) + vision hidden state 8 layer (M3 에서). Visual-token 위치의 LM
  hidden state 에 logit lens 가 다음 natural figure.
- **Sub-task 4 (M5, activation patching)** 전제: attention capture 필요.
  `capture_lm_attentions=True` 로 flip 하고 mini-batch 재실행.
- **Photorealistic 자극 확장** (추가 아이디어) 은 encoder AUC 가 saturated
  되지 않은 상황에서 boomerang 이 살아남는지 검증할 것.
