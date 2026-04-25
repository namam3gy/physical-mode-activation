# M4b Label-Free Prompt — 실행 로그

ROADMAP §4.9 ("H2 null-hypothesis test 로서의 label-free prompt") 를 실행하고,
부산물로 M4 의 switching-layer degeneracy 를 재검토한다.

실행일: 2026-04-25.

## 설정

- Config: `configs/label_free.py` (신규).
- Stimuli: M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/` 재사용
  (480 stim: 4 obj × 3 bg × 4 cue × 1 event × 10 seeds).
- 모델 / 생성: M2 와 동일 — Qwen2.5-VL-7B-Instruct, bf16, T=0.7, top_p=0.95,
  `max_new_tokens=96`.
- Prompt: `open_no_label` variant (신규), 텍스트는
  `"What do you see in the image? What might happen next? Answer in one short sentence."`.
- Labels: `("_nolabel",)` sentinel, single iteration (label × variant 조합 없음).
- Activation capture: `open_no_label` 프롬프트 하에서 `lm_hidden_{5,10,15,20,25}`
  (M2 layer set 매칭으로 paired M4 재실행).
- 출력: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions + 480
  activation safetensors (각 5 layer). Wall-clock ≈ 13 분.

## 행동 결과

### 전체

| metric | 값 |
|---|---|
| n | 480 |
| PMR | 0.948 |
| hold_still | 0.127 |
| abstract_reject | 0.002 |
| GAR | 0.728 |

### M2 open prompt 와의 paired PMR delta (동일 `(obj, bg, cue, seed)`)

| M2 label | 480 pair 평균 `PMR(label) − PMR(_nolabel)` |
|---|---|
| ball   | **+0.006** |
| circle | **−0.065** |
| planet | **+0.006** |

- `ball` 과 `planet` label 은 no-label baseline 대비 PMR 에 기여 없음.
- `circle` label 은 평균 ~6.5 pp **억제**.
- M2 의 "ball 이 PMR 을 높인다" 프레이밍이 뒤집힘 — 실제 delta 는
  "circle 이 PMR 을 낮춘다" (ball ≈ no-label = 시각 default).

### object_level 별 PMR — open prompt

| object | ball (M2) | circle (M2) | planet (M2) | _nolabel (LF) | ball − _nolabel | circle − _nolabel |
|---|---|---|---|---|---|---|
| line     | 0.950 | 0.850 | — | 0.942 | +0.008 | −0.092 |
| filled   | 0.950 | 0.892 | — | 0.933 | +0.017 | −0.042 |
| shaded   | 0.933 | 0.892 | — | 0.942 | −0.008 | −0.050 |
| textured | 0.983 | 0.900 | — | 0.975 | +0.008 | −0.075 |

`circle` 억제는 `line` 에서 가장 크고 (−9.2 pp), `filled` 에서 가장 작다 (−4.2 pp).
패턴: 이미지가 더 추상적일수록 label 이 시각 cue 를 override 할 여지가 커진다.

### cue_level 별 PMR — open prompt

| cue          | ball  | circle | planet | _nolabel | circle − _nolabel |
|---|---|---|---|---|---|
| both         | 0.992 | 0.992 | 0.992 | 0.983 | +0.008 |
| cast_shadow  | 1.000 | 0.850 | 0.958 | 0.967 | −0.117 |
| motion_arrow | 1.000 | 1.000 | 0.992 | 1.000 | 0.000  |
| none         | 0.825 | 0.692 | 0.875 | 0.842 | −0.150 |

`motion_arrow` 는 `circle` label 억제를 완전히 override (+0.000).
Cue 가 전혀 없으면 circle 이 15 pp 억제 — 모든 cell 중 최대 억제폭.

### 가장 정보가 많은 cell — `line/blank/none` (완전 추상 이미지)

| label    | PMR  | hold_still |
|---|---|---|
| _nolabel | 0.40 | 0.20 |
| ball     | 0.40 | 0.60 |
| circle   | 0.10 | 1.00 |
| planet   | 0.70 | 0.30 |

- `_nolabel` 과 `ball`: 동일 PMR (0.40); `ball` 은 regime 을 "stays" 쪽으로 이동
  (hold_still 0.20 → 0.60), 전체 physics-mode 비율은 올리지 않음.
- `circle`: PMR 이 30 pp 하락; 10/10 이 hold-still.
- `planet`: PMR 이 30 pp 상승 — 이 셀은 **label 이 시각 baseline 대비 physics-mode
  를 실제로 증가시키는 유일한 경우**. Planet label 이 가져오는 orbital physics
  prior 가 이미지 단독으로는 촉발하지 못하는 것을 보완.

## LM probing 결과 (M4 재실행)

스크립트: `uv run python scripts/05_lm_probing.py --run-dir outputs/label_free_<ts> --sources open_no_label`.

### Physics margin — (layer, object_level) 별

Label-free:

| layer | filled | line | shaded | textured |
|---|---|---|---|---|
| 5  | 0.08 | 0.09 | 0.12 | 0.15 |
| 10 | 0.29 | 0.27 | 0.33 | 0.38 |
| 15 | 0.38 | 0.35 | 0.44 | 0.49 |
| 20 | 0.89 | 0.87 | 0.97 | 1.05 |
| 25 | 3.94 | 3.76 | 4.29 | 4.35 |

M2 (ball/circle/planet-labeled): **위와 완전히 동일** (max diff = 0.0).

### 방법론적 발견 — visual-token capture 는 prompt 독립

visual-token 위치의 activation 이 M2 run (label 있음) 과 label-free run 간
**bit-for-bit 동일** 함을 `outputs/*/activations/line_blank_none_fall_000.safetensors`
를 layer 5/10/15/20/25 각각에 대해 load + diff 하여 확인 — max 절대 diff = 0.0.
오직 `input_ids` 와 `visual_token_mask` (둘 다 prompt-length artefact) 만
길이가 다름.

이는 Qwen2.5-VL 의 chat template 구조에서 나오는 결과: image token 이 user message
내에서 question text 보다 앞에 있으므로, causal attention 하에서 question text
(label 포함) 는 visual-token 위치로 역전파되지 않는다. 따라서 visual-token 에서
의 capture 는 label 기여를 관찰하지 못한다.

함의: M4 의 switching-layer 가 480 샘플 모두에서 L5 인 것은 **label 무관하게 LM 이
L5 부터 physics-mode 로 커밋한다** 는 증거가 아님 — **vision encoder + 초기 LM
layer 가 이미지만으로 physics-biased representation 을 encoding 하며**,
label 의 행동적 효과는 capture 영역 바깥 (last question-token 또는 start-of-answer
위치) 의 downstream 에 위치함.

### Per-layer PMR probe AUC (open_no_label source)

| layer | AUC mean | AUC std | accuracy | n_pos | n_neg |
|---|---|---|---|---|---|
| 5  | 0.937 | 0.036 | 0.956 | 455 | 25 |
| 10 | 0.955 | 0.016 | 0.956 | 455 | 25 |
| 15 | 0.951 | 0.037 | 0.956 | 455 | 25 |
| 20 | 0.952 | 0.027 | 0.956 | 455 | 25 |
| 25 | 0.958 | 0.027 | 0.956 | 455 | 25 |

참고: label-free 전체 PMR = 0.948 이어서 n_pos/n_neg split 이 극단적으로 불균형
(455/25). 여기서의 AUC 0.94-0.96 은 불균형 문제를 반영한 것이지 M2 의
balanced (312/168) `forced_choice` source 대비 0.94-0.95 의 메커니즘 변화가
아님.

## 원본 artifacts

- `outputs/label_free_20260425-031430_315c5318/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/label_free_20260425-031430_315c5318/activations/*.safetensors` — 480 files, M2 와 동일한 visual-token hidden states.
- `outputs/label_free_20260425-031430_315c5318/probing_lm/` — layer-sweep + logit-lens outputs.
- `outputs/label_free_20260425-031430_315c5318/summary_*.csv` — factor-level rollups.

## 샘플 raw 응답 (T=0.7, `open_no_label` prompt)

- `line/blank/none_000`: "The image shows a simple outline of a circle, and it is unlikely to change as it appears to be a static image."
- `textured/blank/none_000`: "The image shows a bowling ball with holes, and it's likely to roll if thrown or pushed."
- `line/blank/both_000`: "The circle is likely to fall towards the oval shape due to gravity."
- `textured/ground/both_000`: "The ball is about to roll down the incline toward the ground." (전형적)

모델은 시각적으로 객체를 식별 (circle / bowling ball) 하고 image content + cue
에 기반하여 physics 를 적용 — prompt 측 label 이 필요 없음.
