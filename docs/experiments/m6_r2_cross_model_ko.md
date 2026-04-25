# M6 round 2 — Cross-model 실행 로그

세 sub-task 를 2026-04-25 에 실행하여 M6 r1 (LLaVA-1.5 cross-model behavioral)
을 직교 세 방향으로 확장:

- **r2a**: 세 번째 행동 cross-model 점 — InternVL3-8B-hf.
- **r2b**: LLaVA-1.5 activation captures + cross-model M3 (vision probe) /
  M4 (LM probe) AUC 비교.
- **r2c**: forced-choice first-token logit-ratio 채점 — LLaVA 의 "A" 편향
  (M6 r1, M4c) 이 greedy decoding 단계인지 underlying logit 단계인지 확인.

## r2a — InternVL3-8B-hf cross-model

### 설정

- Configs: `cross_model_internvl3.py` + `cross_model_internvl3_label_free.py`.
- Stimuli: M2 manifest 재사용 (480 stim).
- 생성: T=0.7, top_p=0.95, max_new_tokens=96.
- Forced-choice 제외 (smoke 에서 LLaVA 와 달리 정상 응답 확인 — 정성적
  evidence 충분; 전체 FC sweep 은 ~80 분 추가).
- 출력:
  - `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — labeled, 1440 rows.
  - `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — label-free, 480 rows.

### 행동 결과

#### InternVL3 labeled — label 별

| label  | PMR | GAR | hold_still |
|---|---|---|---|
| ball   | 1.000 | 0.816 | 0.000 |
| circle | 1.000 | 0.791 | 0.010 |
| planet | 1.000 | 0.431 | 0.023 |

PMR 이 **모든 (object × cue × label) 조합에서 1.000** — InternVL3 는
M2 자극의 open prompt 에서 abstract reading 을 절대 선택하지 않음.

#### InternVL3 label-free — 전체

| metric | 값 |
|---|---|
| PMR | 0.989 |
| GAR | 0.644 |
| hold_still | 0.065 |

Label 없이도 PMR 이 0.99. InternVL3 가 본 연구에서 가장 강력한 시각 physics
prior 보유.

#### InternVL3 paired delta vs label-free

| label  | mean `PMR(label) − PMR(_nolabel)` |
|---|---|
| ball   | +0.010 |
| circle | +0.010 |
| planet | +0.010 |

**세 label 모두 +0.010 — essentially noise.** Language prior 가 작동할 여지
없음.

### 3-model 교차 비교

`object_level` 별 `PMR(_nolabel)`:

| object   | Qwen2.5-VL | LLaVA-1.5 | InternVL3 |
|---|---|---|---|
| line     | 0.942 | 0.142 | 0.992 |
| filled   | 0.933 | 0.317 | 0.967 |
| shaded   | 0.942 | 0.592 | 1.000 |
| textured | 0.975 | 0.483 | 1.000 |

Paired delta `PMR(label) − PMR(_nolabel)` (480 matched seed, T=0.7):

| label  | Qwen   | LLaVA  | InternVL3 |
|---|---|---|---|
| ball   | +0.006 | +0.475 | +0.010 |
| circle | −0.065 | +0.173 | +0.010 |
| planet | +0.006 | +0.244 | +0.010 |

H7 GAR by label:

| label  | Qwen  | LLaVA | InternVL3 |
|---|---|---|---|
| ball   | 0.706 | 0.356 | 0.816 |
| circle | 0.753 | 0.153 | 0.791 |
| planet | 0.319 | 0.072 | 0.431 |

`planet GAR << ball/circle GAR` 가 세 모델 모두에서 재현. H7 cross-model
재확인.

### `line/blank/none` 의 InternVL3 샘플 응답

- ball: "The ball will likely fall downward due to gravity." × N
- circle: "The circle will likely move downward." / "The circle will likely start to move or rotate."
- planet: "The planet will continue to remain stationary in the center of the image." / "The planet will continue to rotate on its axis."

InternVL3 는 `line/blank/none` 의 `circle` 에서도 physics-mode 로 commit
하지만, regime 선택 (planet → orbital) 은 여전히 작동.

## r2b — LLaVA-1.5 activation captures

### 설정

- Config: `cross_model_llava_capture.py`.
- `capture_lm_layers=(5, 10, 15, 20, 25)`, `capture_vision_layers=(3, 7, 11, 15, 19, 23)`.
- 480 stim × 3 labels × 1 prompt (open) = 1440 inferences + 480 captures.
- 출력: `outputs/cross_model_llava_capture_20260425-054821_65214a5d/`.
- 디스크: 14 GB activations.

### 코드 변경

`_resolve_vision_blocks` 가 LLaVA-1.5-hf wrapper (CLIPVisionModel 의
encoder 가 `vt.encoder.layers` 에 직접 위치, `vision_model` wrapper 없음)
를 처리하도록 업데이트. Vision-hook 출력도 (1, 577, 1024) 에서 (577, 1024)
로 squeeze 하여 probing 함수 (`(n_tokens, dim)` 기대) 가 변경 없이 작동.

### 행동 일관성

LLaVA capture run vs LLaVA M6 r1 run (no captures):
- (object × label) 별 PMR 평균 절대 차이: 0.056
- 최대 차이: 0.158
- GAR 평균 절대 차이: 0.043

Stochastic seed 차이로 spread 설명 가능; 행동 패턴은 보존.

### Cross-model probe AUC 비교

#### Vision encoder probe AUC (open prompt)

| layer | Qwen2.5-VL (M3) | LLaVA-1.5 (M6 r2b) |
|---|---|---|
| 3  | 0.980 | 0.707 |
| 7  | 0.985 | 0.731 |
| 11 | 0.986 | 0.726 |
| 15 | 0.979 | 0.732 |
| 19 | 0.986 | 0.715 |
| 23 | 0.986 | 0.728 |

(Qwen 은 27, 31 도 측정; 둘 다 0.985 / 0.985.)

LLaVA 의 CLIP-ViT-L vision encoder 가 Qwen 의 SigLIP encoder 보다 physics-vs-
abstract 분리에서 **~25 percentage point 뒤처짐**. 깊이 전반에서 격차 균일
— 두 encoder 모두 "더 깊은 layer 에서 더 큰 분리" 패턴 없이 평탄.

#### Visual-token 위치에서의 LM probe AUC (open prompt)

| layer | Qwen2.5-VL (M4) | LLaVA-1.5 (M6 r2b) |
|---|---|---|
| 5  | 0.939 | 0.732 |
| 10 | 0.944 | 0.753 |
| 15 | 0.947 | 0.747 |
| 20 | 0.953 | 0.748 |
| 25 | 0.944 | 0.736 |

LLaVA 의 LM AUC 는 vision AUC 를 추적 — 둘 다 ~0.73-0.75, boomerang
"recovery" 없음. Qwen 은 LM 통과 시 약간의 손실 (0.985 → 0.94 ≈ 4 pp).
LLaVA 는 LM 이 vision encoder 대비 신호를 추가하지도 빼지도 않음.

#### Boomerang gap (encoder AUC vs LM AUC vs behavioral)

| pipeline stage | Qwen | LLaVA |
|---|---|---|
| Vision encoder AUC (open) | 0.985 | 0.728 |
| LM AUC at visual tokens | 0.946 | 0.745 |
| Behavioral PMR (open) | 0.93 | 0.78 |

Qwen 의 "encoder knows, decoder gates" boomerang 존재 (∆ = 0.985 → 0.93
= 5 pp). LLaVA pipeline 은 거의 평탄 — encoder 와 behavior 가 비슷한 수준
(0.73 vs 0.78).

### 함의: visual-saturation 은 vision encoder 에 뿌리내림

LLaVA-1.5 의 "약한 visual prior" (낮은 `PMR(_nolabel)`) 는 후반-단계 gating
현상이 아니라 **vision encoder 자체에서 결정**됨. LLaVA 의 CLIP-ViT-L 은
physics-vs-abstract 차원에서 AUC ~0.73 만 달성하는데, Qwen 의 SigLIP 은
AUC ~0.99 달성. Label 추가는 누락된 visual signal 을 보완하므로 LLaVA 의
ball paired delta 가 +0.475; Qwen 에서는 visual signal 이 이미 saturated
이므로 label 이 할 일이 없음.

## r2c — Forced-choice first-token logit-ratio 채점

### 설정

`option_logits` 가 FC row 별로 이미 저장되어 있음 (Qwen M2 forced_choice;
Qwen FC label-free; LLaVA FC label-free). 재산출:
- `logit_argmax`: 첫 생성 토큰 단계의 4 letter logit argmax (sampling
  temperature + top_p 의 warping 후).
- `pmr_from_logit_argmax`: logit_argmax ∈ {A, B, C} 이면 1, 아니면 0.

데이터 한계: 저장된 logit 은 post-warping (HF `output_scores` 동작 방식)
이므로 top_p=0.95 로 필터된 토큰은 `-inf` 로 표시. 그래도 non-inf entry
의 argmax 는 정보적 — 한 letter 만 finite logit 이면, top_p 하의 그
확률 mass 는 ≥0.95.

### Top_p 필터를 통과하는 letter 수

| run | n_real 분포 (rows × n_real_letters) |
|---|---|
| Qwen M2 FC labeled (1440 rows) | 1: 736, 2: 387, 3: 317 |
| Qwen FC label-free (480 rows) | 1: 278, 2: 81, 3: 119, 4: 2 |
| LLaVA FC label-free (480 rows) | 1: 430, 2: 50 |

LLaVA 의 90% rows 가 한 letter (항상 `A`) 만 top_p 를 통과 — A 의
underlying probability 가 90% 의 경우에 ≥0.95. C 와 D 는 top_p set 에
한 번도 진입 안 함.

### Logit-argmax 분포

| run | A | B | C | D |
|---|---|---|---|---|
| Qwen M2 FC labeled | 938 | 5 | 0 | 497 |
| Qwen FC label-free | 371 | 75 | 0 | 34 |
| LLaVA FC label-free | 480 | 0 | 0 | 0 |

LLaVA logit-argmax 가 100% A. Greedy first_letter 는 477/480 = 99.4% A
(stray B 3개). 편향은 logit 수준이지 greedy sampling 으로 인한 것이 아님.

### Greedy vs logit-argmax 일치율

| run | greedy == logit_argmax |
|---|---|
| Qwen M2 FC labeled  | 0.695 |
| Qwen FC label-free | 0.756 |
| LLaVA FC label-free | 0.994 |

Qwen 불일치는 주로 greedy 가 "other" 토큰 생성 (예: 답변 시작 시 줄바꿈
또는 따옴표) 때문이며, A/B/C/D 와 매칭 안 됨. Logit-argmax 는 항상 4 중
하나를 선택.

### PMR-from-logit vs PMR-from-text

| run | text-PMR | logit-PMR | Δ |
|---|---|---|---|
| Qwen M2 FC labeled | 0.510 | 0.655 | +0.145 |
| Qwen FC label-free | 0.769 | 0.929 | +0.160 |
| LLaVA FC label-free | 1.000 | 1.000 | 0.000 |

Qwen 의 경우 logit-PMR 이 text-PMR 보다 14-16 pp 높음 — text-PMR 이
greedy "other" 출력으로 under-count (예: 모델이 답을 apostrophe 또는
newline 으로 시작). LLaVA 는 두 metric 모두 1.000 (둘 다 A-locked).

### 결론

- LLaVA 의 FC pathology 는 underlying logit 수준. Greedy → first-token
  logit ratio rescue 가 작동하지 않음. LLaVA FC 를 살리려면 추론을
  steering 해야 함 (A 생성 금지 → B/C/D 중 선택 강제) — 다른 probe.
- Qwen 의 경우 logit-argmax 가 text-PMR 보다 깨끗한 FC metric: trivial
  first-token formatting drift 무시 + ~14 pp 신호 회복.

## 원본 artifacts

- `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — InternVL3 labeled.
- `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — InternVL3 label-free.
- `outputs/cross_model_llava_capture_20260425-054821_65214a5d/` — LLaVA captured (predictions + 14 GB activations + probing_vision/ + probing_lm/ subdirs).
- `outputs/{mvp_full_*, fc_label_free_*, label_free_*, cross_model_llava_*}/predictions.jsonl` — option_logits 가 r2c 재분석을 위해 보존됨.
