---
section: M2 cross-model (M6 r7)
date: 2026-04-26
status: complete (5-model M2-stim 커버, bootstrap CI + per-model v_L 추출)
hypothesis: M2 의 H1/H2/H7 protocol 이 5-model chain 에서 architecture-level reframe 이 예측하는 방향으로 일반화된다
---

# M2 cross-model (5-model M2-stim apples-to-apples)

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H1** — abstraction 축 (line → filled → shaded → textured) 을 따라 PMR 이 S-shape 으로 상승; ground 도입이 가장 큰 단일 jump.
> - **H2** — label (ball / circle / planet) 자체가 PMR 을 독립적으로 끌어올림 — 시각 증거를 넘는 language-prior 기여.
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택.
> - **H-encoder-saturation** — 합성 stim 위 behavioral PMR(_nolabel) saturation 이 architecture 수준 (encoder + LM 결합) 에서 결정.
> - **M2** — ST1 MVP-full — 5-axis factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M4b** — M4 + label-free 프롬프트로 H2 null test; Qwen 에서 H2 가 비대칭 (circle 억제, ball 증강 아님).
> - **M5a** — ST4 VTI steering — LM L10 시각 토큰에 +α·v_L10 더하면 line/blank/none 이 "정지" → physics-mode 로 뒤집힘.
> - **M6** — ST5 cross-model sweep — M6 r1-r6 참조.
> - **M8a** — Stim 다양화 — 비-원 합성 shape (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **§4.6** — VTI 역방향 counterfactual stim — Qwen2.5-VL pixel_values 위 픽셀-공간 gradient ascent.
> - **v_L10** — M5a class-mean diff (physics − abstract) 에서 유도된 layer 10 LM hidden space (모델별 dim 3584 / 4096) steering 방향. Unit norm.

## 질문

M2 5-axis factorial (480 stim × 3 라벨 × 2 프롬프트 = 2880 추론) 은
원래 Qwen 만 실행. M6 r1 가 같은 protocol 을 LLaVA-1.5 에 재실행했고,
나머지 3 모델은 M8a-stim (다른 factorial) 에서만 검증됨. 이 마일스톤
은 갭을 채운다: **5 모델 모두 같은 M2 stim 위에서 같은 M2 protocol
실행**, 그리고 LM-layer activation 캡처도 포함하여 §4.6 cross-model
의 전제인 per-model v_L10 추출도 가능.

## 방법

5 capture-enabled run + 5 label-free run, 모두 M2 stim
(`inputs/mvp_full_20260424-093926_e9d79da3`) 위:
- Qwen2.5-VL: 원래 M2 + label-free (기존)
- LLaVA-1.5: M6 r1 + M6 r2b capture (기존)
- LLaVA-Next: 이 마일스톤 (`cross_model_llava_next_capture_*`)
- Idefics2: 이 마일스톤 (`cross_model_idefics2_capture_*`)
- InternVL3: M6 r2a (기존) + 이 마일스톤 (capture run 추가)

Bootstrap CIs: 5000 iter, prediction-level 재샘플링 (seed 42).

## 결과

![M2 cross-model PMR ladder (5 VLMs × 4 카테고리)](../figures/m2_cross_model_pmr_ladder.png)

| Model | ball | circle | planet | _nolabel | H1 range |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL | 0.946 | 0.850 | 0.917 | 0.938 | +0.05 |
| LLaVA-1.5 | 0.858 | 0.556 | 0.627 | **0.383** | **+0.30** |
| LLaVA-Next | 0.979 | 0.871 | 0.950 | 0.790 | +0.14 |
| Idefics2 | 0.994 | 0.927 | 0.881 | 0.967 | +0.09 |
| InternVL3 | 1.000 | 0.998 | 0.983 | 0.988 | +0.02 |

### 모델별 H1 abstraction ramp

![M2 cross-model H1 ramp (line → filled → shaded → textured)](../figures/m2_cross_model_h1_ramp.png)

LLaVA-1.5 가 가장 깨끗한 monotone S-curve (0.51 → 0.81, 범위 0.30).
Qwen / Idefics2 / InternVL3 는 천장에 있어 ramp 가 보이지 않음.
LLaVA-Next 는 중간 (범위 0.14, shaded / textured 에서 대부분 saturate).

### 모델별 H2 paired-delta

![M2 cross-model H2 paired-delta (PMR(label) − PMR(_nolabel))](../figures/m2_cross_model_h2_paired_delta.png)

| Model | ball Δ | circle Δ | planet Δ |
|---|---:|---:|---:|
| Qwen2.5-VL | +0.008 | **−0.088** | −0.021 |
| LLaVA-1.5 | **+0.475** | **+0.173** | **+0.244** |
| LLaVA-Next | +0.190 | +0.081 | +0.160 |
| Idefics2 | +0.027 | **−0.040** | **−0.085** |
| InternVL3 | +0.012 | +0.010 | −0.004 |

세 가지 distinct H2 패턴:

1. **LLaVA-1.5 / LLaVA-Next** (포화되지 않은 CLIP 인코더): 모두 양수,
   "classical H2" — 모든 라벨이 label-free 대비 PMR 증가. LLaVA-1.5
   가 가장 깨끗 (ball +0.48), LLaVA-Next 는 CLIP 부분 포화로 약화.

2. **Qwen / Idefics2** (포화된 SigLIP-계열 인코더): 비대칭 — 비-물리
   라벨 (circle, planet) 이 PMR 을 no-label baseline 아래로 억제.
   M4b 의 "circle override" 패턴이 Idefics2 에서 재현 (circle −0.04,
   planet −0.09). 메커니즘은 abstract 성향 라벨이 포화된 image-prior
   를 less commitment 방향으로 override.

3. **InternVL3** (완전 포화): 모든 delta ≈ 0. 라벨이 작용할 헤드룸
   없음.

이게 H2 수준의 **architecture-level reframe**: H2 는 "라벨이 항상
PMR 을 더한다" 가 아님 — "encoder 에 헤드룸이 있을 때 라벨이 PMR 을
더하고, 없으면 아무것도 하지 않거나 baseline 아래로 억제한다." 어느
경로가 적용되는지는 encoder 포화가 결정.

## Per-model v_L 추출 (§4.6 cross-model 준비)

각 모델의 M2 캡처에서 layer L ∈ {5, 10, 15, 20, 25} 별 class-mean
diff `mean(h_L | PMR=1) − mean(h_L | PMR=0)` 계산.

| Model | hidden dim | n_pos | n_neg | ||v_L10|| |
|---|---:|---:|---:|---:|
| LLaVA-1.5 | 4096 | 375 | **105** | 0.99 |
| LLaVA-Next | 4096 | 471 | 9 | 0.18 |
| Idefics2 | 4096 | 475 | 5 | 10.4 |
| InternVL3 | 3584 | 479 | **1** | 8.6 |

### Class-imbalance 한계

비-Qwen 모델 4 개 중 3 개는 M2 stim 에서 너무 포화되어 깨끗한 v_L10
추출 불가:
- LLaVA-1.5: PMR=0 예시 105 개 (105/480 = 22%) — 적절.
- LLaVA-Next: 9 개; Idefics2: 5 개; InternVL3: 1 개 PMR=0 예시.

n_neg ∈ {1, 5, 9} 로 class-mean diff 하면 "abstract → physics-mode"
축이 깔끔하게 표현되지 않은 noisy direction 이 나옴 — 오히려 negative-
class 대표 예시의 stim-별 분산을 표현. 이게 v_L10 수준에서 표현된
architecture-level reframe: **포화된 모델에 대해 M2 stim 은 v_L10 을
class-mean diff 로 정의할 만큼 충분한 행동 분산을 갖지 않는다**.

§4.6 cross-model 함의: 의미 있는 per-model v_L10 추출에는 (a) 포화
모델도 가끔 실패하는 더 어려운 stim (예: M8c 사진, M8a 비-원 도형),
또는 (b) 다른 축 정의 (예: PMR 유도 direction 대신 label 유도
direction projection) 가 필요.

LLaVA-1.5 의 v_L10 만이 현 단계에서 직접적인 apples-to-apples
cross-model 유사체. §4.6 픽셀-공간 gradient ascent 는 LLaVA-1.5 에는
가능; 나머지 3 모델은 M2 만으로는 prerequisite v_L10 이 신뢰 불가.

## 가설 함의

- **H1**: cross-model strict → unsaturated-only 확인. LLaVA-1.5 가
  깔끔한 monotone S-curve 보임; 천장 모델은 축 압축.
- **H2**: 5-model granularity 로 특성화. 세 패턴 (포화되지 않은 모델
  positive, near-saturated 비대칭, fully saturated ≈ 0) 이 encoder-
  saturation 예측과 부합.
- **H-encoder-saturation**: 5-model M2-stim apples-to-apples PMR
  ladder 가 이제 존재 (이전엔 M2 = Qwen 만 + cross-model 은 M8a-stim).
  architecture-level reframe 강화.

## 한계

1. **Open-prompt 만**: 비-Qwen 모델에 forced-choice 는 LLaVA-A bias 와
   Mistral-기반 모델 처리 불확실성 때문에 제외. Qwen 만 FC 있음.
2. **M2-stim apples-to-apples 비전 인코더 probe 없음**: M6 r2 가 Qwen
   + LLaVA-1.5 의 M2-stim 비전 활성화를 캡처했지만, LLaVA-Next /
   Idefics2 / InternVL3 는 M2-stim 비전 캡처 없음.
3. **v_L 추출 class-imbalance** (위 "Class-imbalance 한계" 참조). §4.6
   cross-model 은 LLaVA-1.5 만 M2 만으로 가능.

## 재현

```bash
# 5 cross-model M2 capture + label-free run (~22분 H200).
CUDA_VISIBLE_DEVICES=1 bash scripts/run_m2_cross_model_chain.sh

# per-model v_L 추출 (~1-2분).
uv run python scripts/m2_extract_per_model_steering.py

# Cross-model PMR / H1 / H2 분석 + figure.
uv run python scripts/m2_cross_model_analyze.py
```

## Artifacts

- `configs/cross_model_llava_next.py` (+ `_label_free.py`)
- `configs/cross_model_idefics2.py` (+ `_label_free.py`)
- `configs/cross_model_internvl3_capture.py`
- `scripts/run_m2_cross_model_chain.sh` (sequential driver)
- `scripts/m2_extract_per_model_steering.py` (per-model v_L)
- `scripts/m2_cross_model_analyze.py` (PMR / H1 / H2 + 그림)
- `outputs/m2_cross_model_summary/{per_label_pmr,per_object_level_pmr,h2_paired_delta}.csv`
- `outputs/cross_model_{llava_next,idefics2,internvl3}_capture_*/probing_steering/steering_vectors.npz`
- `docs/figures/m2_cross_model_{pmr_ladder,h1_ramp,h2_paired_delta}.png`
