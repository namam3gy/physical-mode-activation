---
section: §4.6 cross-model
date: 2026-04-26
status: complete (transfer test 4 모델 + LLaVA-1.5 per-model gradient ascent)
hypothesis: v_L10 의 픽셀-인코드 가능성은 encoder-saturation 특이적 — Qwen 결과는 cross-architecturally 일반화하지 않음
---

# §4.6 cross-model — Qwen 한정 결과, encoder-saturation 특이적

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H-encoder-saturation** — 합성 stim 위 행동 PMR(_nolabel) saturation 은 architecture 수준 (encoder + LM 결합) 에서 결정.
> - **H-direction-specificity** — v_L10 방향 픽셀-공간 gradient ascent 는 Qwen2.5-VL 의 PMR 을 뒤집음; 매칭 magnitude random direction 은 그러지 못함 (§4.6).
> - **H-shortcut** — Shortcut 해석은 이미지 자체에 인코드 가능 (§4.6) — 픽셀 기반.
> - **M2** — ST1 MVP-full — 5-axis factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M5a** — ST4 VTI steering — LM L10 시각 토큰에 +α·v_L10 더하면 line/blank/none 이 "정지" → physics-mode 로 뒤집힘.
> - **§4.6** — VTI 역방향 counterfactual stim — Qwen2.5-VL pixel_values 위 픽셀-공간 gradient ascent.
> - **v_L10** — Layer 10 LM hidden space 의 steering 방향; 모델별 dim 3584 / 4096; M2 captures 가 cross-model class-mean diff 에 사용됨.

## 질문

§4.6 (Qwen 만) 은 v_L10 방향 픽셀-공간 gradient ascent 가 5 baseline
원 stim 의 5/5 를 ε = 0.05 에서 PMR flip 시킨다는 것을 보였다 (random
direction 0/15). **이 픽셀-인코드 가능성이 다른 VLM 으로 일반화되는
가?** 두 sub-질문:

1. **Transfer test**: Qwen 의 합성 stim 이 다른 4 VLM 의 PMR 을
   뒤집는가?
2. **Per-model gradient ascent**: 각 모델이 자기 자신의 v_L10 (M2
   captures 에서 추출) 으로 픽셀-공간 gradient ascent 했을 때 PMR 이
   뒤집히는가?

## 방법

**Transfer test** (`scripts/sec4_6_cross_model_transfer.py`):
기존 §4.6 sweep manifest 의 각 (sample_id × config) 쌍에서
`baseline.png` 와 `synthesized.png` 를 동일 `"What will happen to the
circle in the next moment?"` 프롬프트로 각 비-Qwen 모델에 입력, PMR
채점. 4 모델 × 35 row × 2 inferences = 280 호출.

**Per-model gradient ascent** (LLaVA-1.5 만): 동일 Adam-기반 gradient
ascent (lr=1e-2, n_steps=200, ε ∈ {0.05, 0.1, 0.2, unconstrained},
3 random control @ ε=0.1) 를 LLaVA-1.5 의 pixel_values 에 LLaVA-1.5
자신의 v_L10 을 타겟으로 적용. 신규 모듈
`src/physical_mode/synthesis/counterfactual_llava.py` 가 표준 CLIP
`(1, 3, 336, 336)` layout 처리 (Qwen 의 patch-flattened
`(T, 1176)` layout 과 다름).

LLaVA-1.5 가 M2 captures 에서 클래스 균형 잡힌 깨끗한 v_L10 (n_pos=375 /
n_neg=105) 을 가진 유일한 비-Qwen 모델. LLaVA-Next / Idefics2 /
InternVL3 는 n_neg = 9 / 5 / 1 — class-mean diff 하기에 M2 에서 너무
포화. 이 모델들의 per-model gradient ascent 는 더 어려운 stim
(M8a / M8c) 으로 향후 라운드에 보류.

## 결과

### Transfer test

![Qwen §4.6 panel — transfer source](../figures/sec4_6_counterfactual_stim_panels.png)

*그림: Qwen 의 §4.6 합성 stim 을 transfer source 로 사용. 각 열:
baseline → bounded ε=0.05 → ε=0.1 → unconstrained.*

| Model | Config | n flipped | baseline_pmr | synth_pmr |
|---|---|---:|---:|---:|
| LLaVA-1.5 | bounded_eps0.05 | 0 / 5 | 0.0 | 0.0 |
| LLaVA-1.5 | bounded_eps0.10 | 0 / 5 | 0.0 | 0.0 |
| LLaVA-1.5 | bounded_eps0.20 | 0 / 5 | 0.0 | 0.0 |
| LLaVA-1.5 | unconstrained | 0 / 5 | 0.0 | 0.0 |
| LLaVA-Next | (모든 config) | 0 / 5 each | 0.0 | 0.0 |
| Idefics2 | (모든 config) | 0 / 5 each | 0.0 | 0.0 |
| InternVL3 | bounded_eps0.05 | 0 / 5 | 1.0 | 1.0 |
| InternVL3 | bounded_eps0.10 | 0 / 5 | 1.0 | 1.0 |
| InternVL3 | bounded_eps0.20 | 0 / 5 | 1.0 | **0.6** |
| InternVL3 | unconstrained | 0 / 5 | 1.0 | **0.2** |

**Headlines**:

1. **어떤 모델도 Qwen 의 adversarial 로 0→1 flip 안 됨.** LLaVA-1.5,
   LLaVA-Next, Idefics2 모두 §4.6 프롬프트 하에서 PMR = 0 으로
   baseline; synth 가 그것을 바꾸지 않음. Qwen-derived adversarial 은
   cross-model 로 transfer 안 됨 — 모델-특이적 adversarial perturbation
   문헌과 일치.
2. **InternVL3 는 큰 ε 에서 음의 transfer.** Baseline 이 PMR = 1.0 으로
   포화; ε=0.2 perturbation 이 synth PMR 을 0.6 으로 떨어뜨리고,
   unconstrained 는 0.2 로 떨어뜨림. 큰 Qwen-derived perturbation 이
   InternVL3 의 saturated physics-mode 를 trigger 하거나 보존하는 대신
   *교란*. Bounded-small-ε perturbation (0.05, 0.1) 은 InternVL3 에
   비가시적.

### LLaVA-1.5 per-model gradient ascent

![LLaVA-1.5 §4.6 panel — 자기 자신의 v_L10 으로 gradient ascent](../figures/sec4_6_counterfactual_stim_panels_llava.png)

*그림: LLaVA-1.5 per-model gradient ascent. baseline → ε=0.05 →
ε=0.1 → unconstrained. 시각적 특성은 Qwen 과 유사 — 작은 ε 에서
abstract gestalt 보존.*

| Config | n | Baseline PMR mean | Synth PMR mean | n flipped | 평균 final projection |
|---|--:|------------------:|---------------:|----------:|----------------------:|
| `bounded_eps0.05` | 5 | 0.0 | 0.0 | **0** | 150.7 |
| `bounded_eps0.10` | 5 | 0.0 | 0.0 | **0** | 176.7 |
| `bounded_eps0.20` | 5 | 0.0 | 0.0 | **0** | 190.5 |
| `unconstrained` | 5 | 0.0 | 0.0 | **0** | 197.9 |
| `control_v_random_*` | 15 | 0.0 | 0.0 | **0** | 2.4–9.8 |

![LLaVA-1.5 §4.6 trajectory](../figures/sec4_6_counterfactual_stim_trajectory_llava.png)

*그림: LLaVA-1.5 config 별 projection trajectory. Gradient ascent 가
타겟 projection 을 성공적으로 최대화 (v_L10 config 에서 8 → 150-200,
random control 에서 0 → 2-10), 그러나 PMR flip 안 함.*

**Headline**: 모든 ε 에서 0/5 flip; gradient ascent 는 타겟 projection
을 성공적으로 최대화 (8 → 150-200, Qwen 의 43-180 와 비슷), 하지만
**LM 의 행동 출력은 변하지 않음**. 이게 gradient-flow / 파이프라인
이슈가 *아님* — projection 은 진정으로 상승. LLaVA-1.5 의 L10 hidden
state 는 class-mean diff 로 "v_L10 방향" 을 가지지만, 그 방향 위
projection 을 최대화하는 것이 Qwen 처럼 **행동 출력을 뒤집지
않는다**.

Random-direction control 도 flip 안 함 (Qwen 과 일관) 그리고 훨씬
작은 projection (2-10) 에 도달, 따라서 v_L10 vs random 방향 특이성
메커니즘은 projection 수준에서 유지됨. 분리는 **projection 과 행동
사이**, v_L10 과 다른 방향 사이 아님.

## Mechanism

H-shortcut 가설에 두 가지 신규 제약:

1. **H-shortcut 은 encoder-saturation 특이적** (수정). §4.6 이 Qwen
   에 시연한 픽셀-인코드 가능 shortcut 경로는 Qwen 의 *포화된* SigLIP
   encoder + Qwen2-7B LM 이 LM 이 직접 읽는 얇은 픽셀-to-L10 채널을
   만들기 *때문에* 존재. LLaVA-1.5 의 포화되지 않은 CLIP-ViT-L +
   Vicuna 는 같은 채널을 가지지 않음 — class-mean diff 로 측정한 L10
   표현 차이가 픽셀 공간을 통해 도달했을 때 행동 flip 으로 번역되지
   않음.

2. **H-direction-specificity 는 Qwen-scoped** (수정). Qwen 에서는
   v_L10 이 projection-수준 *과* 행동-수준 모두에서 특이성을 가짐
   (5/5 v_L10 flip, 0/15 random). LLaVA-1.5 에서는 projection-수준
   특이성은 보존 (v_L10 이 projection ~180, random 이 ~5 도달) 되지만
   행동-수준 특이성은 무너짐 (둘 다 0/5 flip). "v_L10 은 행동을
   뒤집는 축이다" 라는 읽기는 Qwen-특이적이었음.

함께, 이 수정들은 §4.6 결과를 **architecture-특이적인 픽셀 기반
shortcut 증거** 로 만듦, generic VLM 속성이 아님. 원래의 H-shortcut
framing ("shortcut 이 이미지 자체에 인코드 가능") 은 "포화된
architecture 에 대해" 단서가 필요.

이는 M2 / M8a / M9 의 architecture-level reframe 과 완전히 일관:
PMR 천장은 encoder-saturation-driven; regime axis 의 픽셀 인코드
가능성도 *역시* encoder-saturation-driven.

## 가설 함의

- **H-shortcut**: 수정 — Qwen-scoped. H-locus (L10 특이적),
  H-direction-bidirectional (regime axis), H-direction-specificity
  (5/5 vs 0/15 random) 와 함께 세 번째 Qwen-특이적 발견으로 문서화.
- **H-encoder-saturation**: 확장 — saturation 은 픽셀-인코드 가능
  shortcut 의 전제. 두 가지 distinct 시그니처: 행동 PMR 천장 (M9),
  결정-안정성 천장 (§4.7), 픽셀 인코드 가능성 (이 라운드).
- **신규 가설 없음**: cross-model null 은 saturation-specific 읽기 하
  예측되는 결과.

## 한계

1. **단일 비-Qwen per-model 테스트 (LLaVA-1.5 만).** LLaVA-Next /
   Idefics2 / InternVL3 per-model gradient ascent 는 보류 — M2 의
   v_L10 이 class-imbalanced. encoder-saturation-specific 읽기는 모두
   자기 자신의 v_L10 에서 같은 null 을 보일 것이라 예측, 그러나 이는
   현재 예측이지 검증 아님.
2. **단일 layer (L10) 만 LLaVA-1.5 에서 테스트.** LLaVA-1.5 는 32 LM
   layer 보유 (Qwen 의 28 와 다름), L10 이 같은 relative depth 가 아님;
   "physics-mode layer" 가 다를 수 있음. Layer-sweep 은 열림.
3. **단일 방향 (M2 의 v_L10) 만 테스트.** SAE feature 또는 multi-axis
   분해는 LLaVA-1.5 에서도 다른 픽셀-인코드 가능 방향을 찾을 수 있음.
4. **단일 프롬프트** ("What will happen to the circle?"). 4/4 비-Qwen
   모델에서 §4.6 프롬프트 하 baseline PMR=0 은 프롬프트 자체가
   필터링한다는 것을 시사 — "circle" 없는 open-prompt 는 다른
   baseline 보일 수 있음.

## 재현

```bash
# Transfer test (5 baseline × 4 모델 × 7 config × 2 stim).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_cross_model_transfer.py \
    --run-dir outputs/sec4_6_counterfactual_20260426-050343 --device cuda:0

# LLaVA-1.5 per-model gradient ascent (~10분).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_counterfactual_stim_llava.py \
    --device cuda:0

# LLaVA-1.5 PMR re-inference + figure.
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_counterfactual_llava_<ts> \
    --model-id llava-hf/llava-1.5-7b-hf \
    --device cuda:0
```

## Artifacts

- `scripts/sec4_6_cross_model_transfer.py` — transfer test driver
- `scripts/sec4_6_counterfactual_stim_llava.py` — LLaVA-1.5 §4.6 driver
- `src/physical_mode/synthesis/counterfactual_llava.py` — 표준 CLIP
  변형 (pixel_values_from_pil / reconstruct_pil /
  prepare_inputs_for_grad / gradient_ascent)
- `outputs/sec4_6_cross_model_transfer_*/results.csv`
- `outputs/sec4_6_counterfactual_llava_20260426-114111/` — LLaVA-1.5
  sweep + manifest + per-config 합성 PNG
- `docs/figures/sec4_6_counterfactual_stim_panels_llava.png`
- `docs/figures/sec4_6_counterfactual_stim_trajectory_llava.png`
