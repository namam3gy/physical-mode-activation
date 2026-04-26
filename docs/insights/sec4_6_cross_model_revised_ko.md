---
section: §4.6 cross-model — REVISED (2026-04-26 overnight followup)
date: 2026-04-26
status: complete (이전 "Qwen-scoped" 결론 수정)
hypothesis: 픽셀-인코드 가능성은 Qwen 한정 아님 — LLaVA-1.5 도 L25 에서 admits; 이전 null 은 wrong-layer-choice artifact
---

# §4.6 cross-model — REVISED: 픽셀-인코드 가능성은 Qwen 한정 아님

> **이 문서에서 쓰는 코드 한 줄 recap**
>
> - **§4.6** — VTI 역방향 counterfactual stim — `pixel_values` 위 픽셀-공간 gradient ascent 로 `<h_L[visual], v_L>` 최대화.
> - **H-shortcut** — Shortcut 해석은 이미지 자체에 인코드 가능. **여기서 "Qwen-scoped" → "model-conditional, Qwen 만이 아님"** 으로 수정.
> - **H-direction-specificity** — v_L 방향 픽셀-공간 gradient ascent 가 PMR 뒤집고 매칭 magnitude random 은 못함.
> - **M2 / M8a** — single-shape vs 5-shape factorial stim sets.
> - **v_L** — LM layer L 의 class-mean diff 방향; 모델별 dim 다름.

## TL;DR

이전 §4.6 cross-model 결론 ("LLaVA-1.5 self-v_L10 0/5 PMR flip → 픽셀-
인코드 가능성은 encoder-saturation 특이적 / Qwen-scoped") 은
**wrong-layer-choice artifact** 였음. LLaVA-1.5 에 layer sweep
(L5 / L15 / L20 / L25; L10 이 이전 테스트) 재실행:

| layer | bounded ε=0.05 | ε=0.1 | ε=0.2 | unconstrained | random Δ |
|---|---:|---:|---:|---:|---:|
| 5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/15 |
| 10 | 0/5 | 0/5 | 0/5 | 0/5 | 0/15 (이전 null) |
| 15 | 0/5 | 0/5 | 0/5 | 0/5 | 0/15 |
| 20 | 0/5 | 0/5 | 0/5 | 2/5 | 0/15 |
| **25** | **1/5** | **4/5** | **5/5** | 2/5 | **0/15** |

**L25 v_L25 가 깨끗한 픽셀-인코드 가능성** 보임: ε=0.1 에서 4/5 flip,
ε=0.2 에서 5/5. Random-direction control 모든 layer 에서 0/15 — 방향
특이성은 보존.

LLaVA-1.5 L25 ε=0.1 응답 샘플: "The circle will be hit by the dart."
L25 ε=0.2 샘플: "The circle will be hit by a ball."
Random ε=0.1 control: "The circle will be covered by the moon." (PMR=0)

## 무엇이 수정되었나

### 이전 reading (이제 obsolete)
> "regime axis 의 픽셀-인코드 가능성은 encoder-saturation 특이적 —
> Qwen 의 saturated SigLIP 이 LM 이 직접 읽는 thin pixel-to-L10 channel
> 만듦; LLaVA-1.5 의 unsaturated CLIP 은 그 채널이 없음. H-shortcut →
> Qwen-scoped."
>
> 출처: `docs/insights/sec4_6_cross_model.md` (commit `ec2aa77`).

### 수정된 reading (이 문서)
- **픽셀-인코드 가능성은 Qwen 모델-특이적이 아님.** LLaVA-1.5 도 보유,
  L10 대신 L25 에서.
- 이전 LLaVA-1.5 null (L10 에서 0/5) 은 **wrong-relative-depth**
  artifact: Qwen 은 28 LM layer (L10 ≈ 36% 깊이); LLaVA-1.5 는 32
  LM layer (L25 ≈ 78% 깊이). "shortcut layer" 가 architecture 별로
  다름.
- **H-shortcut** 은 supported 유지, 그러나 *causal-locus layer* 는
  모델별로 식별 필요 — single layer L10 은 Qwen-특이적이지
  cross-model 아님.

### 변경 없는 부분
- M9 PMR-천장과 §4.7 결정-안정성 천장은 *별개* saturation 시그니처 —
  §4.6 의 모델별 layer 선택과 무관. Architecture-level reframe 견고.
- v_L 방향 특이성 (encoding direction matters; random 은 안 함)
  LLaVA-1.5 의 모든 테스트 레이어에서 보존.

## 왜 중요한가

이건 paper-grade 수정. 원래 §4.6 cross-model claim ("Qwen-scoped
픽셀-인코드 가능성") 은 *over-strong*. 단순 layer sweep 이 픽셀-인코드
가능성이 **테스트한 5 architecture 중 적어도 2 개에 일반화** 함을
드러냄, 각 모델이 자기 자신의 "shortcut layer" 가짐.

함의:
- LLaVA-1.5 의 LM 을 통과하는 shortcut 경로는 Qwen 보다 *더 깊은
  relative depth* (~78% vs ~36% 깊이).
- `docs/insights/sec4_6_cross_model.md` 의 "third saturation signature"
  framing 은 incorrect — 픽셀-인코드 가능성은 encoder saturation 에
  엄격히 결합되지 않음. (M9 PMR-천장 + §4.7 결정-안정성은 여전히 그러함.)
- **Future work**: Qwen 에도 layer-sweep §4.6 — Qwen 도 L25 / L20 에서
  *두 번째* shortcut layer 가지는가? 현재 Qwen 은 L10 만 테스트.

## M2 vs M8a v_L cosine similarity (모델별, 5 layer 모두)

Class-imbalance 우려 (M2 는 saturated 모델에 n_neg = 1-9; M8a 는
n_neg = 100-280) 가 M8a captures 의 원래 동기. Cosine 분석은
class imbalance 가 *문제 아니었음* 을 보여줌:

| Model | Layer 5 | Layer 10 | Layer 15 | Layer 20 | Layer 25 |
|---|---:|---:|---:|---:|---:|
| LLaVA-Next | 0.40 | 0.39 | 0.42 | 0.33 | **0.25** (약) |
| Idefics2 | **0.79** | **0.79** | **0.80** | **0.79** | **0.79** |
| InternVL3 | **0.76** | 0.69 | **0.76** | **0.76** | 0.59 |

- Idefics2: 모든 레이어에서 **동일 방향** (cos ~0.79). M2 v_L 가
  n_neg=5 임에도 noise 아님 — saturated SigLIP+Mistral 에 대해
  class-imbalance robust.
- InternVL3: 대부분 동일 방향 (~0.7+), 일부 moderate alignment.
- LLaVA-Next: 대부분 레이어에서 moderate alignment (~0.4) —
  class-imbalance 가 부분적 영향; M8a v_L 가 M2 v_L 와 다소 다름.

**함의**: Saturated 모델 (Idefics2, InternVL3) 의 경우 M2-derived v_L
도 M8a-derived 와 같은 null 을 줄 것. *수정* 은 class balance 가
아님 — **layer 선택**. LLaVA-1.5 layer sweep 이 이를 직접 입증.

## 한계

1. **LLaVA-1.5 만 layer sweep**. 다른 3 cross-model 모델 (LLaVA-Next,
   Idefics2, InternVL3) 의 per-model gradient ascent 는 custom
   `counterfactual_<model>.py` 모듈 필요 (각각 다른 processor / pixel
   layout). 미구현. 그들의 "shortcut layer" 미상.
2. **Qwen layer sweep 부재**. Qwen 의 L10 5/5 finding 은 있지만
   L25 미테스트. "L10 이 Qwen 의 shortcut layer" reading 도 wrong-
   layer-choice 일 수 있음 — Qwen 의 진짜 shortcut layer 가 더 깊을
   수도.
3. **단일 방향 (class-mean v_L)**. SAE / multi-axis decomposition 으로
   다른 픽셀-인코드 가능 방향이 어느 layer 에든 추가로 있을 수 있음.
4. **Single-task 평가**. 다른 shortcut 행동은 다른 layer 에 localize
   될 수 있음.

## 재현

```bash
# Phase 1: M8a captures × 3 missing models (~50 min on H200).
CUDA_VISIBLE_DEVICES=1 bash scripts/run_overnight_sec4_6_followup.sh
```

## Artifacts

- `configs/encoder_swap_{llava_next,idefics2,internvl3}_m8a_capture.py`
- `outputs/encoder_swap_{llava_next,idefics2,internvl3}_m8a_capture_2026*/probing_steering/steering_vectors.npz`
- `outputs/sec4_6_counterfactual_llava_L{5,15,20,25}_2026*/`
- `outputs/sec4_6_followup/{comparison,llava_layer_sweep}.csv`

## LLaVA-1.5 L25 합성 응답 샘플

| Config | Sample 0 baseline | Sample 0 synth |
|---|---|---|
| ε=0.05 | "The circle will be filled in with color." | "The circle will be cut out of the paper." |
| ε=0.1 | "...filled in with color." | "**The circle will be hit by the dart.**" |
| ε=0.2 | "...filled in with color." | "**The circle will be hit by a ball.**" |
| unconstrained | "...filled in with color." | "**The circle will be hit by a ball.**" |
| random ε=0.1 | "...filled in with color." | "The circle will be covered by the moon." |

모델이 abstract (filled in with color) → physics-mode (hit by a ball
/ dart) 로 v_L25 perturbation 하에 이동. Random 은 physics-mode 에
도달하지 않음 (covered by the moon — celestial / static).
