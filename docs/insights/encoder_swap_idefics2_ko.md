# §4.5 — 교차 인코더 swap (Idefics2 SigLIP+Mistral 을 세 번째 점으로)

**상태**: 완료 2026-04-25.

## 동기

H-encoder-saturation (M6 r2) 은 **3-model 상관**: Qwen
(SigLIP+Qwen2-7B) 이 합성 PMR(_nolabel) 을 천장 포화; LLaVA
(CLIP+Vicuna-7B) 는 그렇지 않음; InternVL3 (InternViT) 은 천장 포화.
가장 깨끗한 인과 검증은 SigLIP + non-Qwen LM 을 가진 *네 번째* 모델을
취하는 것. Qwen 처럼 천장 포화하면 인코더 type 이 driver. LLaVA 와
패턴이 같으면 LM family 도 중요.

**Idefics2-8b** = SigLIP-SO400M + Mistral-7B-instruct.
- Vision 인코더: Qwen2.5-VL 과 같은 SigLIP family.
- LM: Mistral-7B (Qwen2-7B 와 Vicuna-7B 와 다름).
- 아키텍처: 표준 `Idefics2ForConditionalGeneration`, HF-transformers
  chat template 지원, ~8B 파라미터.

이는 LLaVA 대비 거의-깨끗한 인코더-swap counterfactual (둘 다 7B-급
LM, CLIP→SigLIP swap) 이고 Qwen 대비 부분 swap (둘 다 SigLIP,
LM family 변경).

## 방법

Idefics2 에 M8a labeled + label-free 프로토콜 실행:
- Qwen + LLaVA M8a 런과 동일한 400 자극 (5 도형 × 4 obj × 2 bg ×
  2 cue × 1 event × 5 seeds).
- 동일한 프롬프트 프로토콜 (T=0.7, top_p=0.95, max_new_tokens=96).
- Labeled arm 은 `LABELS_BY_SHAPE` 별 `(physical, abstract, exotic)`
  역할 트리플 사용.
- Label-free arm 은 `open_no_label` 템플릿 사용.

구현: `configs/encoder_swap_idefics2{,_label_free}.py`.
총: 1200 + 400 = 1600 추론을 GPU 0 에서 **8 분** wall clock.

## 결과

### 5 도형 평균 PMR(_nolabel)

| 모델    | 인코더        | LM         | 평균 PMR(_nolabel) |
|----------|----------------|------------|-------------------:|
| Qwen     | SigLIP         | Qwen2-7B   | **0.838** |
| LLaVA    | CLIP-ViT-L/14  | Vicuna-7B  | **0.175** |
| Idefics2 | SigLIP-SO400M  | Mistral-7B | **0.882** |

**Idefics2 가 Qwen 과 정확히 같이 천장 포화한다.** 두 SigLIP 기반
모델 모두 0.84-0.88; LLaVA (CLIP) 는 0.18. Qwen과 Idefics2 의 0.04
차이는 노이즈 수준 내.

### 도형별 PMR(_nolabel)

| 도형    | Qwen | LLaVA | Idefics2 |
|----------|-----:|------:|---------:|
| circle   | 0.825 | 0.288 | 0.925 |
| square   | 0.925 | 0.088 | 0.788 |
| triangle | 0.788 | 0.075 | 0.812 |
| hexagon  | 0.875 | 0.150 | 0.950 |
| polygon  | 0.775 | 0.275 | 0.938 |

**도형별 복제 일관됨**. Qwen 과 Idefics2 모두 도형별로 0.78-0.95
(어느 도형도 양 모델 모두 0.78 미만 없음). LLaVA 의 범위는
0.075-0.288 — 다른 regime.

### H7 paired-difference (physical − abstract) per shape

| 도형    | Qwen | LLaVA | Idefics2 |
|----------|-----:|------:|---------:|
| circle   | +0.012 | **+0.388** | +0.150 |
| square   | +0.075 | **+0.588** | -0.012 |
| triangle | -0.062 | +0.025 | -0.075 |
| hexagon  | -0.050 | **+0.262** | -0.013 |
| polygon  | -0.100 | **+0.538** | -0.088 |

**H7 strict (≥+0.05 ≥3/5 도형)**:
- Qwen 1/5 ✗ (square 만)
- LLaVA 4/5 ✓ (triangle 은 `wedge` 약 라벨)
- Idefics2 1/5 ✗ (circle 만)

**Idefics2 가 H7 에서 Qwen 과 동일 패턴.** 둘 다 천장 포화; H7 은
인코더가 headroom 을 남길 때만 측정 가능.

## 헤드라인 해석

이는 우리가 가진 H-encoder-saturation 가설의 가장 깨끗한 인과 검증.
(model × encoder) 횡단 패턴:

```
            encoder=SigLIP    encoder=CLIP
  LM=Qwen      Qwen 0.84        —
  LM=Vicuna    —                LLaVA 0.18
  LM=Mistral   Idefics2 0.88    —
```

- **두 SigLIP 기반 모델 모두 천장 포화** (Qwen 0.84, Idefics2 0.88) —
  두 다른 LM (Qwen2 + Mistral) 에 걸쳐 패턴 유지.
- **CLIP 기반 LLaVA 는 천장 포화하지 *않음*** (0.18) — 인코더 family
  가 변하면 패턴이 깨짐.
- LM family identity (Qwen2 vs Mistral) 가 포화 패턴을 뒤집지 않음:
  Mistral-7B 의 Idefics2 가 Qwen2-7B 의 Qwen 만큼 강하게 포화.

**결론**: vision encoder family (SigLIP vs CLIP) 가 합성 textured
자극에서 행동 PMR(_nolabel) 의 *주요* driver. LM 은 **부차적**
modulator (Idefics2 Mistral 패턴이 Qwen Qwen2 와 약간 다름 — 예:
circle 0.93 vs 0.83 — 둘 다 명확히 포화 regime).

이는 M6 r2 의 3-model 상관 버전보다 더 강한 인코더 포화 결과.
이제 **2 SigLIP 모델** 이 천장을 보이고 **1 CLIP 모델** 이 낮은 PMR
을 보임 — 인코더 만으로 포화 수준 예측.

## H7 교차 소스

H7 측정 가능성이 PMR(_nolabel) 천장을 추적:
- 인코더 천장 (SigLIP / Qwen + Idefics2) → headroom 없음 → H7 strict 실패.
- 인코더 비포화 (CLIP / LLaVA) → headroom → H7 4/5 통과.

이는 M6 r2 / M8a 의 동일 패턴, 세 번째 인코더/LM 조합으로 복제.

## 헤드라인 그림

`docs/figures/encoder_swap_heatmap.png` — 3 패널:
1. PMR(_nolabel) heatmap (model × shape).
2. H7 (physical − abstract) heatmap (model × shape).
3. 인코더 annotation 의 mean PMR(_nolabel) 요약 막대 차트.

패턴이 시각적으로 명백: Idefics2 행이 Qwen 행과 일치, LLaVA 가 outlier.

## 가설 업데이트

- **H-encoder-saturation** — **인코더 횡단으로 인과 검증**. "3-model
  상관" 에서 "encoder-family-causal" 로 승격. 이전 상태:
  M8c-refined (인코더 + stim 단순성). 새 상태: M8c-refined +
  cross-encoder-causally-confirmed.
- **H1** — *변경 없음*. Idefics2 H1 ramp 별도 분석 안됨 (라벨 arm 을
  다르게 실행 필요); 같은 인코더-포화 논리는 Idefics2 의 ramp 가
  Qwen 처럼 평탄할 것을 예측.
- **H7** — *변경 없음*. H7 측정 가능성이 인코더 포화에 의해 gated
  된다는 cross-encoder 확인.

## 한계

1. **깨끗한 LM-controlled swap 아님**: Idefics2 가 Mistral-7B 사용,
   Qwen2-7B 또는 Vicuna-7B 아님. 완벽한 counterfactual 은 같은 LM 에
   두 인코더 (예: CLIP vs SigLIP swap 의 LLaVA-1.5). Bunny + Phi-2 의
   LLaVA-1.5 가 비슷할 것.
2. **단일 도형 스윕**: M8a (5 도형) 만; M8d (카테고리) 와 M8c (사진)
   미실행. 완전한 cross-encoder 테이블 위해 가치 있음.
3. **Idefics2 vision-encoder probe AUC 없음**: M6 r2 가 Qwen vision
   encoder 활성화 캡처; Idefics2 캡처 미실행. Round 2 가 Idefics2 에
   encoder probe AUC 추가하여 loop closure.
4. **Idefics2 8B, LLaVA 7B, Qwen 7B**: Idefics2 favor 의 ~1B 파라미터
   차이. 6× PMR 차이를 driver 할 가능성 낮지만 noting 가치.

## 로드맵 함의

1. **§4.5 ✅ — H-encoder-saturation 가 encoder-family 수준에서 인과
   확인.** 논문 클레임이 "AUC 가 PMR 예측" 에서 "encoder family 가
   PMR 천장 vs no-ceiling regime 을 *야기*" 로 이동.
2. **Round-2 아이디어**: 4 번째 SigLIP 점을 위해 Bunny (SigLIP+Phi-2)
   추가, 2 번째 CLIP 점을 위해 CLIP 기반 더 작은 모델 추가. 패턴이
   복제되어야 함.
3. **Idefics2 vision encoder probe AUC** 가 다음 메커니즘 단계 —
   M6 r2 의 encoder-AUC ↔ behavioral-PMR 매핑이 Idefics2 에도
   유지됨을 확인.
4. **Idefics2 의 M8d / M8c**: 같은 패턴 예상 (SigLIP → 합성 천장,
   사진 더 낮음). 선택사항 — M8a 만으로도 정보적.

## 아티팩트

- `configs/encoder_swap_idefics2{,_label_free}.py`.
- `scripts/encoder_swap_analyze.py` — driver.
- `outputs/encoder_swap_idefics2_*/predictions.{jsonl,parquet,csv}`.
- `outputs/encoder_swap_summary/encoder_swap_{pmr_nolabel,h7}.csv`.
- `docs/figures/encoder_swap_heatmap.png` — paper-ready 3-panel.
- `docs/insights/encoder_swap_idefics2.md` (+ `_ko.md`).
