---
section: §4.7
date: 2026-04-26
status: complete (5-모델 M8a)
hypothesis: T=0.7 샘플링에서 어떤 입력 축 (object_level / bg_level / cue_level) 이 모델 결정을 안정화 하는가?
---

# §4.7 — M8a 의 per-axis 결정 안정성 (RC)

## 재구성

Pilot 가 T=0 에서 RC 측정 불가 (모든 응답 동일, RC=1). T=0.7 (M8a 설정)
하에서 RC 가 5 seed 횡단 cell 내 안정성 측정. §4.7 가 묻는 것: **어떤
입력 축이 가장 강한 결정 안정자인가?** 그리고 5 architecture 횡단으로
다른가?

## 방법

각 (모델 × 도형 × object_level × bg_level × cue_level) 셀, n_seeds = 5
에 대해 RC = max(count(pmr=v)) / 5 over v ∈ {0, 1} 계산. RC = 1 은 5 seed
모두 PMR 동의; RC = 0.6 은 3-of-5 majority.

축당, 그 축이 주어진 level 을 가지는 모든 셀 횡단으로 RC 평균 (예:
`cue_level=both` 인 모든 셀의 RC 평균). 5 모델, M8a label-free arm 만
(모델당 n=400 = 80 셀 × 5 seed).

## 결과

![§4.7 5-모델 per-axis RC](../figures/sec4_7_rc_per_axis.png)

*그림*: (모델 × 축 × level) 평균 RC. Error bar = 셀 횡단 std. 높은 RC
= seed 횡단 더 결정-안정적.

### 헤드라인 읽기

1. **`cue_level=both` 가 지배적 결정 안정자** (9–16 pp 이득), 3 saturated
   모델에서:
   - Qwen2.5-VL: cue=none 0.84 → cue=both **1.00** (+0.16) — cue 하의
     완벽한 일관성.
   - Idefics2: 0.88 → 0.99 (+0.11)
   - InternVL3: 0.89 → 0.98 (+0.09)

   2 less-saturated 모델에서 cue 효과 반전 또는 사라짐:
   - LLaVA-1.5: 0.85 → 0.85 (변화 없음)
   - LLaVA-Next: 0.78 → 0.78 (변화 없음)

2. **`bg_level=ground` 가 보조 안정자** (3–8 pp), saturated 모델에서:
   - Qwen: blank 0.88 → ground 0.96 (+0.08)
   - Idefics2: 0.92 → 0.95
   - InternVL3: 0.92 → 0.96

   LLaVA-1.5 에서 반전 (0.88 → 0.82, **−0.06**); LLaVA-Next 에서 거의
   움직이지 않음 (0.77 → 0.80).

3. **`object_level` 가 가장 약한 안정자**, 모델별 패턴:
   - Saturated 모델 line / filled / shaded / textured 대체로 평탄
     (Qwen 0.90–0.95, Idefics2 0.92–0.96, InternVL3 0.91–1.00 with
     textured = perfect).
   - LLaVA-1.5 가 low-abstraction 선호 (line 0.90 / filled 0.88) over
     high (shaded 0.81 / textured 0.80).
   - LLaVA-Next 는 U-shape (line 0.69, shaded 0.89, textured 0.78).

### 해석

Saturated 모델 (SigLIP / SigLIP-SO400M / InternViT) 이 contextual cue
(motion arrow + ground plane) 가 있을 때 자신감 있게 commit. 이 시각 cue
가 fire 하면 모델이 모든 5 seed 횡단으로 같은 PMR call 로 수렴.
**Saturation 은 단지 행동 PMR ceiling 만이 아니라 결정-안정성 ceiling
이기도 함**.

LLaVA-1.5 와 LLaVA-Next (CLIP encoder) 는 강한 cue 하에서도 결정 noise
보유. 이 모델들의 cue_level 효과는 RC 에서 본질적으로 0, CLIP+LM
pipeline 이 cue 를 non-CLIP architecture 와 같은 자신감으로 physics-mode
와 abstract-mode 사이 disambiguate 에 사용 안 함을 시사.

이는 architecture-level reframe 의 별도 시그니처: **stim-y AUC = 1.0
architecture 횡단 (encoder discriminability), 그러나 cue 하의 결정 안정성
이 sharply 변동 (joint architecture 효과)**.

## 한계

1. **n_seeds = 5** 가 RC 의 최소; CI 가 넓음. ≥10 pp 차이 (saturated
   모델의 cue 효과) 는 robust; 3–8 pp bg_level 효과는 시사적.
2. **단일 arm (label-free)**. 라벨 자체가 commit 안정화 하므로 labeled
   arm 은 다른 RC 구조 보일 수 있음.
3. **PMR 이 binary**. binary 결과의 RC 가 1.0 까지 빠르게 plateau;
   saturated 모델에는 regime-RC (regime 분류에서의 일관성) 가 더
   sensitive metric.
4. **부트스트랩 CI 없음**. Error bar 가 within-cell std, 모델-level
   재샘플링 CI 가 아님. 엄격한 통계 아님.

## 가설 함의

- **H-encoder-saturation 확장**: saturation 이 PMR-ceiling AND 결정-안
  정성-ceiling 으로 manifest. 둘 관련 — encoder 가 "physics-mode" stim
  에 자신감 있게 fire 하면 LM 이 seed-level 변동 없이 commit.
- **H1 (ramp)**: object_level 이 약한 RC 패턴 보임. Ramp 가 PMR-방향
  (낮은 평균 PMR 에서 높은 쪽으로), RC-방향 아님 (saturated 모델에서
  대부분 셀이 이미 high-RC).

## Reproducer

```bash
uv run python scripts/sec4_7_rc_per_axis.py
```

출력:
- `outputs/sec4_7_rc_per_axis.csv` — (모델 × 축 × level) 평균/std RC
- `docs/figures/sec4_7_rc_per_axis.png` — 3-패널 bar chart

## 산출물

- `scripts/sec4_7_rc_per_axis.py` — 드라이버
- `docs/figures/sec4_7_rc_per_axis.png` — 5-모델 × 3-축 figure
- `outputs/sec4_7_rc_per_axis.csv` — 기반 수치
- `docs/insights/sec4_7_rc_per_axis_ko.md` (이 문서)
