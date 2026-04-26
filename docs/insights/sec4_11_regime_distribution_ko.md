---
section: §4.11
date: 2026-04-26
status: complete (4-모델 M8d)
hypothesis: H7 카테고리 follow-up — 라벨이 PMR (binary) 가 아닌 physics regime 을 선택
---

# §4.11 — 카테고리 H7 follow-up: 4-모델 M8d regime 분포

## 질문

M9 audit 가 H7 을 **PMR_phys − PMR_abs** 로 측정 — binary "모델이 physics-
mode 언어에 commit 했는가?" 대조. §4.11 가 더 세밀한 질문: **라벨이 어떤
physics regime 을 invoke 하는가?** 예 `person` stim 에서:

- 물리 라벨 `"person"` → 예상 kinetic (걷는다, 달린다, 점프한다)
- 추상 라벨 `"stick figure"` → 예상 static (그림, 묘사)
- exotic 라벨 `"statue"` → 예상 static AND abstract (정지, 조각상)

binary PMR 아래의 regime 분포가 라벨-conditioned commit 의 정성적 구조
포착.

## 방법

`classify_regime` (`src/physical_mode/metrics/pmr.py` 의, M8d 에서 car/
person/bird 카테고리-특정 키워드 셋으로 확장) 을 4-모델 M8d label-free
+ labeled run 모두에 적용. **horizontal-event** subset (M8d insight 가
가장 sharp 라고 언급한 곳) 의 (모델 × 카테고리 × label_role) 셀별 응답
의 {kinetic, static, abstract, ambiguous} 비율 계산.

모델: Qwen2.5-VL / LLaVA-1.5 / LLaVA-Next / Idefics2 (InternVL3 는 M8d
에 미실행).

## 결과

![§4.11 4-모델 M8d regime 분포](../figures/sec4_11_regime_distribution_4model.png)

*그림*: (모델 × 카테고리 × label_role) regime 비율. 각 열 = {no label,
physical label, abstract label, exotic label} 중 하나. 각 행 = 한 모델.
색 = regime (녹색=kinetic, 파랑=static, 빨강=abstract, 회색=ambiguous).
horizontal-event subset; 라벨 역할 셀당 n ≈ 40, _nolabel n ≈ 40.

### 헤드라인 읽기

1. **Qwen + Idefics2 는 모든 셀에서 saturated kinetic**. 보이는 예외만:
   Qwen 이 `person × exotic` (`statue`) 에서 ~30% static — `statue` 가
   exotic 해서 saturated SigLIP architecture 도 약간 양보. 그 외에는:
   ~95% kinetic 어디서나.

2. **LLaVA-1.5 가 가장 regime-discriminative 모델** (프로젝트 최강 H7
   M8d +0.31 와 일치):
   - `person × no label` 가 ~40% kinetic + 40% static + 20% ambiguous
     (라벨 없을 때 default regime 없음).
   - `person × physical` (`person` 라벨) 가 kinetic 을 ~62% 로 상승
     (라벨이 disambiguate).
   - `person × abstract` (`stick figure`) 가 kinetic 을 ~58% 로 낮추고
     ambiguous 상승 — LLaVA-1.5 가 "stick figure" 라벨을 mixed signal
     로 봄.
   - **`car × abs` (`silhouette`)**: kinetic 이 ~28% 로 떨어지고
     ambiguous ~70%. 라벨이 kinetic default 를 억제하는 가장 깔끔한 경우.
   - **`bird × abs` (`silhouette`)**: kinetic ~40%, ambiguous ~58%.
     같은 패턴.

3. **LLaVA-Next 는 부분적 regime 선택, 그러나 multi-axis 아키텍처 twist
   포함**:
   - `person × exotic` (`statue`) 가 ~30% kinetic + ~25% static + ~25%
     abstract 로 분해 — AnyRes + Mistral 조합이 LLaVA-1.5 가 보이지
     않는 3-way split 생성.
   - 다른 모든 셀은 ≥80% kinetic — LLaVA-Next 의 더 강한 시각 architecture
     가 대부분의 라벨-conditioning override.

4. **`person × abs` 의 cross-model 대조**: 라벨이 가장 약한 곳 (horizontal-
   event person stim 의 stick figure):
   - Qwen ~91% kinetic (saturated)
   - Idefics2 ~99% kinetic (saturated)
   - LLaVA-1.5 ~58% kinetic + ambiguous (라벨-discriminative)
   - LLaVA-Next ~80% kinetic + 15% static (intermediate)

   이 4-모델 gradient 가 M9 H7 finding 의 granular form: 라벨 하의 regime
   분포가 binary H7 가 가렸던 LM-modulation gradient 드러냄.

## 가설 함의

- **H7 (label-selects-regime)** — *binary 에서 categorical 로 upgrade*.
  LLaVA-1.5 가 regime 을 깔끔하게 선택 (물리 라벨엔 kinetic, 추상엔
  mixed). Qwen + Idefics2 는 saturated, 라벨에 둔감. LLaVA-Next 는
  intermediate. categorical 뷰가 commit 의 *종류* 설명, 단지 commit
  *여부* 가 아님.

- **H-LM-modulation** — *여전히 시사만*. LLaVA-Next person × exotic
  (3-way regime split) 가 LLaVA-1.5 person × abstract (kinetic +
  ambiguous) 와 정성적으로 다름, 그러나 양쪽 모두 multi axes 따라 다름.
  Multi-axis confound 지속.

- **Qwen 천장 설명** — regime 수준에서 확인. Qwen 이 원리적으로 regime
  granularity 부족하지 않음; 그저 car/person/bird stim 어떤 것에든
  "kinetic" default. exotic-only static 깜빡임 (statue → ~30% static)
  이 regime 분류기가 Qwen 의 잔여 변동 검출에 충분 sensitive 함 보임.

## 한계

1. **classify_regime 가 키워드 기반**, M8d insight doc 에 따른 5.6% 손
   라벨링 오차. 미묘한 regime 구분 (예: "rolls down" vs "tumbles") 가
   별도 추적 안 됨.
2. **InternVL3 행 없음**, M8d 미실행.
3. **셀당 n ≈ 40** 가 ±5 pp 변동이 noise 라기엔 작음. 헤드라인 의 ≥10 pp
   차이는 robust; 작은 것은 시사적.
4. **horizontal-event subset 만** (M8d insight 의 라벨 대조 가장 sharp 한
   곳). fall event subset 은 다른 regime 보일 것 (kinetic 에 "falls" /
   "drops" 더 많이).

## Reproducer

```bash
uv run python scripts/sec4_11_regime_distribution.py
```

출력:
- `docs/figures/sec4_11_regime_distribution_4model.png`
- `outputs/sec4_11_regime_distribution.csv` (long-form regime 비율)

## 산출물

- `scripts/sec4_11_regime_distribution.py` — 드라이버
- `docs/figures/sec4_11_regime_distribution_4model.png` — 4×3×4 stacked
  bar matrix
- `outputs/sec4_11_regime_distribution.csv` — 기반 수치
- `docs/insights/sec4_11_regime_distribution_ko.md` (이 문서)
