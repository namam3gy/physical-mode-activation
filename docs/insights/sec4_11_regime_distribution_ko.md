---
section: §4.11
date: 2026-04-26
status: complete (5-모델 M8d)
hypothesis: H7 카테고리 follow-up — 라벨이 PMR (binary) 가 아닌 physics regime 을 선택
---

# §4.11 — 카테고리 H7 follow-up: 5-모델 M8d regime 분포

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택 (ball → 동적 / circle → 정적 / planet → 궤도).
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M8a** — Stim 다양화 — 비-원 합성 shape (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8d** — Stim 다양화 — 비-공 물리 객체 카테고리 (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M9** — Generalization audit — 논문 Table 1 (3 model × 3 stim 소스 × bootstrap CIs, 5000 iter); PASS/FAIL 이진화를 CI 분리로 대체.

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

모델: Qwen2.5-VL / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3
(InternVL3 는 2026-04-26 추가, 5-모델 갭 닫음).

## 결과

![§4.11 5-모델 M8d regime 분포](../figures/sec4_11_regime_distribution_5model.png)

*그림*: (모델 × 카테고리 × label_role) regime 비율. 각 열 = {no label,
physical label, abstract label, exotic label} 중 하나. 각 행 = 한 모델.
색 = regime (녹색=kinetic, 파랑=static, 빨강=abstract, 회색=ambiguous).
horizontal-event subset; 라벨 역할 셀당 n ≈ 40, _nolabel n ≈ 40.

### 헤드라인 읽기

1. **Qwen + Idefics2 + InternVL3 는 대부분 셀에서 saturated kinetic**.
   주목할 예외:
   - Qwen `person × exotic` (statue): ~30% static
   - **InternVL3 `person × exotic` (statue): ~65% static** — 프로젝트
     에서 가장 강한 단일 라벨-driven static commit. statue 라벨이 30%
     kinetic / 65% static split 을 disambiguate, car/bird × exotic 은
     InternVL3 에서 ≥90% kinetic 유지.
   - 그 외에는 3 saturated encoder 모두에서 ≥95% kinetic 어디서나.

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

4. **`person × abs` 의 cross-model 대조** (horizontal-event person stim
   의 stick figure):
   - Qwen ~91% kinetic (saturated)
   - Idefics2 ~99% kinetic (saturated)
   - InternVL3 ~99% kinetic (saturated, Idefics2 와 유사)
   - LLaVA-Next ~80% kinetic + 15% static (intermediate)
   - LLaVA-1.5 ~58% kinetic + ambiguous (라벨-discriminative)

   이 5-모델 gradient 가 M9 H7 finding 의 granular form: 라벨 하의 regime
   분포가 binary H7 가 가렸던 LM-modulation gradient 드러냄.

5. **`person × exotic` 의 InternVL3 cross-model 대조** (statue):
   30% kinetic + 65% static split 가 흥미로움, InternVL3 가 그 외에는
   강하게 saturated 모델 (M8a PMR 0.92, 행동이 Qwen + Idefics2 와 매치).
   `statue` 에 강한 반응은 라벨이 uniquely disambiguate 할 때 (statue 가
   진짜로 움직이지 않는 entity), saturated-encoder architecture 도 언어
   신호에 deferred 보임. 라벨-disambiguation channel 이 모든 architecture
   에 존재; 단지 대부분 조건에서 dormant.

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
2. **셀당 n ≈ 40** 가 ±5 pp 변동이 noise 라기엔 작음. 헤드라인 의 ≥10 pp
   차이는 robust; 작은 것은 시사적.
3. **horizontal-event subset 만** (M8d insight 의 라벨 대조 가장 sharp 한
   곳). fall event subset 은 다른 regime 보일 것 (kinetic 에 "falls" /
   "drops" 더 많이).
4. **5-카테고리 fine-grained regime** (gravity-fall / gravity-roll /
   orbital / inertial / static) M2 circle-only 데이터에 여전히 열림.
   Circle-shape regime 별 신규 키워드 셋 필요.

## Reproducer

```bash
uv run python scripts/sec4_11_regime_distribution.py
```

출력:
![sec4_11_regime_distribution_4model](../figures/sec4_11_regime_distribution_4model.png)
- `outputs/sec4_11_regime_distribution.csv` (long-form regime 비율)

## 산출물

- `scripts/sec4_11_regime_distribution.py` — 드라이버
![5×3×4 stacked bar matrix (4model 대체)](../figures/sec4_11_regime_distribution_5model.png)
- `outputs/sec4_11_regime_distribution.csv` — 기반 수치
- `configs/encoder_swap_internvl3_m8d{,_label_free}.py` — InternVL3 M8d
  config (2026-04-26 추가, 5-모델 갭 닫음)
- `docs/insights/sec4_11_regime_distribution_ko.md` (이 문서)
