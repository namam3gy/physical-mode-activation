# 원이 공으로 보이는 순간 — 종합 검토 슬라이드 노트 (2026-05-01)

> 본 문서는 `docs/review_ppt/physical_mode_full_review_ko.pptx` (30 슬라이드)의
> 동반 자료입니다. 기존 26-슬라이드 storyline 위에 4개의 새 슬라이드 (M-LMSwap
> Pillar B 진행 + post-projection SAE round 2 + scoring artifact + 결정 시점)
> 가 슬라이드 24-27 자리에 삽입되었고, 기존 closing 3장은 28-30 으로
> 재번호화되었습니다.
>
> 슬라이드 1-23, 28-30 의 자세한 해설은 `physical_mode_storyline_ko.md`
> (기존 26-슬라이드 노트) 와 동일하므로, 이 문서는 **새로 추가된 4 슬라이드
> (24-27) 와 종합 흐름의 변화** 만 자세히 풀어 적습니다.

---

## 흐름 변화 요약 (1 분 발표)

기존 26-슬라이드 storyline 은 M0–M9 + §4.5/4.6/4.8 + M5b SAE intervention
cross-model round 1 까지 다뤘습니다. 이 종합 검토 deck 은 **2026-04-29 →
2026-05-01 사이에 새로 진행된 작업** 4 개를 슬라이드 24-27 로 추가합니다:

1. **슬라이드 24** — Pillar B 의 동기. LLaVA-1.5 vs LLaVA-Next 의 4-axis
   confound 가 \"LM 결정자\" 결론을 막는다는 점을 정리하고, M-LMSwap 의 단일축
   통제 설계를 제시.
2. **슬라이드 25** — M-LMSwap Variant A (CLIP+Vicuna) 학습 21K 완료, 그러나
   regression eval gate 실패 (PMR 0.825). 진단 결과 axis ordering 이 보존되어
   A↔B 비교는 살아있음.
3. **슬라이드 26** — Post-projection SAE round 2. 5 모델 × 4 cell 표로
   regime-cross capacity ladder 를 보임. \"NULL\" 의 3 가지 의미를 분리.
4. **슬라이드 27** — Scoring artifact (binary PMR 가 regime-shift 를 가린다)
   + 결정 시점 (A1: B 진행 / A2: A 재학습 / A3: M-PSwap 부활).

기존 28-30 (한계, 결론, 감사) 은 본 새 진행 사항을 반영하도록 일부 문구만
조정했고, 슬라이드 1-23 은 변경 없습니다.

---

## 슬라이드 1-23 — 기존 storyline 그대로

기존 `physical_mode_storyline_ko.md` 의 슬라이드 1-23 해설을 그대로
사용하시면 됩니다. (표지 → 놀라운 관찰 → 3축 질문 → 추상화 사다리 → 5축
factorial → 5개 모델 → PMR 사다리 → encoder boomerang → M9 외부 타당성 →
VTI L10 → SAE encoder ablation → MLP knockout → 픽셀 인코딩 → cross-model
픽셀 routability → M8 외부 타당성 → scaling/multilingual → 5겹 시그니처)

각 슬라이드의 재인용은 `physical_mode_storyline_ko.md` 참조.

---

## 슬라이드 24 — Pillar B 동기: 왜 controlled LM swap 이 필요한가

**한 문장**: LLaVA-1.5 (PMR 0.18) → LLaVA-Next (0.79) 라는 거대한 PMR 점프는
\"LM 이 결정자\" 라고 결론 내릴 만큼 깨끗한 비교가 아닙니다 — 4개의 변인이
동시에 바뀌고 있기 때문입니다.

- **왼쪽 (붉은 박스) — LLaVA-1.5 ↔ LLaVA-Next 의 4-axis confound**:
  1. **LM 백본 자체**: Vicuna-7B → Mistral-7B-Instruct.
  2. **시각 토큰 처리 방식**: 단일-tile 576 토큰 → AnyRes 다중 grid (해상도
     adaptive, 최대 4×4 patch). 토큰 개수도 다르고 grid 처리 코드 경로도 다름.
  3. **SFT 데이터셋**: LLaVA-Instruct-150K → LLaVA-Next 760K.
  4. **Vision-language 정렬 절차**: LLaVA-1.5 는 단일 stage projector 정렬,
     LLaVA-Next 는 2-stage refresh.
  - 따라서 \"인코더만 동일\" 한 가지로는 LM identity, AnyRes, SFT 데이터,
    정렬 recipe 중 어느 것이 PMR 점프의 원인인지 분리할 수 없음.

- **오른쪽 (파란 박스) — M-LMSwap 의 controlled 설계**:
  - 공통: CLIP-ViT-L-336 인코더 + 2-layer MLP projector (랜덤 초기화).
  - Variant A: + Vicuna-7B-v1.5.
  - Variant B: + Mistral-7B-Instruct-v0.2.
  - LoRA(r=32, α=64) 로 LM 의 q/k/v/o_proj 만 학습.
  - Stage 1: projector pretrain (LCS-558K, 17K steps).
  - Stage 2: instruction tune on LLaVA-Instruct-665K, 21K steps, LoRA on LM.
  - 결과: **단일축 (LM identity 만)** 통제. AnyRes 등은 모두 끔.

- **하단 주의** — LLaVA family 가 M5b post-proj 에서 NULL 이 나왔다는 사실
  자체는, \"인코더에 physics-mode commitment 가 없다\" 와 \"LM 이 결정자\"
  두 가지 해석을 동시에 허용. Pillar B 는 후자를 직접 검증하는 단계.

**발표 팁**: 청중이 LLaVA-1.5 vs LLaVA-Next 결과를 \"LM 이 결정자\" 로 받아
들이려는 순간, 이 슬라이드를 \"잠시만요\" 로 시작해서 4-axis confound 를
보여주면 메시지가 분명해집니다.

---

## 슬라이드 25 — M-LMSwap Variant A 학습 완료 + Variant B 게이트 대기

**한 문장**: Vicuna 백본 학습 21K step 까지 완료되었으나 regression eval
게이트는 PMR 천장이 너무 높아 실패. 그러나 axis ordering 은 LLaVA-1.5 와
동일하게 보존되어 A↔B Δ-PMR 비교는 가능.

- **상단 — 학습 타임라인** (4 단계):
  1. Stage 1: projector pretrain. LCS-558K, 17K step. ~12 시간 H200.
  2. Stage 2: instruction tune. LLaVA-Instruct-665K, 21K step. ~12 시간.
  3. Final ckpt: step21000. MLP weight + LoRA adapter merged.
  4. Regression eval (gate): PMR_nolabel + baseline cell + generation sanity.

- **왼쪽 (붉은 박스) — Regression eval (step9000) gate FAIL**:
  - Gate 1 (generation sanity, > 5 단어, no degeneracy): PASS.
  - Gate 2 (PMR_nolabel ∈ [0.03, 0.50]): **FAIL** (0.825).
  - Gate 3 (line/blank/none baseline ≤ 0.6): **FAIL** (1.000).
  - 해석: A 의 PMR 천장이 LLaVA-1.5 (0.18) 보다 훨씬 높아서, \"LLaVA-1.5
    floor 를 replicate\" 라는 원래 목적은 실패. Vicuna+CLIP recipe 가
    LLaVA-1.5 의 학습 분포와 다른 곳으로 수렴했다는 뜻.

- **오른쪽 (파란 박스) — A↔B 비교 가능성은 살아있다**:
  - 응답 다양성: 480 stim 중 201 unique 응답. 즉 image-blind 가 아니라
    이미지에 따라 다양한 텍스트를 생성. (만약 image-blind 라면 unique 가
    훨씬 적었을 것).
  - 축 ordering 보존: line 0.483 < filled 0.908 ≤ shaded 1.000. LLaVA-1.5
    의 ordering 과 동일하고, 단지 +0.4 만큼 위로 shift 된 형태.
  - 결론: A 는 LLaVA-1.5 \"floor\" replicate 는 실패했지만, **모델 내부의
    PMR ordering 은 살아있어서** A↔B Δ-PMR 비교 (\"같은 입력에서 두 LM 이
    얼마나 다른가\") 자체는 유의미할 수 있다.

- **하단** — 결정 대기. step21000 final ckpt 의 regression 재실행 결과를
  본 다음 (B6 작업) A1 vs A2 결정. 슬라이드 27 참고.

**발표 팁**: 게이트 fail 사실을 숨기지 말고 정직하게 보여준 뒤, axis
ordering 이라는 \"살아있는 시그널\" 로 회복 가능한 그림을 그리는 것이
핵심. \"Recipe 가 한 번 실패해도 완전 폐기는 아니다\" 라는 메시지.

---

## 슬라이드 26 — Post-projection SAE round 2: regime-cross capacity ladder

**한 문장**: Encoder-side SAE intervention round 1 에서 LLaVA family 가 NULL
로 나왔던 것을 \"projector 출력\" 위에서 다시 측정 — 5 모델이 4 가지
의미로 갈리는 ladder 가 드러난다.

- **표 (5 모델 × 4 cell)** — 행: Qwen / Idefics2 / LLaVA-Next / LLaVA-1.5 /
  InternVL3. 열: 3 가지 cell + verdict 칼럼:
  - **Qwen2.5-VL** — circle/filled/blank+both 에서 k=20 만 ablate 해도
    \"The circle is likely to remain stationary\" 로 깔끔하게 break.
    PMR 1.0 → 0.0. ★★★ 가장 깨끗한 break.
  - **Idefics2** — circle/filled/blank+both 에서 k=20 만 ablate 시
    \"The circle will disappear\" 로 PMR=0 break. **하지만 k=40, 80, 160
    에서는 \"The circle will continue to expand outward\" 로 PMR=1
    회복 (non-monotonic)**. circle/shaded 는 \"spin\" 으로 motion verb
    유지. ★★ 부분 break.
  - **LLaVA-Next** — circle/filled/blank+both 에서 k=20 → \"move
    downward\", k=40+ → \"expand\" 로 PMR=0 break. ★★ 부분 break.
  - **LLaVA-1.5** — circle/filled/blank+both 의 **baseline 자체가
    PMR=0** (\"The circle will be drawn towards the red arrow\"). 즉
    이미 abstract regime 이라 ablate 할 physics commitment 가 없음.
    ✦ baseline-already-abstract.
  - **InternVL3** — 모든 k 에서 \"The circle will likely fall downward
    due to gravity\" 유지. ✗ regime-cross 불가능.

- **하단 (오렌지 callout) — 핵심 발견**:
  - regime-cross capacity 는 (1) circle vs ball 라벨 의존, (2) 5-model
    사다리 (Qwen > Idefics2 ~ Next > LLaVA-1.5 > InternVL3).
  - LLaVA-1.5 의 \"NULL\" 은 진짜 NULL 이 아니라, **circle cell 의
    baseline 자체가 이미 abstract regime 이라 physics commitment 가
    ablate 할 대상으로 존재하지 않는 것**. round 1 의 \"NULL\" 헤드라인을
    재해석 필요.

**발표 팁**: 이 슬라이드는 storyline 의 슬라이드 17 (encoder SAE
intervention) 과 짝을 이루는 \"projector 출력 단계\" 의 메커니즘 결과.
시간이 짧으면 표 한 줄 (LLaVA-1.5 baseline=0!) 만 강조하면 청중이 다른
모델의 ladder 도 따라옵니다.

---

## 슬라이드 27 — Regime-shift vs binary PMR + 결정 시점

**한 문장**: 텍스트는 fall → hit → roll → redrawn 으로 분명히 이동하지만
binary PMR 점수는 모두 1 로 처리 — scoring artifact 가 \"NULL\" 헤드라인을
과장. 동시에 다음 한 발을 결정해야 하는 시점.

- **왼쪽 (오렌지 박스) — Scoring artifact**:
  - 예시 — LLaVA-1.5 ball / filled / blank+both:
    - baseline: \"The ball will fall.\" (PMR=1)
    - k=20 ablate: \"hit by the red arrow.\" (PMR=1)
    - k=80 ablate: \"will roll down the hill.\" (PMR=1)
    - k=160 ablate: \"redrawn.\" (PMR=0, but rare).
  - Text 는 fall → hit → roll → redrawn 으로 분명히 다른 \"motion verb\"
    들로 이동. Regime 자체가 흔들리지만, binary PMR 은 모든 motion verb 를
    동등하게 PMR=1 으로 점수.
  - 결론: M5b paper 헤드라인 \"NULL\" 은 사실 3 가지가 섞여 있음:
    1. **진짜 NULL** (InternVL3) — 어떤 ablation 도 응답을 흔들지 못함.
    2. **baseline-already-abstract** (LLaVA-1.5 circle) — baseline 자체가
       PMR=0 이라 break 할 대상이 없음.
    3. **binary PMR 의 가림** (LLaVA-1.5 ball) — text 는 분명히 이동했지만
       PMR 은 1 로 고정.
  - → \"NULL\" 라는 paper 헤드라인을 \"regime-cross capacity ladder\" 로
    reframe 하는 것이 paper-level 작업 (C3).

- **오른쪽 (파란 박스) — 결정 시점**:
  - **A1**: Variant B 진행 (gate override). GPU 24h. A 의 +0.4 shift 가
    B 에도 동일 적용된다고 가정. PRO: A↔B Δ-PMR 비교 가능 (axis
    ordering 보존). CON: A baseline 이 LLaVA-1.5 와 다른 분포라
    \"controlled LM swap\" 의 통제력이 약화.
  - **A2**: Variant A recipe 재학습. GPU 24h. LR / data-mix / num
    epochs 조정. PRO: LLaVA-1.5 floor 와 정렬 가능. CON: 확률적 — 또
    실패하면 시간 손실.
  - **A3**: M-PSwap (perceiver swap) 부활. 현재 NaN 미해결, backlog 에
    있음. PRO: Idefics2 의 perceiver-resampler 가설을 직접 검증.
    CON: 학습 안정화가 우선 미해결.

- **하단 권장**: step21000 final ckpt 로 regression 재실행 (B6) 결과를
  본 다음 A1 vs A2 결정. 병렬로 binary PMR → regime-shift score
  정의 작업 (C3 — paper draft reframe).

### B6 결과 추신 (2026-05-01 14:10 시점)

step21000 의 regression 재실행 완료. Aggregate PMR 0.869 (gate FAIL)
이지만 line/blank/none baseline cell n=10 에서 **PMR = 0.000 (PASS!)**.
샘플 응답: \"A circle is drawn on a white background. It is not clear
what will happen next. It could be a new circle drawn, or it could be
a different shape.\" — 전형적 abstract regime 응답.

해석: step21000 은 cell-discrimination 을 학습했음. 가장 추상적인
cell (line/blank/none) 에서는 PMR=0 으로 정확히 abstract response 를
내고, 물리-cue 가 강한 cell 에서만 PMR=1. **LLaVA-1.5 의 baseline
behavior (line baseline ≈ 0.05) 와 구조적으로 일치** — 단지 aggregate
가 +0.4 만큼 위로 shifted 되었을 뿐.

→ A1 (gate override + Variant B 진행 + A↔B Δ-PMR 비교) 가 원래 초안
보다 더 viable. 결정 시점: A1 추천. (자세한 내용:
`docs/insights/lmswap_a_recipe_drift.md` 의 \"Update — step21000 result\"
section.)

**발표 팁**: 이 슬라이드는 \"솔직한 진행 상황\" 슬라이드. 청중에게 결정을
나누는 게 아니라, 우리가 어떤 trade-off 위에서 다음 단계를 고를 것인지
공유. 발표 후 토론에서 가장 활발한 의견이 나올 슬라이드일 가능성이 높음.

---

## 슬라이드 28 — 한계 & Future (기존 슬라이드 24)

**한 문장**: Architecture-level finding 은 lock 되었지만, isolation 은 아직
미해결 — 다음 단계는 controlled swap 실험들.

- 기존 storyline 슬라이드 24 와 동일. 다만 다음 항목이 추가/수정됨:
  - **Pillar B (M-LMSwap)**: Variant A 21K 학습 완료. Variant B 게이트
    대기 중. recipe drift 는 항상 따라오는 잠재 위험.
  - **Post-projection SAE round 2 done**: regime-cross ladder 발견.
    M5b 헤드라인 reframe 필요 (paper draft 작업 C3).

(자세한 텍스트는 기존 storyline_ko.md 슬라이드 24 절 참조.)

---

## 슬라이드 29 — 결론 (기존 슬라이드 25)

**한 문장**: 행동 → 메커니즘 → 픽셀 의 3 차원으로 \"shortcut 의 위치\" 를
좁혀 들어갔다. 추가로 Pillar B 통제 실험을 통해 \"LM identity 의 인과적
역할\" 이 직접 답해질 단계 직전.

- 기존 storyline 슬라이드 25 의 3 가지 결론 + 추가 한 줄:
  4. **Architecture-level finding 의 isolation 작업 진행 중** — Pillar B
     (M-LMSwap) Variant A 학습 완료, Variant B 대기. M-PSwap 은 NaN
     미해결로 backlog.

(자세한 텍스트는 기존 storyline_ko.md 슬라이드 25 절 참조.)

---

## 슬라이드 30 — 감사합니다 / Q&A (기존 슬라이드 26)

(기존 storyline_ko.md 슬라이드 26 그대로. \"동반 자료\" 항목에 본 종합
검토 deck 노트를 추가.)

---

## 부록: 발표 시간 가이드 (30 슬라이드 기준)

| 단락 | 슬라이드 | 권장 시간 |
|---|---|---|
| 도입 (hook + question + why) | 1–4 | 4 분 |
| 배경 (VLM 한 장 도식 + 선행 연구) | 5–6 | 3 분 |
| 자극 + 모델 + 지표 | 7–10 | 4 분 |
| 결과 흐름 + 행동 결과 | 11–13 | 5 분 |
| 메커니즘 결과 (encoder boomerang → SAE → MLP) | 14–18 | 7 분 |
| 픽셀 결과 + Idefics2 미스터리 | 19–20 | 4 분 |
| 외부 타당성 + scaling/multilingual | 21–22 | 3 분 |
| 5-fold 종합 시그니처 | 23 | 2 분 |
| **Pillar B 진행** (M-LMSwap + 결정) | **24–27** | **6 분** |
| 한계 + 결론 + Q&A | 28–30 | 5 분 |
| **합계** |  | **~43 분** |

질의응답 포함 60 분 슬롯 가정. 시간이 짧다면 슬라이드 18 (knockout)·
22 (scaling) 를 가장 먼저 떨어뜨리고, 슬라이드 16 (VTI)·19 (픽셀)·24-27
(Pillar B) 는 어떤 시간 제약에서도 살려두는 것을 추천합니다.
