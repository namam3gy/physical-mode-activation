# M7 — Human baseline collection plan (Prolific)

논문 (EMNLP) blocking item. VLM의 PMR/H7/H1 패턴이 인간과 어떻게 비교되는지 측정.

목표: **20 raters × 50 stim**의 응답을 모아서 (1) human PMR vs VLM PMR 비교, (2) H7 (label 효과)이 인간에서도 보이는지, (3) H1 (abstraction ramp)이 인간에서 어떤 모양인지 확인.

---

## 1. Stim 선정 (50개)

### 추천 전략 — "balanced minimal subset"

M2 stimulus pool (480 stim = 4 object × 3 bg × 4 cue × 1 event × 10 seeds)에서 **factorial 균형 + 모델별 PMR variance를 max하는 50개**를 뽑음.

#### 옵션 A: 5×5×2 = 50 cell sampling (추천)

axis별 균등 분포:
- **object_level** (5개): line, filled, shaded, textured + (옵션: hex_filled or polygon)
  - **추천 4개만 사용** (line, filled, shaded, textured) → 각 axis 12-13 stim per level
- **bg_level** (3개): blank, ground, scene → 각 12-17 stim
- **cue_level** (4개): none, cast_shadow, motion_arrow, both → 각 12-13 stim
- **label** (3개): ball, circle, planet → 각 ~17 stim
- **seed**: cell당 1개씩 (10 seed 중 1개 random sample)

이렇게 하면:
- 4 obj × 3 bg × 4 cue × 1 event = **48 cells**
- + 2 bonus stim (e.g., 가장 strong cue 셀에서 추가 seed) = **50 stim**
- 각 stim에 3 label 변형 적용 → **150 trials per rater** (너무 많음)
- 또는 cell당 1 label만 → **50 trials per rater** ← **이게 paper-friendly**

#### 옵션 B: VLM disagreement-driven sampling

5 모델의 응답이 가장 갈리는 50 stim을 선정 (standard deviation of per-stim PMR across 5 models ranked highest). 이건 "human baseline은 모델 간 disagreement 영역에서 어떤 ground truth?" 질문에 답함.

장점: paper 다 쓰일 셀이 가장 informative
단점: factorial 균형 깨짐 — H1 ramp 같은 axis-별 분석 불가

#### 권장: **옵션 A + label 변형 1개씩**

→ 48 cells × 1 label/cell + 2 bonus = 50 stim, 50 trials per rater.

stim 선정 노트북/스크립트는 별도 작성 가능 (`scripts/select_human_baseline_stim.py`).

### Stim source

`inputs/mvp_full_20260424-093926_e9d79da3/images/<sample_id>.png` 사용. 모두 PNG, 224×224 (또는 stim spec 따라).

---

## 2. 질문 형식 (open + FC 둘 다)

VLM과 직접 비교하려면 **same prompt** 써야 함. M2의 OPEN prompt + M9의 FC prompt 둘 다 수집 권장.

### Variant 1: Open-ended (PMR primary metric)

```
[이미지]

이 이미지에는 {label}가 보입니다.
다음 순간에 {label}에 무슨 일이 일어날 것 같습니까?
한 문장으로 짧게 답변해주세요.
```

응답 예 (한국어 또는 영어 둘 다 허용 표시):
- "공이 바닥에 떨어집니다."
- "The ball will fall."
- "원이 그대로 있을 것 같습니다."

VLM scorer (PMR rubric)을 응답에 동일하게 적용 — physics verb stem 매칭. (Korean / Japanese stems 도 이미 scorer에 있음.)

### Variant 2: Forced-choice (4-way MCQ; M9 protocol)

```
[이미지]

이 이미지에는 {label}가 보입니다.
다음 중 무슨 일이 일어날 것 같습니까?

A) 떨어진다
B) 그대로 있는다
C) 옆으로 움직인다
D) 이는 추상적인 도형일 뿐 — 어떤 물리적 일도 일어나지 않는다

A, B, C, D 중 하나로 답변하고 짧게 이유를 적어주세요.
```

(영어 변형 — paper에서 cross-language 비교에 사용 가능.)

### 어떤 prompt를 쓸지

권장: **OPEN만** (50 trials per rater). FC는 추가 budget 있으면.
이유: OPEN이 PMR primary metric이고, VLM도 OPEN protocol이 5-model uniform.

---

## 3. Prolific 셋업

### Account / Study type

- Prolific.co (academia 가격 = $9-12/hour 기준)
- "Study" type: image-based questionnaire
- Single-session, 10-15분 estimated

### Recruitment criteria

- **언어**: English fluent (paper는 영어이므로). 또는 Korean fluent (KO label 비교를 위해 separately collected).
- **위치**: US/UK/CA/AU (English) — 응답 품질 일관성
- **나이**: 18-60 (vision normal age)
- **Vision**: "no uncorrected visual impairment" pre-screen
- **Past performance**: ≥ 95 % approval rate
- **Demographics quota**: 가능하면 50/50 gender, age-balanced

### 인원 + 예산

20 raters × 50 stim × ~10 sec per stim = **8 minutes per rater + reading time = ~12-15 minutes** total.

Prolific 권장 시급 $9 (= ~£7) → 12 min × $9/60 = **$1.80 per rater × 20 = $36 total** (학술 기준 minimum). $12 시급으로 olerk → **$48 total**.

여기에 Prolific 수수료 (33% surcharge for academic) 추가:
- Total cost: ~$48 × 1.33 = **~$64 USD** for 20 raters.

만약 60 raters까지 늘리면 (paper에서 Wilson CI tighter): **~$192 USD**.

### Pre-registration

Open Science Framework (OSF) preregistration 권장 — paper-defensible. 등록 항목:
1. 가설 (H1, H2, H7, H-encoder-saturation)
2. 분석 계획 (PMR comparison, GAR comparison, H7 effect)
3. 제외 기준 (예: < 30sec response time, attention check fail)
4. 샘플 사이즈 결정 근거

---

## 4. Survey 플랫폼

### 옵션 A: Qualtrics (대학 라이선스 있으면)
- 가장 robust, attention check 내장
- 이미지 hosted (Qualtrics 또는 외부 URL)

### 옵션 B: Google Forms (무료)
- 가벼움, 빠름. 단 randomization 한계
- 50 stim를 form 1개에 다 넣고, *image randomization* 은 Google Apps Script로 가능

### 옵션 C: Custom HTML survey + S3 hosting
- Most flexible. paper-defensible.
- 우리 stim이 PNG라 S3/CDN URL로 host
- 응답은 form POST → 우리 서버 또는 Google Sheets

### 권장: **Qualtrics or 단순 HTML form**
- 50 stim는 form 작성 부담 적음
- HTML 직접 작성 시: per-rater random ordering of 50 stim, attention check (예: stim 25번에 "이 텍스트는 'control' 입니다. C 선택해주세요" 삽입)

---

## 5. Attention check + quality control

### Attention check (필수)
20 trials 마다 1개 — 총 50 stim 중 2-3개:

> 이 trial은 응답 정확도 확인용입니다.
> "control" 단어를 응답에 포함시켜주세요.

또는 명백한 stim (예: "blue square" + "이 사각형 무슨 색?") 삽입.

### Exclusion criteria (pre-registered)
- Attention check fail rate > 1/3 → exclude
- Mean response time < 5 sec/stim → exclude (rushing)
- Mean response length < 3 words/stim → exclude (effort)
- Total time > 30 min → exclude (distraction)

### Bonus payment
완료 시 $0.50-1.00 bonus → 응답 품질 향상.

---

## 6. 데이터 분석 계획

### Output format

응답 CSV: `human_baseline.csv`
```
rater_id, sample_id, label, prompt_variant, response_text, response_time_ms, attention_pass
H001,    filled_blank_both_fall_001, ball, open, "공이 바닥에 떨어진다", 8453, 1
...
```

### 분석 코드

`scripts/analyze_human_baseline.py` (작성 예정):
1. Load `human_baseline.csv`
2. Apply same PMR scorer (`physical_mode/metrics/pmr.py`)
3. Compute per-stim mean human PMR (across 20 raters)
4. Join with VLM per-stim PMR (5 models)
5. **Comparison metrics**:
   - **Overall PMR ladder**: Human vs Qwen vs LLaVA-1.5 etc. (bootstrap CI)
   - **Per-cell agreement**: Cohen's κ between human and each model
   - **H7 (label effect)**: human GAR by label (ball / circle / planet) — does ordering match VLM?
   - **H1 (abstraction ramp)**: human PMR by `object_level` — same monotone ramp? (line < filled < shaded < textured?)
   - **Cue effect**: human PMR by `cue_level` — none < cast_shadow < motion_arrow < both?
6. **Headline plot**: 5-model VLM PMR ladder + human PMR overlay (additional point on the ladder)

### Paper integration

- 새 figure: `docs/figures/human_vs_vlm_pmr_ladder.png` — 6-bar (5 models + human) PMR comparison.
- 새 paper section: §4.5 또는 §8.6 "Human baseline" — short subsection.
- Discussion 통합: "Humans pattern more like {cluster}" 또는 "Humans don't show the saturation pattern" (예측)

---

## 7. 예측 (paper 작성에 도움)

### Most likely human pattern

1. **Human PMR aggregate < non-CLIP cluster**, ≈ LLaVA-Next:
   - Humans recognize "circle" as abstract more often than non-CLIP models (no encoder-saturation)
   - But cues (cast_shadow, motion_arrow) drive humans to physics interpretation
   - Predicted human aggregate PMR ≈ 0.50-0.65 (similar to LLaVA-Next 0.70 floor)

2. **H1 ramp: humans clean monotone**:
   - line (most abstract) → filled → shaded → textured (most physics-like)
   - 인간은 saturation 없으니 H1이 가장 깔끔하게 나타날 것

3. **H7 label effect: humans show it**:
   - "ball" 라벨에서 가장 높은 PMR, "circle"에서 가장 낮음, "planet" 중간
   - 인간도 label-prior에 영향 받음 (linguistic prior)

4. **cue=none + label=ball**: 가장 미묘한 cell. 인간이 어떻게 응답할지 가장 informative.

### Mismatch areas (paper에서 강조)

- **VLM 모두 "circle + cue=none"에서 PMR이 너무 높을 것** (Qwen 0.797, Idefics2 0.711 등); 인간은 < 0.3 예상
- **VLM의 cue=both에서 PMR ≈ 1.0** vs 인간 ~0.7-0.9 (인간이 더 신중)

이 mismatch가 "VLM saturation"의 quantitative 근거.

---

## 8. Timeline (사용자 예상)

| 단계 | 시간 | 비고 |
|---|---|---|
| Stim 50 sample 선정 + 정리 | 1-2 hr | `select_human_baseline_stim.py` 작성 |
| Survey form 만들기 (HTML or Qualtrics) | 2-4 hr | randomization + attention check |
| Prolific study 등록 | 30 min | Account setup + pre-registration |
| OSF preregistration | 1 hr | 가설 + 분석 계획 |
| Data collection (Prolific 자동) | 4-12 hr (일정 따라) | 일반적으로 빨리 모집됨 |
| Data cleaning (exclusions) | 1 hr | scripted |
| Analysis | 2-3 hr | `analyze_human_baseline.py` 작성 |
| Paper integration | 1-2 hr | figure + subsection |

**Total: ~13-26 hours of work (excluding Prolific waiting)**.

---

## 9. 다음 단계 (사용자가 할 수 있는 것)

1. **Decision**: 옵션 A (factorial 균형) vs 옵션 B (disagreement-driven) — paper 강조점 따라.
2. **Stim 50 선정**: scripts/select_human_baseline_stim.py 작성 요청 가능.
3. **Survey form**: Qualtrics 라이선스 유무에 따라.
4. **Prolific 계정 + 결제**: 사용자 직접.
5. **OSF preregistration**: 가설 + 분석 plan 정리해서 등록.
6. 우리가 이 단계에서 도울 수 있는 것:
   - Stim selection script
   - Analysis script (`analyze_human_baseline.py`)
   - Paper integration section

질문/논의 필요한 점:
- Open vs FC vs 둘 다? (예산 trade-off)
- 60 raters로 늘릴지 (CI tighter) vs 20 raters?
- KO 라벨도 별도 수집할지 (KO native speakers, 추가 budget)
- Pre-registration 어디서? (OSF / AsPredicted)
