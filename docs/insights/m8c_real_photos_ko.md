# M8c — 실사진 (외적 타당성 라운드 3)

**상태**: 완료 2026-04-25.

## 동기

M8d의 카테고리 횡단 스윕은 합성 비공 카테고리에서 H7
(label-selects-regime) 을 검증하고 인코더-포화 가설을 카테고리 횡단으로
재검증했다. 그러나 지금까지의 모든 조사는 *프로그램 생성* 자극 — 검은
실루엣, 그라디언트 채움, 손 그린 텍스처 — 을 사용했다. "인코더 probe
AUC" 결과가 합성 패턴에 overfit 한 것일 수도 있다. M8c는 OOD 검증:
M2 / M8a / M8d의 VLM 물리-모드 결과가 실제 사진에서도 성립하는가?

## 자극 설계

5 카테고리 × 12 사진 = **60 사진**. 출처:

- **ball / car / person / bird** (각 12장): COCO 2017 validation set.
  카테고리 키워드로 캡션 필터링 (예: ball 은 basketball / soccer ball).
  각 사진은 512×512 (흰색 사각 패딩).
- **abstract** (12): WikiArt 코퍼스, abstract 스타일 클래스로 필터링.
  라이선스: WikiArt — public domain 또는 fair use (huggan/wikiart 기준).

라이선스 메타데이터는 `inputs/m8c_photos_<ts>/photo_metadata.csv` 에
논문 시점 attribution 용으로 기록.

카테고리별 라벨 트리플 via `LABELS_BY_SHAPE`:

| 카테고리 | physical | abstract     | exotic   |
|----------|----------|--------------|----------|
| ball     | ball     | circle       | planet   |
| car      | car      | silhouette   | figurine |
| person   | person   | stick figure | statue   |
| bird     | bird     | silhouette   | duck     |
| abstract | object   | drawing      | diagram  |

샘플링 설정 M8a / M8d 와 일치 (T=0.7, top_p=0.95).

## 결과

총 wall clock: H200 GPU 0 에서 **5분** (16:24:59 → 16:30:06).
4 runs × 60 사진 = 모델당 240 추론 = 총 480.

### 헤드라인 결과: 사진이 Qwen PMR(_nolabel) 을 *낮춤*; LLaVA 는 혼재

| 카테고리 | Qwen synthetic-textured | Qwen photo | Δ |
|----------|-----:|-----:|-----:|
| ball     | 0.900 | 0.667 | **−0.233** |
| car      | 0.975 | 0.500 | **−0.475** |
| person   | 0.850 | 0.667 | **−0.183** |
| bird     | 0.875 | 0.417 | **−0.458** |

| 카테고리 | LLaVA synthetic-textured | LLaVA photo | Δ |
|----------|-----:|-----:|-----:|
| ball     | 0.450 | 0.500 | +0.050 |
| car      | 0.375 | 0.000 | **−0.375** |
| person   | 0.025 | 0.417 | **+0.392** |
| bird     | 0.600 | 0.500 | −0.100 |

**이는 "사진 사실성이 인코더를 더 saturate 시킨다" 라는 단순한 예측을
반박한다.** Qwen의 PMR(_nolabel) 이 사진에서 18-48 pp 떨어진다; LLaVA
는 car (−37), bird (−10) 에서 떨어지지만 person (+39) 에서 *상승*.

### 왜 사진이 Qwen PMR(_nolabel) 을 낮추는가

합성 텍스처 자극은 최소한의 물리-객체 신호 (지면 + 화살표 + cast
shadow 가 있는 단일 텍스처 자동차; 그 외 없음). 실제 사진은 풍부한
scene context (다른 차량과 신호등이 있는 주차된 자동차; 글러브에
공을 던지는 야구 선수; 모이를 보는 새). "다음에 무엇이 일어날까?"
물을 때 모델은 이제 여러 객체와 맥락 단서 중에서 골라야 하며, 응답이
운동 *예측* 보다 scene *기술* 인 경우가 더 많다:

- "The car is parked at the curb beside other vehicles" — PMR=0
- "The image shows a baseball game in progress" — 동사 일치 없으면 PMR=0
- "There is a duck looking down at something on the ground" — PMR=0

합성 자극은 반대로, 이미지에 이미 모션 단서가 내장된 단일 격리된
객체에 대해 명확하게 묻는 것이다.

### 왜 LLaVA person 사진은 올라가는가

LLaVA 의 합성 person/_nolabel = 0.025 는 인코더가 합성 stick-figure /
텍스처 person primitive 를 물리-모드 추론 가치가 있는 사람으로 인식하기
어려운 것을 반영. 사람들이 무언가 하는 (스키, 던지기, 걷기) 실제
사진에서 LLaVA 의 응답 분포가 동작 동사 쪽으로 이동 → PMR(_nolabel)
= 0.417. 이는 M6 의 *encoder-recognition* 효과: 합성 클래스에 대해
인코더의 시각 prior 가 비포화일 때, 실제 사진이 합성 primitive 보다
prior 를 더 강하게 활성화 가능.

### 사진에서의 H7

LLaVA H7 paired-difference `physical − abstract` on 사진:

| 카테고리 | physical PMR | abstract PMR | Δ | strict pass (≥+0.05) |
|----------|---|---|---|---|
| ball     | 0.667 | 0.500 | +0.167 | ✓ |
| car      | 0.083 | 0.083 | 0.000  | ✗ |
| person   | 0.333 | 0.583 | −0.250 | ✗ (sign flip) |
| bird     | 0.833 | 0.167 | +0.667 | ✓ |
| abstract | 0.000 | 0.083 | −0.083 | n/a |

LLaVA H7 strict on 사진: **2/4** (ball + bird PASS; car + person FAIL).

car 실패는 전반적 낮은 PMR 때문 (car/_nolabel = 0.000; `car` 라벨도
0.083 까지만 push). person 실패는 sign flip — `stick figure` 사진이
`person` 사진보다 더 많은 motion narration 을 유도 (모델이 `stick
figure` 를 "걷기 시작하는 hand-drawn human figure" 로 매핑).

Qwen H7 paired-difference `physical − abstract` on 사진:

| 카테고리 | physical PMR | abstract PMR | Δ |
|----------|---|---|---|
| ball     | 0.667 | 0.583 | +0.083 |
| car      | 0.667 | 0.833 | **−0.167** |
| person   | 0.750 | 0.250 | **+0.500** |
| bird     | 0.750 | 0.750 | 0.000 |
| abstract | 0.500 | 0.500 | 0.000 |

Qwen H7 strict on 사진: **2/4** (ball + person PASS; car + bird FAIL).

Person/abstract = `stick figure` 이 Qwen 에서 0.250 PMR 산출 (`person`
의 0.750 대비 낮음) — Qwen 이 "stick figure" 사진을 그림으로 해석 →
abstract-leaning. 이건 *원래의 H7 패턴* — 라벨이 사진에서도 regime 을
선택하지만, 합성 자극보다 덜 일관됨.

### 방법론적 함의: 시각-포화는 일부 합성-자극 artefact

M2 / M8a / M8d 천장 패턴 (Qwen 합성 PMR(_nolabel) ≈ 0.85-1.00) 은
일부 합성 stim 의 *최소성* 에 의해 구동: 격리된 단일-객체 이미지 +
명시적 모션 단서 (화살표, cast shadow) 가 "물리-모드" 읽기를 최대화.
같은 카테고리의 실제 사진은 운동 동사를 발화시키지 않는 scene-기술
응답을 자주 유도.

이것은 M6 r2 시각-포화 가설을 **무효화하지 않음** (linear-probe AUC on
physics-vs-abstract 는 모델 간 여전히 추적). 이를 정제: **행동 PMR**
은 일부 합성 stim 단순성에 의해 구동, 인코더 표현뿐 아니라. 두 corollary:

1. **Qwen 의 합성 PMR(_nolabel) 천장은 순수 인코더 포화가 아님** —
   *프롬프트-맥락 단순성* 과 *인코더 신뢰* 의 결합. 단순성을 제거 (사진)
   하면 인코더 응답이 실제로 포화되지 않음을 드러냄.
2. **M5a 스티어링 / M3 probe 결과 (인코더 가 Qwen 에서 AUC ~0.99 로
   physics-vs-abstract 를 linearly separates) 는 행동 PMR(_nolabel) 과
   다른 양을 측정**. 인코더가 식별을 *할 수 있음*; LM이 그것에
   *작용하는지* 는 프롬프트 맥락에 의존.

이는 시각-포화 가설의 **논문 관련 정제**.

### 카테고리별 노트

- **ball** — 합성-사진 갭 적당 (Qwen 0.900 → 0.667). COCO ball 사진은
  스포츠 장면 (글러브에 공을 던지는 야구 선수) 이 자주 → kinetic-rich.
- **car** — 가장 큰 합성-사진 갭 (Qwen 0.975 → 0.500; LLaVA 0.375 →
  0.000). COCO car 사진은 자주 신호등 + 거리 표지가 있는 정적 주차된
  차량 샷.
- **person** — 모델별 반대 방향. LLaVA 는 인코더가 마침내 사람을
  인식해서 상승; Qwen 은 scene context 가 distract 해서 하락.
- **bird** — Qwen 큰 하락 (0.875 → 0.417); COCO bird 사진은 모이를 보는
  새가 자주 (PMR=0 unless kinetic 동사 확장).
- **abstract** — 두 모델 모두 LLaVA ~0.000, Qwen ~0.500 PMR. Qwen 의
  WikiArt abstract 회화는 자주 "춤추는 인물 묘사" 로 기술 (PMR>0) —
  추상이 인간 형태를 함축하는 경우가 많음.

### 분류기 노트

`classify_regime` 은 car / person / bird (M8d 카테고리) 만 정의됨. ball
/ abstract 사진은 원본 `score_pmr` (gravity-fall 동사 매치) 사용. 모든
M8c PMR 값은 M2 / M8a 와 일관성을 위해 `score_pmr` (binary gravity-fall
메트릭) 사용; M8d-스타일 regime 분포는 car / person / bird 에 대해서만
analyzer CSV 에 보고.

## 가설 업데이트

- **H1** — *변경 없음*. M8c 에서 직접 검증되지 않음 (추상화-축 변동 없음;
  사진은 단일 "레벨").
- **H7** — **사진에서 부분적 성립** (LLaVA 2/4, Qwen 2/4). 도형 횡단 /
  카테고리 횡단 복제가 합성 (M8d 3/3 LLaVA) 보다 사진 (2/4 LLaVA) 이
  더 약함. 사진은 H7 신호를 마스킹하는 scene-context 노이즈 추가하지만
  반전시키진 않음.
- **H-encoder-saturation** — **정제**. 인코더 probe AUC (M6 r2) 가 행동
  PMR(_nolabel) 의 유일한 driver 가 아님. 합성-stim 단순성이 공동 인자:
  같은 인코더, 더 단순한 stim → 더 높은 PMR(_nolabel). 행동 포화는 (a)
  인코더 표현 포화와 (b) "최소 물리-객체" 읽기를 가능하게 하는
  입력-맥락 단순성의 결합.

## 로드맵 함의

1. **논문에서 행동 측정은 stim type 명시** — "Qwen PMR(_nolabel) =
   합성 텍스처 자극에서 0.95; COCO 동등 사진에서 0.50".
2. **인코더 probe AUC (M6 r2) 는 행동 PMR 과 독립** — 인코더가 클래스를
   linearly separate; 행동 PMR 은 prompt + 입력-맥락 단순성에 의존.
3. **M8e (cross-source paired analysis)** 가 잘 동기화. 4 물리 카테고리에
   대해 합성-텍스처 (M8a/M8d) + 사진 (M8c) 둘 다 보유; 카테고리 × 모델
   paired delta 가 정보적.
4. **Round-2 큐레이션 업그레이드**: COCO 는 정적 "scene" 사진이 많음;
   업그레이드는 카테고리가 프레임을 채우는 OpenImages bbox-cropped
   subset (맥락 노이즈 적음). Round-2 는 또한 더 깨끗한 exotic 역할을
   위해 bird 카테고리에 penguin / chicken 추가.
5. **논문 클레임 정제**: M2 / M8a / M8d 의 "Qwen 이 abstract stim 을
   물리적으로 처리" 는 여전히 성립 — Qwen 합성 PMR 천장. M8c 가 이를
   "Qwen 이 *최소* 시각-prior stim 을 물리적으로 처리" 로 reframe,
   사진이 유용한 counterfactual.

## 아티팩트

- `docs/figures/m8c_photo_grid.png` — 샘플 grid (5 cat × 4 photo).
- `docs/figures/m8c_pmr_by_category.png` — PMR by (category × label_role).
- `docs/figures/m8c_paired_synthetic_vs_photo.png` — synthetic-textured
  PMR(_nolabel) vs photo PMR(_nolabel) per (category × model).
- `outputs/m8c_summary/` — 모델별 rollup CSV.
- `notebooks/m8c_real_photos.ipynb` — 셀별 재현.
- `inputs/m8c_photos_<ts>/photo_metadata.csv` — 라이선스 + URL
  attribution 표.
