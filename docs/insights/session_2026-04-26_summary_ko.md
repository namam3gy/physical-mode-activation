---
session: 2026-04-26 자율 (이어짐)
date: 2026-04-26
status: complete
scope: §4.7 (RC per-axis 안정성) + §4.11 (categorical H7 regime 분포)
commits: 309bdf6 → bbf01f9
---

# 세션 2026-04-26 — §4.7 + §4.11

## 이 세션의 산출물

기존 M8d / M8a label-free 데이터를 재사용하는 §4 add-on 2 개, 새 추론
없음. §4 backlog 의 분석-only 항목 마무리.

1. **§4.11 — categorical H7 regime 분포** (commit `309bdf6`).
   `classify_regime` 을 4-모델 M8d label-free + labeled run (Qwen /
   LLaVA-1.5 / LLaVA-Next / Idefics2) 에 적용. 4×3×4 stacked-bar matrix
   가 (모델 × 카테고리 × label_role) 별 kinetic / static / abstract /
   ambiguous 비율 보임.

2. **§4.7 — per-axis RC 결정 안정성** (commit `bbf01f9`).
   T=0.7 하의 RC 를 M8a label-free 에서 per-axis 결정 안정성으로 재해석.
   5 모델 × 3 축 (object_level / bg_level / cue_level) × {각 2-4 level}.

## 헤드라인 발견

### §4.11 — regime 분포가 binary H7 가 가린 모델 차이를 드러냄

![§4.11 4-모델 M8d regime 분포](../figures/sec4_11_regime_distribution_4model.png)

- **Qwen + Idefics2**: 모든 곳에서 saturated kinetic (~95%). Qwen
  `person × exotic` (statue) 만 ~30% static.
- **LLaVA-1.5**: 가장 regime-discriminative. `car × abs` (silhouette) 가
  kinetic 을 28% 로 낮추고 ambiguous 70%.
- **LLaVA-Next**: intermediate. `person × exotic` (statue) 가 3-way split
  보임 (30% kinetic + 25% static + 25% abstract) — LLaVA-1.5 에 없는
  multi-axis architectural twist.

`person × abs` (stick figure) 의 4-모델 gradient:
| Model | % kinetic |
|---|---:|
| Qwen | 91 |
| Idefics2 | 99 |
| LLaVA-Next | 80 |
| LLaVA-1.5 | 58 |

M9 H7 finding 의 granular form. Categorical 뷰가 commit 의 *종류* 를
드러냄, 단지 commit *여부* 가 아님.

### §4.7 — saturated 모델에 cue_level 이 지배적 결정 안정자

![§4.7 5-모델 per-axis RC](../figures/sec4_7_rc_per_axis.png)

| model | cue=none → cue=both | bg=blank → bg=ground |
|-------|---------------------|----------------------|
| Qwen2.5-VL | 0.84 → **1.00** (+0.16) | 0.88 → 0.96 (+0.08) |
| Idefics2 | 0.88 → 0.99 (+0.11) | 0.92 → 0.95 (+0.03) |
| InternVL3 | 0.89 → 0.98 (+0.09) | 0.92 → 0.96 (+0.04) |
| LLaVA-1.5 | 0.85 → 0.85 (0) | 0.88 → 0.82 (**−0.06**) |
| LLaVA-Next | 0.78 → 0.78 (0) | 0.77 → 0.80 (+0.03) |

**해석**: saturation 이 단지 행동 PMR ceiling 만이 아니라 **결정-안정성
ceiling** 이기도 함. Non-CLIP 모델이 cue fire 시 5 seed 모두 같은 PMR
call 로 수렴; CLIP-기반 모델이 강한 cue 에서도 seed-level variance 보유.
H-encoder-saturation reframe 의 별도 시그니처.

## 가설 상태 업데이트

- **H7** — 이미 "unsaturated-only AND architecture-conditional"; §4.11
  가 **categorical** 차원 추가 (binary→regime 분포), binary H7 수치가
  muted 인 곳에서도 LLaVA-1.5 의 라벨-disambiguate 가 regime 수준에서
  작동함을 보임.
- **H-encoder-saturation** — 이미 "5 모델 점 × 3 stim source 에서
  architecture-level 확인"; §4.7 가 **결정-안정성 차원** 추가:
  saturation 이 cue 하의 seed-level commit 도 lock. 같은 architectural
  속성의 두 별도 시그니처.

## 세션 후반 추가: InternVL3 M8d (§4.11 5-모델 갭 닫음)

§4.7 + §4.11 4-모델 commit 후, InternVL3 가 M8d 에서 실행됨 (GPU 0
~13분) 그리고 §4.11 figure 가 5-모델로 재생성. Commit `be29792`
(§4.11 5-모델 close) 와 `3b1e5d8` (M9 audit InternVL3 M8d 행).

**InternVL3 M8d 신규 발견**: `person × exotic` (statue) PMR 이 0.800
(physical "person") 에서 0.481 (exotic "statue") 로 하락 — 32 pp 억제.
Categorical 뷰: 30% kinetic / 65% static — **프로젝트에서 가장 강한
단일 라벨-driven static commit**. saturated-encoder architecture (M8a
PMR 0.92) 도 라벨이 uniquely non-moving entity 가리킬 때 fire 하는 활성
라벨-disambiguation channel 보유함을 보임.

업데이트된 5-모델 `person × abs` (stick figure) gradient:
| Model | % kinetic |
|---|---:|
| Idefics2 | 99 |
| InternVL3 | 99 |
| Qwen | 91 |
| LLaVA-Next | 80 |
| LLaVA-1.5 | 58 |

5-모델 §4.11 figure: `docs/figures/sec4_11_regime_distribution_5model.png`.
Roadmap §4.11 가 "partial" 에서 "complete" 로 승격.

## 이어지는 한계

1. ~~§4.11 InternVL3 누락~~ — *닫음* (commit `be29792`).
2. **§4.11 5-카테고리 fine-grained 분류기** (gravity-fall / gravity-roll
   / orbital / inertial / static) M2 circle-only 데이터에 여전히 열림 —
   신규 키워드 셋 + `classify_regime` 의 `circle` 도형 확장 필요.
3. **§4.7 n_seeds=5** 가 RC 의 최소. ≥10 pp 차이 robust; <5 pp 차이
   시사적.
4. **§4.7 단일 arm (label-free)**. Labeled arm 은 다른 RC 구조 보일 수
   있음, 라벨 자체가 commit 안정화 하므로.

## 산출물

### Commit (이 세션, 9 substantive + bookkeeping)

- `309bdf6` — §4.11 4-model M8d regime distribution
- `bbf01f9` — §4.7 per-axis RC stability
- `be29792` — §4.11 5-model 마무리 (InternVL3 M8d)
- `73a9bf9` — §4.3 Qwen-only 한국어 라벨
- `df44a19` — §4.3 5-model cross-model 확장
- `c05e170` — Korean PMR scorer (lexicons + fallback)
- `38ef1c4` — §4.3 framing 다듬기 (advisor 반영)
- `622468e` — §4.3 Japanese scaffold (configs + script + JA lexicon)
- `b754fdf` — §4.3 Japanese 5-model + Chinese-fallback scorer
- `56d65ea` — §4.6 design spec (사용자 review 대기)

### 신규 figure

- `docs/figures/sec4_11_regime_distribution_4model.png`
- `docs/figures/sec4_11_regime_distribution_5model.png`
- `docs/figures/sec4_7_rc_per_axis.png`
- `docs/figures/sec4_3_korean_vs_english.png` (Qwen-only KO)
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` (5-model KO)
- `docs/figures/sec4_3_japanese_vs_english_cross_model.png` (5-model JA)

### 신규 insight 문서 + design spec

- `docs/insights/sec4_11_regime_distribution.md` (+ ko)
- `docs/insights/sec4_7_rc_per_axis.md` (+ ko)
- `docs/insights/sec4_3_korean_vs_english.md` (+ ko, Japanese ext 포함)
- `docs/insights/session_2026-04-26_summary.md` (이 문서, + ko)
- `docs/superpowers/specs/2026-04-26-sec4_6-counterfactual-stim-design.md`
  (design — 사용자 review 대기)

### 신규 script

- `scripts/sec4_11_regime_distribution.py`
- `scripts/sec4_7_rc_per_axis.py`
- `scripts/sec4_3_korean_vs_english.py` (Qwen-only KO)
- `scripts/sec4_3_korean_vs_english_cross_model.py` (5-model KO)
- `scripts/sec4_3_japanese_vs_english_cross_model.py` (5-model JA)

### 신규 config

- `configs/sec4_3_korean_labels.py` (Qwen)
- `configs/sec4_3_korean_labels_{llava,llava_next,idefics2,internvl3}.py`
- `configs/sec4_3_japanese_labels.py` (Qwen)
- `configs/sec4_3_japanese_labels_{llava,llava_next,idefics2,internvl3}.py`

### Lexicon / scorer 변경

- `src/physical_mode/metrics/lexicons.py` — KOREAN, JAPANESE, CHINESE
  physics-verb stems + abstract markers 추가.
- `src/physical_mode/metrics/pmr.py` — `score_pmr` 가 4 언어 path
  체크 (English / Korean / Japanese / Chinese fallback).
- `tests/test_pmr_scoring.py` — 23 신규 regression 케이스 (5 KO+,
  3 KO−, 6 JA+, 3 JA−, 6 CN+, 1 Katakana). Total 51 PMR 케이스.
- `docs/scoring_rubric.md` (+ ko) — Korean fallback 문서화.

### Roadmap

- §4.11 가 "complete" 표시 (5-model with InternVL3 M8d 늦게-세션 추가)
- §4.7 가 "complete" 표시
- §4.3 가 "Qwen-only" → "5-model × 2 언어 (Korean, Japanese)" 로
  promotion
- §4.10 milestone 표 row 가 ✅ 로 업데이트 (여전히 PRIORITY 6 태그됨이었음)
- §4.6 design spec 작성 (autonomous defaults), 사용자 review 대기.
  구현 시작 안 함 (HARD-GATE).

## Late-session addition #2: §4.3 한국어 vs 영어 라벨 prior (5-model + scorer 수정)

§4.7 + §4.11 + InternVL3-M8d 후 §4.3 (원래 "PMR scorer 가 English-only"
caveat 와 함께 열림 표시) 가 end-to-end 실행:

1. **Qwen-only 초기** (commit `73a9bf9`): 단일 모델 Korean labels
   (공/원/행성) on M8a circle. 라벨 간 ordering 언어 횡단 보존; `행성`
   이 `planet` 대비 ~9 pp 감소. PMR scorer 가 cross-language 적용 가능
   — 모델이 한국어 라벨에도 영어로 응답 (Qwen 에서 0/240 한국어-only).

2. **5-model cross-model 확장** (commit `df44a19`): 같은 한국어-라벨
   config 가 LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3 에 복제
   (configs: `configs/sec4_3_korean_labels_<model>.py`). 각 모델의
   기존 M8a circle EN baseline 이 새 한국어 run 과 페어링. 헤드라인:
   라벨 간 ordering 4/5 보존; LLaVA-1.5 swing 최대 (avg |Δ|=0.11;
   Vicuna LM 약한 한국어 SFT); Idefics2 가 KO `공 > 원 > 행성` vs
   EN `ball > planet > circle` 으로 rank-flip; InternVL3 swing 최소
   (천장 + InternLM3 강한 한국어).

3. **Korean scorer 수정** (commits `c05e170` + `38ef1c4`): advisor flag
   가 1200 응답 중 12개 한국어-only 응답 (LLaVA-Next 4, Idefics2 8) 노출
   — English-keyword scorer 가 조용히 누락. Korean physics-verb stems
   (`떨어` / `이동` / `움직` / ...) + Korean abstract markers (`그대로`
   / `움직이지 않` / ...) 를 `src/physical_mode/metrics/lexicons.py` 에
   추가, `score_pmr` 에 substring fallback. `tests/test_pmr_scoring.py`
   에 8개 신규 regression test (총 36 케이스, 모두 통과). Scorer 수정
   이 Idefics2 exotic deficit 축소 (−0.10 → −0.05) 그러나 rank-flip
   보존; LLaVA-1.5 수치 변화 없음 (0/80 KO-only) — LLaVA-1.5 swing 이
   진짜이고 scorer artifact 가 아님 확인 (advisor 의 blind-spot
   우려를 실증적으로 반박).

**헤드라인 (메커니즘)**: Vision-language joint space 에서 multilingual
semantic representation 이 4/5 모델 유지. Magnitude 가 *LM-측* 한국어
fluency 에 의해 bottlenecked (Vicuna < Mistral < InternLM3 ≈ Qwen2.5),
vision encoder 아님. 같은 encoder + 다른 LM → 다른 KO magnitude.
Encoder-saturation / label-prior 스토리 (M6 r2 / M8a / §4.7) 와 별개의
**language-prior 축** 추가.

문서: `docs/insights/sec4_3_korean_vs_english_ko.md`.
Figures: `docs/figures/sec4_3_korean_vs_english{,_cross_model}.png`.

## Late-session addition #3: §4.3 Japanese (5-model) + Chinese fallback

§4.3 Korean 종료 후, 사용자가 Korean "language-fluency-bottleneck"
finding 이 일반화되는지 테스트하기 위해 자율 모드에서 Japanese arm
승인. End-to-end 실행:

1. **Japanese scaffold** (commit `622468e`): 5 configs, 분석 script,
   JAPANESE_PHYSICS_VERB_STEMS / JAPANESE_ABSTRACT_MARKERS 를 lexicon 에
   추가, Japanese regression test (44 → 51 케이스 모두 통과).

2. **Japanese aggregation + Chinese fallback** (commit `b754fdf`):
   추론 완료 (~50 분, 5 sequential models). 분석 결과 Idefics2 가
   `惑星` 에서 simplified Chinese 응답 (19/80) — Mistral-7B 가 제한된
   Japanese SFT 지만 공유 kanji 를 Chinese 惑星 로 인식. 정확히 점수
   하기 위해 CHINESE_PHYSICS_VERB_STEMS lexicon (下落 / 掉入 / 跌落 /
   坠落 / 下降 / 旋转 / 飞行 / ...) 추가. Fix 없으면 Idefics2 exotic Δ
   가 misleading **−0.15** (scorer artifact); 수정 후 +0.05.

**Cross-language mechanism finding**:
- **Korean 이 language-fluency-bottleneck 테스트**: Hangul 고립이
  engagement 강제; 4/5 ordering 보존; LLaVA-1.5 swing 0.11 이
  Vicuna-Korean 약점 진짜 측정.
- **Japanese 가 kanji-as-bridge 테스트**: bootstrap noise 안에서 5/5
  ordering 보존, 그러나 *다른 path*:
  - Qwen2.5-VL 가 Japanese 라벨 85-91% 유지 (진짜 engagement)
  - LLaVA-1.5 / LLaVA-Next / InternVL3 가 대부분 kanji 영어 번역
  - Idefics2 가 `惑星` 에서 simplified Chinese 로 fallback
- LLaVA-1.5 ↓Korean / ≈Japanese 비대칭 (0.11 vs 0.05) 이 Vicuna-Japanese
  가 더 강함의 증거 **아님**; script 의 번역 가능성을 반영, LM SFT
  깊이 아님.

이는 Korean-only 결과보다 더 풍부한 발견이며, §4.3 을 두 distinct
mechanism 의 테스트로 reframe.

문서: `docs/insights/sec4_3_korean_vs_english_ko.md` (Japanese
cross-model 섹션 추가됨).
Figure: `docs/figures/sec4_3_japanese_vs_english_cross_model.png`.
Roadmap §4.3 Japanese ext 세부 사항 업데이트.

## Late-session addition #4: §4.6 design spec 작성 (autonomous)

사용자가 §4.6 (SAE/VTI reverse counterfactual stim 생성) 을 다음 우선
순위로 승인. brainstorming skill 호출 → autonomous defaults 적용 →
spec 작성 `docs/superpowers/specs/2026-04-26-sec4_6-counterfactual-
stim-design.md` (commit `56d65ea`).

Approach (recommended): pixel-space gradient ascent on Qwen2.5-VL 의
post-processor `pixel_values` tensor. Loss = `−⟨mean(h_L10[visual_tokens]),
v_L10_unit⟩` (M5a steering direction). Bounded ε ∈ {0.05, 0.1, 0.2}
sweep + unconstrained ablation. Random-direction control (n=3) 으로
v_L10-specificity 확인. Causal sanity check: PIL reconstruction
round-trip + 새 PMR 추론.

**writing-plans skill 호출 전 사용자 review 대기** (brainstorming
skill 의 HARD-GATE). 구현 시작 안 함.

## 이 세션 후 통합 backlog

열린 §4 항목:
- ~~§4.3 — 한국어 vs 영어 라벨 prior~~ — *닫힘* (commit
  `73a9bf9` + `df44a19` + `c05e170` + `38ef1c4` + `622468e` +
  `b754fdf`). Scorer 가 Korean + Japanese + Chinese (3 언어) 로 확장.
  스페인어 와 완전 target-language 프롬프트는 미래 확장으로 열림.
- §4.4 — Michotte 2-frame causality (2-이미지 prompt 지원 필요)
- ▶ §4.6 — VTI-reverse counterfactual stim — **design spec 작성됨
  (autonomous), 사용자 review 대기** at `docs/superpowers/specs/
  2026-04-26-sec4_6-counterfactual-stim-design.md`. 구현 estimate
  10-11 hrs across 4-5 iterations.
- §4.8 — PMR scaling (Qwen 32B/72B — 새 대형 모델 로드 필요)

주요 milestone:
- **M5b** — SIP / activation patching / SAE 특징 분해 (mechanism-level
  증거, 다음 논문-section 갭)
- **M7** — 논문 초고 + Prolific 인간 baseline

## 세션 누계 (2026-04-25 + 2026-04-26)

- M6 r6 시작 이래 총 commit: ~25 substantive (+ session / scorer /
  bookkeeping / spec commits)
- 총 insight 문서: 17 (영어) + 17 (한국어) = 34 페어 문서
  (research_overview, session summaries, m6 r1-r6, m8 a/c/d/e, m9,
  encoder_saturation_paper, sec4_2/4_3/4_7/4_10/4_11, m5/m4 series)
- 총 figure: 33+ (프로젝트 횡단); 이 2일 run 에서 8 추가
- 총 notebook: 13 (프로젝트 횡단); 1 신규 (attention_viz.ipynb) +
  1 확장 (encoder_saturation_chain.ipynb). §4 follow-up 은 프로젝트
  convention 상 reproduction notebook 안 함.
- Scorer **3회 확장**: English-only → English+Korean →
  English+Korean+Japanese → English+Korean+Japanese+Chinese.
  51 PMR regression 케이스 (세션 시작 시 28), 모두 통과.
- 신규 design spec: 1 (`docs/superpowers/specs/2026-04-26-sec4_6-
  counterfactual-stim-design.md` — 사용자 review 대기).
- pytest: 51/51 PMR 케이스 (이전 28/28). 다른 테스트 변화 없음.
