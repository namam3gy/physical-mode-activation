# H-encoder-saturation — 논문용 통합본 (5-model)

**상태**: 2026-04-25 기준 5-model M8a chain 완료 (Qwen, LLaVA-1.5,
LLaVA-Next, Idefics2, InternVL3). 논문 Section 4는 5개 모델 점에서 lock.

## 한 줄 주장

오픈소스 VLM의 behavioral physics-mode reading (PMR) — 최소 합성 stim에서 —
은 encoder representational capacity가 아닌 **architecture 수준**(encoder + LM
fusion)에서 결정된다. 시험된 모든 vision encoder는 physics-vs-abstract stim
범주를 AUC = 1.0으로 linear separation한다. Architecture 별 behavioral
PMR(_nolabel) 사다리는 LM이 encoder 출력을 physics-mode signal로 어떻게
소비하는지를 반영한다 — non-CLIP architecture는 saturate, CLIP-LLaVA-Vicuna는
saturate 안 함, 합성 minimal stim에서.

## 증거 사슬 (서술 순서)

### 1. Behavioral PMR 사다리 (5-model M8a, 각 n=400)

| Model       | Encoder         | LM           | M8a PMR(_nolabel) | 95% CI            |
|-------------|-----------------|--------------|------------------:|-------------------|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | **0.838**         | [0.800, 0.872]    |
| LLaVA-1.5   | CLIP-ViT-L/14   | Vicuna-7B    | **0.175**         | [0.140, 0.212]    |
| LLaVA-Next  | CLIP-ViT-L/14   | Mistral-7B   | **0.700**         | [0.653, 0.743]    |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | **0.882**         | [0.850, 0.912]    |
| InternVL3   | InternViT       | InternLM2-7B | **0.917**         | [0.890, 0.943]    |

3개 non-CLIP 모델이 PMR ~0.84–0.92에서 saturate. LLaVA-1.5는 0.18.
LLaVA-Next는 M8a에서 5번째 모델 점을 추가한다 — LLaVA-1.5와 같은 encoder
계열 (CLIP-ViT-L)이지만 **multi-axis architectural difference** (AnyRes
다중 타일 분할, 다른 fusion projector, 다른 학습 레시피, Mistral-7B LM
포함)를 가진다. CI [0.65, 0.74]는 saturated cluster (Qwen [0.80,0.87] /
Idefics2 [0.85,0.91] / InternVL3 ≈0.92)보다 전부 아래에 있고, LLaVA-1.5의
[0.14, 0.21]보다 전부 위에 있다. 이는 LM-controlled encoder swap이
*아니다*. 그러려면 같은 architecture에서 LM만 swap된 모델이 필요하다.
이를 5번째 관측치로 보고하며 counterfactual로 보고하지 않는다: PMR이
0.18 → 0.70으로 4개 동시 변경 축 (encoder fusion, image tiling, 학습
데이터 + 레시피, LM 계열) 사이에서 이동.

### 2. M9 cross-stim bootstrap CI (합성 vs 사진)

| stim | model       | mean PMR(_nolabel) | 95% bootstrap CI |
|------|-------------|-------------------:|-------------------|
| M8a  | Qwen        | 0.838              | [0.800, 0.872]   |
| M8a  | LLaVA-1.5   | 0.175              | [0.140, 0.212]   |
| M8a  | LLaVA-Next  | 0.700              | [0.653, 0.743]   |
| M8a  | Idefics2    | 0.882              | [0.850, 0.912]   |
| M8a  | InternVL3   | 0.917              | [0.890, 0.943]   |
| M8d  | Qwen        | 0.869              | [0.840, 0.898]   |
| M8d  | LLaVA-1.5   | 0.331              | [0.294, 0.371]   |
| M8d  | LLaVA-Next  | 0.625              | [0.583, 0.667]   |
| M8d  | Idefics2    | 0.890              | [0.862, 0.917]   |
| M8c  | Qwen        | 0.550              | [0.433, 0.667]   |
| M8c  | LLaVA-1.5   | 0.283              | [0.183, 0.383]   |
| M8c  | LLaVA-Next  | 0.417              | [0.300, 0.533]   |
| M8c  | Idefics2    | 0.417              | [0.317, 0.517]   |

합성 M8a에서 4개 PMR cluster가 깔끔하게 분리: LLaVA-1.5 floor [0.14, 0.21]
→ LLaVA-Next mid-band [0.65, 0.74] → saturated non-CLIP cluster [0.80,
0.92]. 동일 encoder 계열 (CLIP-ViT-L)에서 LLaVA-1.5 → LLaVA-Next 점프는
**4개 동시 confound 축에 걸친 0.52 PMR 단위** — LM modulation과 *부합*
하지만 *분리되지 않음*. M8d 에서도 같은 architecture-stratified 순서
유지: LLaVA-Next 0.625 [0.58, 0.67] 이 LLaVA-1.5 0.331 [0.29, 0.37] 과
saturated cluster [0.84, 0.92] 사이에 위치. 사진 (M8c, 4 모델 모두)
에서는 encoder gap 이 [0.18, 0.67] 로 collapse — LLaVA-Next M8c PMR
0.417 이 Idefics2 M8c 0.417 과 *통계적 구분 불가*, **사진이 5번째 모델
에서도 encoder gap 압축**. M8c finding 일반화.

### 3. Vision-encoder probe AUC — apples-to-apples M8a (5개 모델, M8a stim)

| Model       | Encoder         | LM           | M8a behavioral-y AUC | M8a stim-y AUC |
|-------------|-----------------|--------------|---------------------:|---------------:|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | 0.880                | **1.000**      |
| LLaVA-1.5   | CLIP-ViT-L      | Vicuna-7B    | 0.771                | **1.000**      |
| LLaVA-Next  | CLIP-ViT-L      | Mistral-7B   | 0.809                | **1.000**      |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | 0.926                | **1.000**      |
| InternVL3   | InternViT       | InternLM2-7B | 0.886                | **1.000**      |

Behavioral-y AUC (각 모델의 자체 PMR을 target으로) 0.77–0.93 — encoder
계열 패턴처럼 보인다. **하지만 stim-defined y AUC는 5개 encoder 모두에서
1.0**: 모든 encoder가 4개 stim-y target (rendered_vs_line,
physics_cell_vs_abstract_cell, within_line_context, within_textured_context)
에서 factorial cell을 완벽 linear separation한다. Encoder discriminability는
**계열 간 균일** — 2번째 CLIP 점 (LLaVA-Next) 포함, LLaVA-1.5 PMR floor에
대한 CLIP-as-encoder 설명을 배제한다.

### 4. Cross-stim probe — M8c 사진 (n=60)

| Model       | M8c PMR(_nolabel) | M8c behavioral-y AUC | M8c stim-y AUC |
|-------------|------------------:|---------------------:|---------------:|
| Qwen2.5-VL  | 0.550             | 0.582                | **1.000**      |
| LLaVA-1.5   | 0.283             | 0.785                | **0.988**      |
| Idefics2    | 0.417             | 0.745                | **0.992**      |
| InternVL3   | 0.533             | 0.661                | **0.996**      |

**Behavioral-y AUC가 cross-stim에서 반전** (M8a → M8c): Qwen 0.88→0.58,
Idefics2 0.93→0.75, InternVL3 0.89→0.66; LLaVA 0.77→0.79 (안정).
Encoder-behavior alignment는 stim에 따라 다르다.

**Stim-y AUC는 1.0 유지** — encoder discriminability도 stim-invariant.

## 메커니즘 (수정본)

stim-y check 이전 버전의 H-encoder-saturation은 메커니즘을 "encoder 계열
→ encoder probe AUC → behavioral PMR → H7 measurability"로 frame했다.
stim-y check가 다음의 정제를 강제:

```
encoder 계열 + LM 계열
       ↓
joint architecture (encoder + LM fusion)
       ↓
LM 측의 encoder 출력을 physics-mode signal로 읽기
       ↓
behavioral PMR(_nolabel) saturated vs unsaturated
       ↓
H7 measurability gating
```

Encoder representational capacity는 균일; behavioral PMR은 joint encoder+LM
시스템에 의해 결정. "encoder probe AUC with behavioral y"는
*downstream-conditional* 측도 — encoder 표현이 *behavioral* PMR 분포와
얼마나 잘 정렬되는지를 반영하며, encoder discriminability 그 자체는 아니다.

## 가설 상태

- **H-encoder-saturation** — *architecture 수준 cross-stim 확인*.
  5 모델 점 (3 non-CLIP + 2 CLIP) × 2 stim source × 2 y mode;
  "encoder 계열"에서 "joint encoder+LM architecture"로 reframe.
  LLaVA-Next는 LLaVA-1.5와 multi-axis architectural difference를 갖는
  2번째 CLIP 점 추가 (clean LM swap 아님).
- **H-LM-modulation** (M9 유래) — *여전히 시사만*. Idefics2 M8d H7 CI
  [+0.000, +0.094]가 0에 닿음; LLaVA-Next M8d H7 CI [−0.102, −0.006] 은
  0을 ~0.005 만큼만 대칭적으로 배제. **양쪽 모두 noise floor 안** (M9
  부트스트랩 프레임워크 기준). M8d 에서 두-Mistral 의 H7 ≈ 0 클러스터링
  은 시사적이나 multi-axis-confounded (인코더 family, image pipeline,
  projector, 학습 모두 다름). 논문 옹호 불가.
- **H7** (label-selects-regime): LLaVA-1.5 의 unsaturated-only 가 프로젝트
  최강 신호 (M8d +0.31). LLaVA-Next 가 M8a 에서는 H7 보존 (+0.26, 5/5
  PASS) 하지만 **M8d 에서 H7 신호 완전 제거** (-0.05, CI 가 0 바로
  아래). LLaVA-Next 의 M8d PMR 은 천장 한참 아래 (0.625) — 이는 saturation
  효과 아님; 측정 헤드룸이 있어도 아키텍처 변경이 H7 깸. H7 강도는 동일
  encoder family architecture 변경에서 보존되지 않음.

## 한계

1. ~~n=1 CLIP 점~~ → LLaVA-Next (5번째 모델)로 해소. LLaVA-1.5 → LLaVA-Next
   의 0.52 PMR 점프는 architecture-level reframe과 부합하나 **4개 축에
   걸쳐 confound**: AnyRes 다중 타일 이미지 분할, fusion projector, 학습
   데이터+레시피, LM 계열 (Vicuna → Mistral).
2. **동일 encoder LM swap**이 가장 깔끔한 counterfactual. 시험된 어떤
   pair도 encoder + image pipeline + projector + 학습을 일정하게 두고 LM만
   바꾸지 않는다. LLaVA-Next로 LLaVA-1.5 대비 최소 변경은 "encoder + 4
   architecture 축"이며 "encoder + LM only"가 아니다.
3. **카테고리당 n=12 사진 on M8c**는 H7 검출에 underpowered.
4. **합성 stim factorial은 M8a-style** — line/blank/none vs
   textured/ground/both. 실세계 stim 분포는 더 다양하다.

## 로드맵

- §4.5 + M9 + M6 r3 + r4 + r5 + LLaVA-Next = 논문 Section 4 완료.
- M5b (SIP+SAE)는 layer 수준 메커니즘 증거용 — round 7.
- M7 논문 초고.

## 산출물 (통합)

- `docs/insights/m8c_real_photos.md` (M8c 사진 behavioral)
- `docs/insights/m9_generalization_audit.md` (논문 Table 1, bootstrap)
- `docs/insights/encoder_swap_idefics2.md` (§4.5)
- `docs/insights/m6_r3_idefics2_probe.md` (§4.5 ext probe)
- `docs/insights/m6_r4_internvl3_probe.md` (4-model + stim-y check)
- `docs/insights/m6_r5_m8c_photo_probe.md` (cross-stim probe)
- `docs/insights/m6_r6_llava_next.md` (5번째 모델, 2번째 CLIP — LLaVA-Next, multi-axis confound)
- `notebooks/encoder_saturation_chain.ipynb` (재현)
- `docs/figures/encoder_chain_5model.png` (논문 headline figure — 4model 대체)
- `docs/figures/encoder_chain_4model.png` (frozen 4-model 스냅샷, r3/r4/r5 doc 용)
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
- `outputs/m9_audit/m9_table1.csv` 와 `m9_summary.csv`
