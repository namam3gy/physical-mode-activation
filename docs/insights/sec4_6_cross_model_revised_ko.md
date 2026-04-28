---
section: §4.6 cross-model — REVISED (2026-04-26 overnight; 2026-04-27 오후 3-모델; **2026-04-27 overnight 5-모델 n=10**)
date: 2026-04-28 (overnight chain 21:42 KST 종료)
status: complete (5-architecture layer sweep — Qwen + LLaVA-1.5 + LLaVA-Next + Idefics2 + InternVL3)
hypothesis: 픽셀-인코드 가능성은 architecture-conditional, 모델별 shortcut-layer profile이 다름. Encoder saturation 은 shortcut breadth 와 *상관* (Qwen 넓음, LLaVA-1.5 좁음) 이지만 **엄밀히 인과적이지 않음** — Idefics2 (saturated SigLIP-SO400M) 는 projection 깨끗하게 상승함에도 어느 layer 에서도 shortcut 없음, InternVL3 는 protocol baseline 포화 (`line_blank_none_fall_*` 에서 PMR=1.0).
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

2026-04-27 overnight chain 이 §4.6 layer sweep 을 5 architecture × n=10
stim/layer 로 확장. 이전 3-모델 주장 ("shortcut layer 개수가 encoder
saturation 따라감, AUC 0.73 → 0.81 → 0.99 = 1 → 2 → 4-5 layer") 은
**일반화되지 않음**: Idefics2 (SigLIP-SO400M, AUC 0.93) 은 v_L
projection 깨끗하게 상승함에도 PMR flip 본질적으로 0; InternVL3
(InternViT, AUC 0.89) 은 protocol baseline 포화. 업데이트된 reading:
**픽셀-인코드 가능성은 architecture-conditional, 모델별 shortcut
profile 이 다름; encoder saturation 은 CLIP/SigLIP+Qwen 샘플 안에서는
breadth 와 상관이지만 유일 결정 요인 아님**.

![Cross-model layer sweep](../figures/sec4_6_cross_model_layer_sweep.png)

**5-모델 PMR flip rate at eps=0.1 (v_unit_<L> vs random control), n=10 stim per layer**:

| Layer | LLaVA-1.5 (CLIP+Vicuna) | LLaVA-Next (CLIP+AnyRes+Mistral) | Qwen2.5-VL (SigLIP+Qwen2) | Idefics2 (SigLIP-SO400M+Mistral) | InternVL3 (InternViT+InternLM3) |
|------:|------------------------:|---------------------------------:|--------------------------:|---------------------------------:|--------------------------------:|
|   5   | 0/10                    | 0/10                              | **8/10 (80 %)**           | 0/10                              | (baseline=1)                    |
|  10   | 0/10                    | 0/10                              | **10/10 (100 %)**         | 0/10                              | (baseline=1)                    |
|  15   | 0/10                    | 3/10                              | **8/10 (80 %)**           | 0/10                              | (baseline=1)                    |
|  20   | 1/10                    | **10/10 (100 %)**                 | **8/10 (80 %)**           | 0/10                              | (baseline=1)                    |
|  25   | **4/10 (40 %)**         | **10/10 (100 %)**                 | **9/10 (90 %)**           | 1/10 (10 %)                       | (baseline=1)                    |
| random eps=0.1 (모든 layer 합산) | 0/50 | 0/50 | 1/50 (Qwen L10 만 — 10 % rate, Wilson [0.02, 0.40], v_unit 100 % 보다 훨씬 낮음) | 0/50 | 0/50 (no-op — baseline 이미 physics) |

(굵게 = n=10 에서 Wilson lower bound > random upper bound 인 깨끗한 shortcut layer.)

### Architecture-level 패턴 (n=10 paper-grade reading)

- **Qwen2.5-VL** (SigLIP+Qwen2-7B): L5/L10/L15/L20/L25 모두 broad
  shortcut (Wilson lower bound 0.49–0.72).
- **LLaVA-1.5** (CLIP-ViT-L+Vicuna-7B): L25 single shortcut (4/10,
  Wilson [0.17, 0.69]) — n=5 4/5 (80 %) 주장이 n=10 에서 40 % 로
  약화.
- **LLaVA-Next** (CLIP+AnyRes+Mistral-7B): L20 + L25 shortcut (둘 다
  10/10).
- **Idefics2** (SigLIP-SO400M+Mistral-7B): **테스트한 어떤 layer 에도
  shortcut 없음**. Gradient ascent 에서 v_L projection 모든 run
  ~+38 상승 (즉 morning §4.6 의 "projection 수준 vs 행동 수준
  분리" 패턴 재현). L25 Wilson upper bound 0.40. 가능: Idefics2 의
  shortcut 이 L26-31 (Mistral 32 layer 중 미테스트) 또는 이 eps 에서
  픽셀-공간 shortcut 없음.
- **InternVL3** (InternViT+InternLM3-8B): protocol-saturated —
  `line_blank_none_fall_*` 가 §4.6 "circle" 프롬프트 하에 baseline_pmr
  =1.0 (10/10), abstract→physics flip 측정 불가. mvp_full label-free
  예측에서 어떤 (object × bg × cue) cell 도 InternVL3 가 reliably
  abstract 라고 응답하지 않음 (최저 `filled_blank_none_fall` 0.6).

### 새 데이터가 수정하는 것

1. **n=3-5 의 "100 %" 주장이 n=10 에서 약화**. LLaVA-1.5 L25 4/5 →
   4/10 (Wilson [0.17, 0.69]); Qwen L15/L20 3/3 → 8/10. 가장 깨끗한
   주장의 Wilson lower bound 가 이제 0.49 (Qwen L5) ~ 0.72 (Qwen
   L10, LLaVA-Next L20/L25).
2. **Encoder-saturation → shortcut-breadth 스케일링은 *상관, 법칙
   아님*.** Idefics2 가 strict 스케일링을 falsify: CLIP+Mistral 안에서
   가장 saturated (AUC 0.93, PMR 0.97) 이지만 깨끗한 shortcut layer
   0개. 3-모델 1<2<4-5 ordering 은 CLIP-LLaVA + SigLIP-Qwen 부분집합
   에서만 성립.
3. **"Wrong-layer-choice" 가설이 cross-model 에 여전히 적용** —
   Idefics2 의 projection 모든 layer 에서 ~+38 상승하지만 PMR flip
   안됨, morning LLaVA-1.5 L10 null mirror. L26-L31 미테스트 (해당
   깊이 LM hidden capture 없음).
4. **InternVL3 는 다른 baseline stim 필요** — pure-line + blank bg 가
   InternVL3 의 saturated InternViT+InternLM3 pipeline 에 너무 쉬움.
   in-distribution stim 중 abstract-baseline 인 것이 있는지 open.

Random-direction control 은 5 모델 전부의 깨끗한 shortcut layer 에서
0/50 (단일 예외: Qwen L10 random 1/10, v_unit 10/10 보다 훨씬 낮음) —
방향 특이성 보존.

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

### 수정된 reading (이 문서, 2026-04-27)
- **픽셀-인코드 가능성은 architecture-conditional**, 각 모델이 1개
  이상의 shortcut LM layer 보유. *개수* 가 encoder saturation 을
  따라감.
- LLaVA-1.5 는 **1** shortcut layer (L25). LLaVA-Next 는 **2** (L20,
  L25). Qwen 은 **4-5** (L10/L15/L20/L25 clean + L5 marginal).
- 이전 LLaVA-1.5 L10 null 은 **wrong-layer-choice** artifact: CLIP-
  encoder 모델은 shortcut layer 가 LM 의 깊은 위치에 집중되고, SigLIP-
  encoder Qwen 은 LM 전반에 분산.
- **H-shortcut** 은 supported 유지 + **강화** — 픽셀-인코드 가능성은
  encoder→LM pipeline 의 일반적 속성, architecture-dependent locus +
  breadth.

### 변경 없는 부분
- M9 PMR-천장과 §4.7 결정-안정성 천장은 *별개* saturation 시그니처 —
  §4.6 의 모델별 layer 선택과 무관. Architecture-level reframe 견고.
- v_L 방향 특이성 (encoding direction matters; random 은 안 함)
  LLaVA-1.5 의 모든 테스트 레이어에서 보존.

## 왜 중요한가

5-모델 n=10 sweep 이 이전 "capacity-scales-with-saturation" 주장을
더 nuanced 한 그림으로 교체:

| Model | Vision encoder | LM | PMR(_nolabel) M2 | Behavioral-y AUC | 깨끗한 shortcut LM layer (n=10) |
|-------|----------------|----|-----------------:|-----------------:|-------------------------------:|
| Qwen2.5-VL-7B | SigLIP | Qwen2-7B | 0.94 | 0.99 | 5 (L5/10/15/20/25) |
| LLaVA-Next-Mistral-7B | CLIP+AnyRes | Mistral-7B | 0.79 | 0.81 | 2 (L20, L25) |
| LLaVA-1.5-7B | CLIP-ViT-L | Vicuna-7B | 0.38 | 0.73 | 1 (L25, 약함 — 4/10) |
| Idefics2-8B | SigLIP-SO400M | Mistral-7B | 0.97 | 0.93 | **0 (anomaly)** |
| InternVL3-8B | InternViT | InternLM3-8B | 0.99 | 0.89 | (untestable — baseline=1.0) |

CLIP+Qwen 부분집합 (LLaVA-1.5, LLaVA-Next, Qwen) 은 saturation-tracks-
breadth ordering (1 < 2 < 5) 따름. 그러나 **Idefics2 가 universal
주장을 falsify**: LLaVA-Next 보다 saturation 높지만 동일 eps=0.1, layer
5-25 테스트 벤치에서 깨끗한 shortcut 0. 따라서 메커니즘은 *"encoder
saturation alone"* 아님; LM family + projector 설계도 매터.

Idefics2 anomaly 의 두 후보 해석:

(a) **Wrong-relative-depth**. Idefics2 + Mistral-7B 의 shortcut layer 가
    L26-31 일 수 있음 (Mistral 32 layer; L25 = 78 % depth 까지 테스트).
    Morning §4.6 LLaVA-1.5 L10 null 이 정확히 이 종류의 artifact 였음.
    해결하려면 더 깊은 Mistral layer 에 새 LM activation capture +
    L28/L30 v_L 추출 + sweep 필요.

(b) **Idefics2 perceiver-resampler 병목**. Idefics2 의 projector 는
    perceiver resampler — 64 tile-token 을 고정 visual-token 예산으로
    압축 (Qwen / LLaVA 의 MLP projector 와 질적으로 다름). 이 병목이
    `v_L`-aligned 정보가 LM 에 도달하기 전에 strip 가능, 픽셀-공간
    gradient ascent 가 flipping perturbation 못 찾게 만듦.

두 해석 분리 테스트 가능; 현재 데이터로는 구별 안됨.

InternVL3 결과는 순수 protocol 한계: §4.6 "circle" 프롬프트 (자체가
physics-mode prime) 에서 `line_blank_none_fall_*` 도 InternVL3 의 ladder
에서 baseline PMR=1.0. mvp_full 예측이 (cell × label-free) 조건 중
InternVL3 가 reliably abstract 라고 답하는 곳 없음 — 즉 이 protocol 이
InternVL3 에서 flip 할 수 있는 in-distribution `line_blank` 스타일
baseline 이 존재하지 않음.

**업데이트된 H-shortcut framing (이번 라운드)**:
- 픽셀-인코드 가능성은 **architecture-conditional**, 모델별 shortcut
  profile 다름.
- Encoder saturation 은 CLIP+Qwen 모델 안에서는 shortcut breadth 와
  상관, 그러나 **엄밀히 인과적이지 않음** (Idefics2 falsify).
- "Shortcut layer 의 개수" ordering 은 *CLIP/SigLIP+Qwen 부분집합 안
  에서의 경험적 패턴* 으로 defensible, 그러나 universal 스케일링
  법칙 아님.

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

1. ~~**LLaVA-1.5 만 layer sweep**.~~ ~~**LLaVA-Next 2026-04-27 해결**~~.
   **Idefics2 + InternVL3 도 확장** (`counterfactual_idefics2.py` —
   5-tile + pixel_attention_mask, `counterfactual_internvl3.py` —
   single 448×448 tile).
2. ~~**Qwen layer sweep 부재**.~~ ~~**2026-04-27 해결**.~~ 다른 4
   모델과 cross-model 비교 가능하게 n=10 으로 재실행.
3. ~~**LLaVA-Next n=3 stim per layer**.~~ **n=10 에서 해결** (L20/L25
   10/10 에서 Wilson [0.69, 1.00]). 다른 모델의 n=3-5 100 % 주장의
   over-confidence 노출 (LLaVA-1.5 L25 4/5 → 4/10, Qwen L15/L20
   3/3 → 8/10).
4. **Idefics2 의 더 깊은 layer (L26-31) 미테스트**. Idefics2 의 M2 LM
   activation capture 는 L5/10/15/20/25 만 저장
   (`configs/cross_model_idefics2.py::capture_lm_layers` 기준). L28-30
   shortcut 테스트하려면 해당 깊이의 활성화 재캡처 + v_L 재추출 필요
   (~50 분 추론 + steering 추출).
5. **InternVL3 는 §4.6 프롬프트 protocol 하에 abstract-baseline stim
   없음**. `line_blank_none_fall_*` 가 baseline=1.0; mvp_full label-
   free run 의 lowest-PMR cell `filled_blank_none_fall` 도 0.6 — 여전히
   대부분 physics-mode. 프롬프트 ("circle" priming 제거) 또는 stim
   (InternVL3 에서 진짜로 abstract-baseline 인 카테고리 찾기) 재설계
   필요.
6. **Idefics2 projection-vs-behavior 분리**. Gradient ascent 가 모든
   테스트 layer 에서 v_L projection ~+38 상승 (Qwen / LLaVA-Next
   magnitude 매칭) 그러나 PMR flip 안됨. Morning §4.6 LLaVA-1.5 L10
   null 의 mirror. 본문에 두 후보 설명 — 둘 다 미테스트.
7. **LLaVA-Next pixel_values 재구성 1-tile 만**. `synthesized.png`
   round-trip 이 LLaVA-Next AnyRes 에서 spatial 정보 손실; PMR 재추론
   은 첫 tile 만으로 gradient-altered 신호 carry 하기에 작동. Paper-
   grade rigor 에는 direct-tensor 재추론 (processor 의 re-AnyRes 우회)
   이 더 깨끗.
8. **단일 방향 (class-mean v_L)**. SAE / multi-axis decomposition 으로
   다른 픽셀-인코드 가능 방향이 어느 layer 에든 추가로 있을 수 있음.
9. **Single-task 평가**. 다른 shortcut 행동은 다른 layer 에 localize
   될 수 있음.

## 재현

5-모델 n=10 sweep 은 chained overnight 작업으로 launch:

```bash
# Single chain: B (Qwen + LLaVA-1.5 + LLaVA-Next n=10) → A (Idefics2 +
# InternVL3 n=10) → C (Qwen 32B M2 inference). H200 에서 ~4 시간 wall.
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/overnight_b_a_c.sh \
    > /tmp/overnight_chain.log 2>&1 &

# Per-stage breakdown (개별 실행 가능):
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_qwen_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_qwen_layer_sweep_n10_<ts>
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_llava15_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_llava_next_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_idefics2_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_internvl3_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...

# Sweep 별 PMR 재채점 (synthesized.png + baseline.png 페어 사용).
for d in outputs/sec4_6_*_layer_sweep_n10_*/; do
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_summarize.py \
        --run-dir "$d" --model-id <matching-model-id>
done

# 5-모델 통합 + figure.
uv run python scripts/sec4_6_cross_model_layer_summary.py
```

## Artifacts

- 모델별 gradient-ascent 모듈:
  `src/physical_mode/synthesis/counterfactual_llava_next.py`,
  `counterfactual_idefics2.py` (5-tile + mask),
  `counterfactual_internvl3.py` (single 448×448).
- 모델별 layer-sweep driver:
  `scripts/sec4_6_{qwen,llava15,llava_next,idefics2,internvl3}_layer_sweep_unified.py`.
- 통합기: `scripts/sec4_6_cross_model_layer_summary.py` (5 latest
  `_n10_<ts>` sweep 자동 발견 확장).
- Overnight chain: `scripts/overnight_b_a_c.sh`.
- Sweep 출력: `outputs/sec4_6_{qwen,llava15,llava_next,idefics2,internvl3}_layer_sweep_n10_<ts>/`.
- 5-모델 테이블 + figure:
  `outputs/sec4_6_cross_model_layer_summary/cross_model_layer_table.csv`,
  `docs/figures/sec4_6_cross_model_layer_sweep.png`.
- 사전: M8a/M2 captures, 모델별 v_L `outputs/cross_model_*_capture_*/probing_steering/steering_vectors.npz`,
  `scripts/m8a_extract_per_model_steering.py`.

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
