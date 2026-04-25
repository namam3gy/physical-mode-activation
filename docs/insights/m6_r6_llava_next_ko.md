---
milestone: M6 r6
date: 2026-04-25
status: complete
hypothesis: H-encoder-saturation (architecture-level reframe)
---

# M6 r6 — LLaVA-Next-Mistral 5번째 모델 점 (2번째 CLIP 점)

## 요약

LLaVA-v1.6-Mistral-7b가 encoder-saturation chain의 **5번째 모델 점**을
추가한다. LLaVA-1.5와 동일한 비전 인코더 계열(CLIP-ViT-L/14)을 사용하지만,
Vicuna-7B 대신 Mistral-7B와 짝지어졌고 — 거기에 AnyRes 다중 타일 이미지
분할, 다른 fusion projector, 다른 학습 데이터+레시피가 더해졌다. M8a에서
behavioral PMR(_nolabel) = **0.700, 95% CI [0.65, 0.74]** 로, LLaVA-1.5
바닥 [0.14, 0.21]과 saturated non-CLIP cluster [0.80, 0.92] 사이에 깔끔하게
위치한다.

이 2번째 CLIP 점은 깔끔한 "LM swap" counterfactual이 **아니다** — LLaVA-
1.5와 LLaVA-Next 사이에서 4개 아키텍처 축이 동시에 변한다. 우리는 이를
**비전 인코더 계열만으로는 behavioral PMR을 결정할 수 없다**는 점을
배제하는 5번째 관측치로 보고하며, LM modulation의 인과적 분리로 보고하지
않는다.

## 수치

### M8a behavioral PMR(_nolabel)

| Model       | Encoder       | LM           | M8a PMR | 95% CI         |
|-------------|---------------|--------------|--------:|----------------|
| Qwen2.5-VL  | SigLIP        | Qwen2-7B     | 0.838   | [0.800, 0.872] |
| LLaVA-1.5   | CLIP-ViT-L    | Vicuna-7B    | 0.175   | [0.140, 0.212] |
| **LLaVA-Next** | **CLIP-ViT-L** | **Mistral-7B** | **0.700** | **[0.653, 0.743]** |
| Idefics2    | SigLIP-SO400M | Mistral-7B   | 0.882   | [0.850, 0.912] |
| InternVL3   | InternViT     | InternLM2-7B | 0.917   | [0.890, 0.943] |

CI는 5000-iter prediction-level bootstrap, M9와 동일하게 각 shape 내에서
재샘플링.

### M8a vision-encoder probe AUC

| Model       | behavioral-y AUC (가장 깊은 layer) | stim-y AUC (4 target) |
|-------------|----------------------------------:|----------------------:|
| Qwen2.5-VL  | 0.880                              | 1.000                 |
| LLaVA-1.5   | 0.771                              | 1.000                 |
| **LLaVA-Next** | **0.809**                       | **1.000**             |
| Idefics2    | 0.926                              | 1.000                 |
| InternVL3   | 0.886                              | 1.000                 |

LLaVA-Next behavioral-y AUC at layer 23 = 0.809 — LLaVA-1.5 (0.77)와
saturated cluster (0.88–0.93) 사이로, PMR 순서와 동일하다. Stim-y AUC는
4개 target 모두에서 1.0 (rendered_vs_line, physics_cell_vs_abstract_cell,
within_line_context, within_textured_context). **다른 4 모델과 동일한
결과**: encoder가 physics-cell vs abstract-cell factorial cell을 완벽
linear separation한다. CLIP-ViT-L은 encoder bottleneck이 아니다.

## H-encoder-saturation 함의

2번째 CLIP 점이 가장 명확한 encoder 측 counter-hypothesis를 닫는다:
"CLIP이 그냥 physics-vs-abstract 구분을 표현할 수 없는 것 아닌가?" 표현
가능하다 — 같은 encoder, 다른 downstream architecture로 PMR이
0.18 → 0.70 이동. LLaVA-1.5의 PMR 바닥을 일으키는 무엇이든 **vision
encoder에는 없다**.

남은 미지수는 **confound된 4개 축 중 어느 것이** (AnyRes tiling, fusion
projector, training, LM family) 로드를 짊어지는가다. 분리하려면 동일
architecture에서 LM만 swap한 모델이 필요하지만, 공개된 모델 중에는
없다. 상태: H-encoder-saturation은 **joint-architecture** 수준에서
확인되었으며, LM modulation 가설은 Idefics2 (SigLIP-SO400M+Mistral
PMR 0.88) vs LLaVA-1.5 (CLIP-ViT-L+Vicuna 0.18)에서 시사되지만 가용
데이터로는 분리 불가능.

## Multi-axis confound 읽는 법

LLaVA-1.5 → LLaVA-Next 변경점:
1. **AnyRes 다중 타일 이미지 분할** (5 tile vs 1) — 각 이미지가 더 높은
   유효 해상도로 처리됨. Visual capture에서 5×577 patch token 확인.
2. **Fusion projector** — linear projector → MLP, 다른 파라미터 초기화 +
   학습.
3. **학습 데이터 + 레시피** — LLaVA-Next는 760k example로 학습 (LLaVA-1.5
   158k 대비), 다른 mix (reasoning + chart QA 비중 증가).
4. **LM 계열** — Vicuna-7B → Mistral-7B-Instruct.

각 한 축 (또는 상호작용)이 0.52 PMR 점프를 일으킬 수 있다. 이 행만으로는
"LM이 했다" 주장을 하지 않는다.

## Reproducer

Configs:
- `configs/encoder_swap_llava_next.py` — labeled-arm M8a inference
- `configs/encoder_swap_llava_next_label_free.py` — open-prompt arm

추론 (labeled + label-free):
```bash
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next.py
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next_label_free.py
```

캡처 (Qwen run 의 M8a stim 사용):
```bash
uv run python scripts/04_capture_vision.py \
    --stimulus-dir inputs/m8a_qwen_<ts> \
    --output-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --layers 5,11,17,23 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --torch-dtype bfloat16
```

Probes:
```bash
# Behavioral-y
uv run python scripts/encoder_swap_probe.py \
    --model-name llava_next \
    --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --predictions outputs/encoder_swap_llava_next_m8a_<ts>/predictions.parquet \
    --out-dir outputs/encoder_swap_llava_next_m8a_probe \
    --layers 5,11,17,23 \
    --behavioral-pmr 0.700

# Stim-y (4개 target)
for target in rendered_vs_line physics_cell_vs_abstract_cell \
              within_line_context within_textured_context; do
    uv run python scripts/encoder_swap_probe_stim_y.py \
        --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
        --stim-dir inputs/m8a_qwen_<ts> \
        --layers 5,11,17,23 \
        --target $target \
        --out-dir outputs/encoder_swap_llava_next_m8a_probe_stim_y
done
```

요약 재생성 (5-model figure):
```bash
uv run python scripts/encoder_swap_probe_summary.py \
    --idefics2 outputs/encoder_swap_idefics2_probe \
    --internvl3 outputs/encoder_swap_internvl3_probe \
    --llava-next outputs/encoder_swap_llava_next_m8a_probe \
    --internvl3-pmr 0.917 \
    --llava-next-pmr 0.700 \
    --out-dir outputs/encoder_swap_probe_summary
```

`docs/figures/encoder_chain_5model.png` 출력 (r3/r4/r5 insight doc에서
사용된 4-model figure를 대체).

## Cross-stim 부록 (M8d + M8c, 같은 날 추가)

LLaVA-Next 를 M8d (3 카테고리 × 4 추상화 × 2 bg × 2 cue × 2 events × 5
seeds = 480) 와 M8c (60 사진) 에서도 실행, labeled + label-free, GPU 0
에서 총 ~16분 추론. 3가지 발견:

### 1. PMR mid-band 이 3 stim source 모두에서 유지

| stim | model      | mean PMR(_nolabel) | 95% CI         |
|------|------------|-------------------:|----------------|
| M8a  | LLaVA-1.5  | 0.175              | [0.140, 0.212] |
| M8a  | LLaVA-Next | **0.700**          | [0.653, 0.743] |
| M8a  | Idefics2   | 0.882              | [0.850, 0.912] |
| M8d  | LLaVA-1.5  | 0.331              | [0.294, 0.371] |
| M8d  | LLaVA-Next | **0.625**          | [0.583, 0.667] |
| M8d  | Idefics2   | 0.890              | [0.862, 0.917] |
| M8c  | LLaVA-1.5  | 0.283              | [0.183, 0.383] |
| M8c  | LLaVA-Next | **0.417**          | [0.300, 0.533] |
| M8c  | Idefics2   | 0.417              | [0.317, 0.517] |

LLaVA-Next 가 두 합성 stim source 모두에서 **LLaVA-1.5 바닥과 saturated
cluster 사이에 위치** (M8a CI 가 LLaVA-1.5 위 + Idefics2 아래로 분리;
M8d 도 비슷). LLaVA-1.5 → LLaVA-Next 의 0.30–0.52 PMR 점프는 합성 stim
간 일관 — 아키텍처 차이가 stim 모양과 무관하게 같은 방향으로 PMR 이동.

### 2. 사진 collapse (M8c) 가 LLaVA-Next 에도 적중

LLaVA-Next M8c PMR(_nolabel) = **0.417**, Idefics2 M8c (0.417) 와 통계적
구분 불가. 합성 cluster 를 3 PMR 밴드 (0.18 / 0.70 / 0.88) 로 분리하던
인코더 갭이 사진에서 [0.18, 0.67] 단일 밴드로 압축, **M8c finding (M6 r5)
의 예측대로**. 5번째 모델 점이 같은 M8c-collapse 패턴에 부합.

### 3. LLaVA family 에서 H7 가 아키텍처 횡단 collapse

| stim | model      | mean H7 (phys − abs) | 95% CI            |
|------|------------|---------------------:|-------------------|
| M8a  | LLaVA-1.5  | +0.360               | [+0.300, +0.418]  |
| M8a  | LLaVA-Next | +0.260               | [+0.205, +0.317]  |
| M8d  | LLaVA-1.5  | +0.306               | [+0.250, +0.360]  |
| M8d  | LLaVA-Next | **−0.054**           | [−0.102, −0.006]  |
| M8c  | LLaVA-1.5  | +0.100               | [−0.033, +0.233]  |
| M8c  | LLaVA-Next | +0.017               | [−0.133, +0.167]  |

LLaVA-1.5 가 M8d 에서 프로젝트 최강 H7 (+0.31). LLaVA-Next 아키텍처
스위치가 H7 강하게 약화: M8a +0.26 (여전히 5/5 PASS, mid-strong), M8d
−0.05 (CI 가 0 바로 아래), M8c +0.02. **H7 강도가 동일 encoder family
의 architecture 변경에서 보존되지 않음** — architecture-level reframe 과
부합.

**Advisor 경고**: M8d −0.054 효과 크기가 프로젝트 noise floor 안 (CI 가
0 을 ~0.005 만큼만 배제). **Idefics2 M8d +0.048 과 대칭** — 양쪽 모두
"0 바로 위/아래"이며 M9 부트스트랩 프레임워크에서 "시사만 / 논문 옹호
불가"로 강등. **"Mistral-7B 가 H7 을 억제"로 해석하지 말 것** — Idefics2
vs LLaVA-Next 는 LM 외에도 인코더 family, image pipeline (no AnyRes vs
AnyRes), fusion projector, 학습이 모두 다름. 두-Mistral 클러스터링은
시사적이나 multi-axis-confounded, M6 r6 main result 와 같은 caveat.

### 가설 함의

- **H-encoder-saturation** (architecture-level): 5번째 모델 점에서
  cross-stim 확인. LLaVA-Next 의 PMR mid-band 가 M8a + M8d 에서 유지;
  사진 collapse 가 5 모델 모두 적중.
- **H7** (label-selects-regime): LLaVA-1.5 의 unsaturated-only 가 프로젝트
  최강 신호였음. LLaVA-Next 가 그 깔끔함을 제거 — PMR 측정 헤드룸이
  있어도 (M8d PMR 0.625 는 천장 한참 아래) 아키텍처 스위치가 H7 깸.
- **H-LM-modulation**: 여전히 시사만. 승급 안 함.

### Cross-stim reproducer

```bash
# M8d
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8d.py --stimulus-dir inputs/m8d_qwen_<ts>
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8d_label_free.py --stimulus-dir inputs/m8d_qwen_<ts>
# M8c
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8c.py --stimulus-dir inputs/m8c_photos_<ts>
uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_m8c_label_free.py --stimulus-dir inputs/m8c_photos_<ts>

# M9 audit 가 4-모델 × 3-stim 테이블에 LLaVA-Next 추가 재실행
uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit
```

## 산출물

- `outputs/encoder_swap_llava_next_m8a_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8d_<ts>/predictions.{jsonl,parquet,csv}` (cross-stim 부록)
- `outputs/encoder_swap_llava_next_m8d_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8c_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8c_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_vision_activations/*.safetensors` (400 stim × 4 layer)
- `outputs/encoder_swap_llava_next_m8a_probe/{layer_sweep,by_object_level}.csv`
- `outputs/encoder_swap_llava_next_m8a_probe_stim_y/layer_sweep_stim_y_*.csv`
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
- `outputs/m9_audit/m9_table1.csv`, `m9_summary.csv` (5-모델 × 3-stim, 부트스트랩 CI)
- `docs/figures/encoder_chain_5model.png`
- `docs/figures/m9_summary.png`, `m9_table1_heatmap.png` (5-모델 부트스트랩 막대 + 히트맵)
