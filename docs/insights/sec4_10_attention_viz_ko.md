---
section: §4.10
date: 2026-04-25
status: complete (초기 release)
scope: Qwen2.5-VL 만 — 논문 부록 figure 인프라
---

# §4.10 — Attention 시각화 UI

## 목적

M3 / M4 / M6 의 per-layer probe AUC 수치에 대한 정성적 보완: attention
heatmap 이 LM 이 생성을 commit 할 때 **어디** 를 보는지 layer 별로 보여
줌. Probe 는 *어떤 정보* 가 인코더 표현에 있는지 알려주고, attention map
은 *모델이 응답 전에 무엇에 주목하는지* 시사.

## 캡처 인프라

`src/physical_mode/models/vlm_runner.py` 에 작은 변경 필요: `capture_lm_
attentions=True` 일 때 모델을 `attn_implementation="eager"` 로 (대신
`"sdpa"`) 로드. SDPA 는 attention weight 를 반환 안함 — SDPA 캡처는
조용히 빈 `lm_attn_*` tensor 를 생성. Toggle 자동 — config flag 만 설정
하면 runner 가 알아서 처리.

Qwen2.5-VL-7B 캡처 비용 (M8a stim, limit=20 stim × 3 라벨): ~30초 wall
time, safetensors 파일당 ~7 MB (4 캡처 layer 에서 28 head × 390 q × 390 k
attention, fp16).

## 저장된 tensor

각 `outputs/attention_viz_qwen_<ts>/activations/<sample_id>.safetensors`:

| key                  | shape          | dtype         |
|----------------------|----------------|---------------|
| `lm_attn_5/15/20/25` | (28, 390, 390) | torch.float16 |
| `lm_hidden_5/15/20/25`| (324, 3584)   | torch.bfloat16|
| `visual_token_mask`  | (390,)         | torch.uint8   |
| `input_ids`          | (390,)         | torch.int64   |

Visual token 이 위치 37–360 점유 (연속) — 256×256 M8a stim 에서 324
token = 18×18 patch grid.

## 노트북 구조 (`notebooks/attention_viz.ipynb`)

6 섹션 + 재현:
1. **캡처 로드 + shape 검사** — `load_capture(sample_id)` helper.
2. **Per-layer heatmap (head 평균)** — 마지막 input token → 시각 grid,
   4 캡처 layer 횡단 side-by-side.
3. **원본 이미지 위 overlay** — 18×18 grid 를 image 해상도로 upsample,
   alpha-blend.
4. **Physics-mode vs abstract-mode 비교** — PMR=1 예시 + PMR=0 예시
   (label=ball arm) 선택.
5. **Per-head 미세 구조** — 단일 layer 의 28 head 를 grid (4 행 × 7 열)
   로 표시.
6. **집계** — 캡처된 subset 횡단 layer × PMR 별 시각-token attention
   entropy.

섹션 6 이 정량적 주장에 가장 가까움: PMR=1 stim 이 후기 layer 에서
PMR=0 stim 보다 **낮은** attention entropy 를 보이면, "physics-mode
commit 이 집중된 시각 attention 과 상관" 시사. 초기 run 이 bias 방향
보여주나, arm 당 n=20 stim 은 유의성 검증에 부족.

## 논문에 주는 것

부록 figure 의 정성적 자료:
- 모델이 후기 layer 에서 관련 patch (예: 객체 실루엣 + ground plane) 에
  attend 함 확인
- Head 특화 보여줌 (per-head heatmap 이 몇 head 가 시각 binding 의 대부분
  을 carry 함을 드러냄, mech-interp 의 sparse attention 작업과 일치)
- Attention-knockout / SIP / SAE 분해 (M5b) 미래 작업의 hook 제공

## Cross-model 확장 (같은 날 추가)

LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 가 같은 M8a subset 에서 캡처
(limit=20, layer 5/15/20/25). 총 디스크 비용 ~2.5 GB (LLaVA-Next 가 AnyRes
5-tile attention 으로 파일당 ~480 MB 차지).

### 시각 token 수가 architecture 마다 다름

| Model       | 시각 token | Grid layout       | Encoder        |
|-------------|----------:|-------------------|----------------|
| Qwen2.5-VL  | 324       | 18×18 (1 tile)    | SigLIP         |
| LLaVA-1.5   | 576       | 24×24 (1 tile)    | CLIP-ViT-L     |
| LLaVA-Next  | 2928      | 5-tile AnyRes     | CLIP-ViT-L     |
| Idefics2    | 320       | non-square split  | SigLIP-SO400M  |
| InternVL3   | 256       | 16×16 (1 tile)    | InternViT      |

### Cross-model 마지막 token 의 시각 token 으로 향하는 attention 비율

| Model       | layer 5 | layer 15 | layer 20 | layer 25 | n_visual / seq_len |
|-------------|--------:|---------:|---------:|---------:|-------------------:|
| Qwen2.5-VL  | 0.030   | 0.044    | **0.146**| 0.102    | 0.831              |
| LLaVA-1.5   | 0.049   | 0.097    | **0.143**| 0.059    | 0.901              |
| LLaVA-Next  | 0.187   | **0.206**| 0.085    | 0.090    | 0.976              |
| Idefics2    | 0.106   | **0.256**| 0.161    | 0.060    | 0.823              |
| InternVL3   | 0.033   | 0.036    | **0.169**| 0.048    | 0.793              |

**핵심 발견**: 시각 token 이 입력 시퀀스의 79–98% 를 차지함에도 5 VLM 모두
시각 token 에 단지 **3–26%** 의 attention 만 할당. LM 의 마지막 token
attention 은 최근 prompt + system token 이 지배. 시각 attention 은 모든
모델에서 mid-layer (15 or 20) 에 정점 — M4 가 label-physics margin 발달
관찰한 layer band 와 동일.

이는 architecture-level reframe 과 일치: encoder 출력은 균일 (stim-y AUC
= 1.0) 이지만 LM 은 mid-layer 에서 짧게 "보고", attention 대역폭의 대부분
을 언어적 맥락에 할당. **architecture 차이가 짧은 시각 peek 가 PMR commit
으로 번역되는 강도를 형성** — 시각 정보가 존재하느냐 여부가 아님.

### Heatmap overlay (square-grid 모델)

노트북 섹션 8 이 3 square-grid 모델 (Qwen 18×18 / LLaVA-1.5 24×24 /
InternVL3 16×16) 의 layer 20 overlay 를 같은 stim 에서 side-by-side 표시.
시각 attention 이 saturated 모델에서 객체 실루엣 + ground plane 영역에
집중.

LLaVA-Next AnyRes (5 tile) 와 Idefics2 (non-square split) 는 overlay viz
에서 생략 — multi-tile 구조라 단일 grid 가 아닌 per-tile 분해 필요.

## 한계

1. **제한된 stim subset (n=20)**. 캡처가 대표 slice 다루나 통계적으로
   힘이 부족.
2. **Attention 은 메커니즘의 한 슬라이스만**. Attention 이 무엇을 읽는지
   에 대한 인과적 주장에는 activation patching (M5b) 또는 SAE 분해가 필요.
3. **Eager attention 이 SDPA 보다 느림** — 일회성 캡처에는 OK, 프로덕션
   run 에서는 활성화하지 말 것.
4. **LLaVA-Next + Idefics2 overlay 미구현** — multi-tile 구조라 per-tile
   분해 필요, 이번 초기 round 에는 미포함.
5. **Layer 25 가 모델마다 다른 의미**: Qwen/InternVL3 (총 28 layer) 에서
   late-late, LLaVA-Next/Idefics2 (32 layer) 와 LLaVA-1.5 (32 layer) 에서
   early-late. Cross-model layer 비교는 비공식 — depth 가 정규화 안됨.

## Reproducer

```bash
# 캡처 (5 모델, M8a Qwen stim, H200 에서 각 ~30초)
for cfg in attention_viz_qwen attention_viz_llava attention_viz_llava_next \
           attention_viz_idefics2 attention_viz_internvl3; do
    uv run python scripts/02_run_inference.py \
        --config configs/$cfg.py \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# 노트북 보기
uv run jupyter lab notebooks/attention_viz.ipynb
```

## 산출물

- `configs/attention_viz_{qwen,llava,llava_next,idefics2,internvl3}.py` —
  캡처 config (각: limit=20, capture_lm_attentions=True,
  layers=(5,15,20,25)).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation`
  자동 `"eager"` 전환 (캡처 시).
- `outputs/attention_viz_<model>_<ts>/` — 예측 + 20 × safetensors
  캡처 파일 (크기: Qwen 7 MB, LLaVA-1.5 21 MB, LLaVA-Next 480 MB,
  Idefics2 10 MB, InternVL3 5 MB / file).
- `notebooks/attention_viz.ipynb` — 8-섹션 인터랙티브 viz 노트북
  (섹션 1-6: Qwen 단일 모델; 섹션 7-8: 5-모델 cross-model).
- `docs/insights/sec4_10_attention_viz_ko.md` (이 문서).
