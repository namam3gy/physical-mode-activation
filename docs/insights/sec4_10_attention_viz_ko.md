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

## 한계

1. **단일 모델만**. LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 으로
   확장하면 디스크 비용 곱셈 (캡처당 ~7 MB × 60 record × 5 모델 = ~2 GB).
2. **제한된 stim subset (n=20)**. 캡처가 대표 slice 다루나 통계적으로
   힘이 부족.
3. **Attention 은 메커니즘의 한 슬라이스만**. Attention 이 무엇을 읽는지
   에 대한 인과적 주장에는 activation patching (M5b) 또는 SAE 분해가 필요.
4. **Eager attention 이 SDPA 보다 느림** — 일회성 캡처에는 OK, 프로덕션
   run 에서는 활성화하지 말 것.

## Reproducer

```bash
# 캡처 (M8a Qwen stim 사용, ~30초)
uv run python scripts/02_run_inference.py \
    --config configs/attention_viz_qwen.py \
    --stimulus-dir inputs/m8a_qwen_<ts>

# 노트북 보기
uv run jupyter lab notebooks/attention_viz.ipynb
```

## 산출물

- `configs/attention_viz_qwen.py` — 캡처 config (limit=20,
  capture_lm_attentions=True, layers=(5,15,20,25)).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation`
  자동 `"eager"` 전환 (캡처 시).
- `outputs/attention_viz_qwen_<ts>/` — 예측 + 20 × safetensors 캡처
  파일.
- `notebooks/attention_viz.ipynb` — 인터랙티브 viz 노트북.
- `docs/insights/sec4_10_attention_viz_ko.md` (이 문서).
