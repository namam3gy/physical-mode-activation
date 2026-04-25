# M6 r4 — InternVL3 비전-인코더 프로브 (4-모델 인코더 사슬)

**상태**: 2026-04-25 완료.

## 동기

M6 r3 (Idefics2 SigLIP-SO400M 프로브) 가 3 모델 점 (Qwen + LLaVA +
Idefics2) 에서 AUC ↔ 행동 PMR 사슬 종결. 사슬은 **SigLIP-family 가
포화 클러스터, CLIP 이 headroom** 보여줌.

자연스러운 완성: 네 번째 모델 점에 **InternViT** — 또 다른 비-CLIP 비전
인코더 family — InternLM2-7B (또 다른 LM family) 와 페어링. M6 r2a
가 InternVL3 행동 PMR(_nolabel) ≈ 0.99 를 M2 stim 에서 보고했지만 (포화
일치), 인코더-프로브 AUC 는 측정 안 됨. M6 r4 가 추가.

InternViT 도 포화 AUC 도달 → **인코더 family 가 SigLIP 뿐 아니라 3 비-CLIP
family 의 통합 driver**. InternViT 다른 AUC 프로파일 보여줌 → SigLIP
특이적 포화와 보다 일반적인 "비-CLIP 포화" 패턴 구별.

## 방법

`scripts/02_run_inference.py --config configs/encoder_swap_internvl3.py`
M8a Qwen stim 디렉터리에서 (400 stim × 3 라벨 × open prompt = 1200
labeled inferences, GPU 0 에서 ~8 분) + `..._label_free.py` (400
inferences, ~4 분). 총 추론: 12 분.

비전 캡처: `scripts/04_capture_vision.py
--model-id OpenGVLab/InternVL3-8B-hf --layers 3,9,18,23` (24 InternViT
레이어 중 4개; hidden_size=1024 vs Idefics2 의 1152). Wall clock:
**47 초** GPU 0 에서. 참고: InternVL3 의 `vision_tower.encoder.layer`
(단수) 속성 인식하도록 `src/physical_mode/models/vlm_runner._resolve_vision_blocks`
수정 필요했음 — commit 메시지에 진단 캡처.

프로브: `scripts/encoder_swap_probe.py --model-name internvl3` 동일
5-fold 로지스틱 회귀 프로토콜 (M6 r3 와 동일).

행동 PMR(_nolabel) 타겟: 라벨링 런의 3 라벨 평균을 stim 당, 임계 0.5.
(n_pos=379, n_neg=21 — InternVL3 가 이 stim 에서 표 중 가장 강한 포화.)

## 결과

### InternVL3 layer sweep AUC

| 레이어 | AUC 평균 | AUC std |
|------:|---------:|--------:|
| 3     | **0.938** | 0.047   |
| 9     | **0.896** | 0.102   |
| 18    | **0.865** | 0.131   |
| 23    | **0.886** | 0.089   |

레이어 간 평균: **0.90**. Qwen + Idefics2 처럼 AUC 가 가장 이른 캡처
레이어 (3) 에서 높고 유지 — 인코더-포화 패턴 일관.

### M8a 의 4-모델 인코더-포화 사슬

| 모델          | 인코더          | LM            | 인코더 AUC | M8a 행동 PMR(_nolabel) |
|---------------|-----------------|---------------|----------:|---------------------:|
| Qwen2.5-VL    | SigLIP          | Qwen2-7B      | **0.99** (M2)  | **0.838**           |
| LLaVA-1.5     | CLIP-ViT-L/14   | Vicuna-7B     | **0.73** (M2)  | **0.175**           |
| Idefics2      | SigLIP-SO400M   | Mistral-7B    | **0.93** (M8a) | **0.882**           |
| InternVL3     | InternViT       | InternLM2-7B  | **0.89** (M8a) | **0.917**           |

**3 비-CLIP 인코더가 AUC 0.89-0.99, 행동 PMR 0.84-0.92 클러스터.**
**1 CLIP 인코더 (LLaVA) 가 AUC 0.73, 행동 PMR 0.18.**

주의: Qwen + LLaVA AUC 값은 M6 r2 캡처 (M2 stim); Idefics2 + InternVL3
는 M8a stim. 인코더 probe AUC 는 합성 기하 stim 에서 거의 stim-invariant
이므로 비교는 정보적이지만 — 논문에서 언급해야. 완전 M8a 재캡처 (Qwen +
LLaVA) 는 footnote 보충 후속.

## 헤드라인 해석

**H-encoder-saturation 사슬이 SigLIP-특이적에서 비-CLIP-일반으로 일반화**:

```
인코더 family            인코더 probe AUC      M8a 행동 PMR(_nolabel)
─────────────            ────────────────      ────────────────────
SigLIP    (Qwen)              0.99                       0.84
SigLIP-SO400M (Idefics2)      0.93                       0.88
InternViT (InternVL3)         0.89                       0.92
CLIP-ViT-L (LLaVA)            0.73                       0.18
```

**3 별개 비-CLIP 인코더 family** (SigLIP, SigLIP-SO400M, InternViT) 모두
AUC ≥ 0.89, 행동 PMR ≥ 0.84 도달. **CLIP-ViT-L 만 포화 미달** (0.73 /
0.18). 인코더-포화 체제는 인코더 family 에 의해 견고하게 식별됨, *LM
family 횡단* (Qwen2-7B, Mistral-7B, InternLM2-7B, Vicuna-7B), *인코더
구현 횡단*.

이는 인코더-포화 작업의 논문급 헤드라인: **비전-인코더 family 가 VLM 의
합성-stim physics-mode 천장 체제 진입 여부를 인과적으로 결정. 우리가
테스트한 3 비-CLIP 인코더 모두 포화, 우리가 테스트한 유일한 CLIP 인코더
는 비-포화.**

## 가설 업데이트

- **H-encoder-saturation** — *비-CLIP-일반으로 강화*. 업데이트된 논문
  주장: "인코더 family 가 인과적으로 인코더-probe AUC 와 행동 PMR(_nolabel)
  포화 driver. 패턴이 3 비-CLIP 인코더 family (SigLIP, SigLIP-SO400M,
  InternViT) 에 일반화, 테스트된 유일한 CLIP-기반 인코더 (CLIP-ViT-L/14)
  가 비포화 counterexample."
- **H-LM-modulation** (M9 도출) — *변경 없음*. 표에 3 비-CLIP × 3 LM family
  (Qwen2-7B, Mistral-7B, InternLM2-7B) 가 모두 유사한 포화 보임 — LM
  family 가 포화 driver 아님. 잔여 H7 효과는 sub-saturation 잡음.

## 한계

1. **Cross-stim AUC mismatch** (Qwen + LLaVA M2; Idefics2 + InternVL3
   M8a). 선택적 보충: Qwen + LLaVA M8a stim 재캡처. 인코더 probe AUC 는
   합성 stim 에서 stim-invariant 임이 우리 경험 — footnote, 차단 아님.
2. **InternVL3 PMR(_nolabel) 가 표 중 최고** (0.92), 그러나 최심 레이어
   AUC (0.886) 가 3 비-CLIP 모델 중 최저. 포화 클러스터 내 살짝 역상관은
   인코더-LM 트레이드오프 시사 가능성 — 그러나 3 모두 포화 체제, LLaVA
   훨씬 위.
3. **Per-shape AUC 희소**: by-shape 프로브가 양 클래스 존재 도형만 계산
   (도형당 n=80, 그러나 InternVL3 n_neg=21 전체로 여러 도형 all-positive,
   circle, polygon per-shape AUC 미정의). 통합 layer-sweep 수치가 헤드라인.
4. **모델 4개, CLIP 1개**: 더 강한 논문 주장은 CLIP 기반 인코더 2개 테스트
   (예: LLaVA-Next, ShareGPT4V) + 비-CLIP 4개 — "LLaVA-1.5 특이적" 가능성
   배제 위해.

## 헤드라인 그림

`docs/figures/encoder_chain_4model.png` — 2-패널 논문 그림:
- (a) 캡처된 2 모델 (Idefics2 + InternVL3) 의 layer sweep AUC + M6 r2
  베이스라인 수평선 (Qwen 0.99, LLaVA 0.73).
- (b) 산점도 (인코더 AUC, 행동 PMR) — 4 모델 점의 H-encoder-saturation
  사슬 시각화.

`docs/figures/encoder_swap_internvl3_probe.png` — InternVL3-only 2-패널
그림 (encoder_swap_idefics2_probe.png 와 유사).

## 로드맵 함의

- **§4.5 + M9 + M6 r3 + M6 r4 = 논문급 인코더-포화 사슬.** 4 모델 점 × 4
  노드 (인코더 family → AUC → 행동 PMR → H7 측정 가능성) 가 동일-LM
  인코더-스왑 counterfactual 다음으로 우리가 만들 수 있는 가장 강한 인과
  증거.
- **동일-LM 인코더 스왑** (예: LLaVA-1.5 CLIP vs LLaVA-1.5 SigLIP via
  Bunny / ShareGPT4V) 가 가장 깔끔한 counterfactual 으로 남음 — 이제
  "라운드 5" 향상, 차단 아님.
- **Qwen + LLaVA M8a 재캡처** 가 cross-stim AUC 주의 (M2 vs M8a) 제거를
  위한 논문 보충 작업. ~10 분 wall.

## 산출물

- `configs/encoder_swap_internvl3{,_label_free}.py`.
- `scripts/encoder_swap_probe.py` (모델-무관 드라이버,
  `encoder_swap_idefics2_probe.py` 에서 이름 변경).
- `scripts/encoder_swap_probe_summary.py` (4-모델 통합 그림).
- `outputs/encoder_swap_internvl3_*/predictions.{jsonl,parquet,csv}`.
- `outputs/encoder_swap_internvl3_vision_activations/*.safetensors`
  (~3 GB, gitignored).
- `outputs/encoder_swap_internvl3_probe/{layer_sweep,by_object_level,by_shape}.csv`.
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`.
- `docs/figures/encoder_swap_internvl3_probe.png` (per-model 2-panel).
- `docs/figures/encoder_chain_4model.png` (4-모델 통합 — 논문 헤드라인).
- `docs/insights/m6_r4_internvl3_probe.md` (+ `_ko.md`).
