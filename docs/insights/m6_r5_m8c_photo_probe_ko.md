# M6 r5 — M8c 사진 인코더 프로브 (4-모델, cross-stim)

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H-encoder-saturation** — 합성 stim 위 behavioral PMR(_nolabel) saturation 은 architecture 수준 (encoder + LM 결합) 에서 결정 — encoder 표현 능력만으로는 부족.
> - **M8a** — Stim 다양화 — 비-원 합성 shape (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim 다양화 — 실사진 (COCO + WikiArt 에서 60 photo × 5 카테고리). Qwen PMR(_nolabel) 을 18-48 pp 감소.
> - **M8d** — Stim 다양화 — 비-공 물리 객체 카테고리 (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M9** — Generalization audit — 논문 Table 1 (3 model × 3 stim 소스 × bootstrap CIs, 5000 iter); PASS/FAIL 이진화를 CI 분리로 대체.
> - **M6 r3** — Idefics2 SigLIP-SO400M probe — vision encoder AUC 0.93 으로 encoder-AUC ↔ PMR chain 마감 (3-point).
> - **M6 r4** — InternVL3 InternViT probe — AUC 0.89 / PMR 0.92, chain 을 4 model 점으로 확장; H-encoder-saturation 이 "non-CLIP-일반".
> - **M6 r5** — M8c 사진 encoder probe (4 model, cross-stim) — behavioral-y AUC 는 역전, stim-y AUC 는 1.0 유지 → encoder 식별력은 균일; architecture-수준으로 재구성.

**상태**: 2026-04-25 완료.

## 동기

M9 가 사진이 행동적으로 인코더 갭을 압축함을 발견: 비-CLIP 포화의
PMR(_nolabel) 이 0.84-0.92 (합성 M8a) 에서 0.28-0.55 (M8c 사진) 으로
하락, CLIP-LLaVA 는 두 stim 모두 저 PMR 유지. 인코더-측 대응은 미검증.
M6 r5 의 질문:

- 사진에서 인코더 probe AUC 도 압축되는가, 행동이 무너지는 동안 높게
  유지되는가?
- M6 r4 의 stim-y 재구성 (M8a 에서 인코더 식별 능력 균일) 이 사진에서도
  유지되는가, 깨지는가?

## 방법

`scripts/02_run_inference.py --config configs/encoder_swap_internvl3_m8c{,_label_free}.py`
로 InternVL3 의 M8c 행동 런 채움 (Qwen/LLaVA/Idefics2 §4.5 ext 에서 완료).

`scripts/04_capture_vision.py` 를 4 모델 모두 M8c 사진 stim 에서 실행
(60 사진 × 4 레이어). 모델당 wall clock: GPU 0 에서 ~10-40 초. 총 추론 +
캡처: ~5 분.

`scripts/encoder_swap_probe.py` (행동-y 모드, 3 라벨에 대한 stim 당 평균
PMR) + `scripts/encoder_swap_probe_stim_y.py --target physical_shape_vs_abstract_shape`
(stim-y 모드, ball/car/person/bird vs abstract).

## 결과

### M8c 사진의 행동-y probe AUC

| 모델       | 인코더          | LM            | M8c PMR(_nolabel) | M8a 행동-y AUC | M8c 행동-y AUC (평균) | M8c 행동-y AUC (최심) |
|------------|-----------------|---------------|------------------:|---------------:|---------------------:|--------------------:|
| Qwen2.5-VL | SigLIP          | Qwen2-7B      | **0.550**         | 0.880          | **0.582**            | 0.438               |
| LLaVA-1.5  | CLIP-ViT-L      | Vicuna-7B     | **0.283**         | 0.771          | **0.785**            | 0.856               |
| Idefics2   | SigLIP-SO400M   | Mistral-7B    | **0.417**         | 0.926          | **0.745**            | 0.771               |
| InternVL3  | InternViT       | InternLM2-7B  | **0.533**         | 0.886          | **0.661**            | 0.585               |

**행동-y AUC 패턴이 합성에서 사진으로 가면서 역전.** M8a 합성에서 비-CLIP
아키텍처는 CLIP-LLaVA (0.77) 보다 높은 행동-y AUC (Qwen 0.88, Idefics2
0.93, InternVL3 0.89). M8c 사진에서: **LLaVA 가 행동-y AUC 최고치 (0.86)**,
Qwen 은 **0.44** 로 하락, Idefics2 **0.77**, InternVL3 **0.59**.

### M8c 사진의 stim-y probe AUC (physical_shape_vs_abstract_shape)

| 모델       | Stim-y AUC (레이어 평균) | Stim-y AUC (최심 레이어) |
|------------|-----------------------:|----------------------:|
| Qwen2.5-VL | **1.000**              | 1.000                 |
| LLaVA-1.5  | **0.988**              | 1.000                 |
| Idefics2   | **0.992**              | 1.000                 |
| InternVL3  | **0.996**              | 1.000                 |

**4 인코더 모두 사진을 물리-도형 (ball / car / person / bird) vs 추상 사진
으로 AUC ≈ 1.0 으로 선형 분리** — M8a 와 동일한 균일-인코더-식별 능력
발견 (3 stim-y 타겟 모두 AUC = 1.0).

## 해석

두 AUC 뷰가 깔끔한 결합 그림:

1. **인코더 표상 능력은 stim-불변 + family-불변.** M8a 합성과 M8c 사진
   모두에서, 우리가 테스트한 모든 인코더 (SigLIP / CLIP-ViT-L /
   SigLIP-SO400M / InternViT) 는 stim-정의 y 로 물리 vs 추상 stim 카테고리
   를 AUC ~ 1.0 으로 선형 분리.
2. **행동 PMR(_nolabel) 은 아키텍처 driver + stim-조건적.** 비-CLIP 아키
   텍처는 합성에서 포화 (0.84-0.92), 사진에서 무너짐 (0.42-0.55). CLIP-LLaVA
   는 합성에서 저 (0.18) + 사진에서 적당히 높음 (0.28). 아키텍처 횡단 비교는
   stim 출처 내에서만 의미.
3. **행동-y AUC 가 cross-stim 역전하는 이유는 그것이 "인코더 ↔ 행동 정렬"
   의 측정이지, 인코더 식별 능력이 아니기 때문.**
   - M8a 에서 비-CLIP 인코더 표상이 자체 포화 행동 PMR 패턴과 강하게 공변
     → 행동-y AUC 높음.
   - M8c 에서 비-CLIP 행동 PMR 이 더 변동, 인코더 표상은 동등하게 정보적
     → 행동-y AUC 낮음.
   - LLaVA 의 CLIP 인코더는 두 체제 모두에서 LLaVA 행동 PMR 과 동등하게
     공변 (행동이 두 곳 모두 변동, CLIP 이 사진-훈련 → 사진-측 정렬이
     자연스러움).

이는 cross-stim 수준에서 M6 r4 재구성을 잠금: H-encoder-saturation 사슬은
**인코더-LM 융합** 수준 (LM 이 인코더 출력을 physics-mode 신호로 어떻게
소비하는가), 인코더 식별 능력 수준 아님.

## 가설 업데이트

- **H-encoder-saturation** — *M6 r4 재구성의 cross-stim 확인*. 합성 stim
  의 패턴 (인코더-LM 융합이 minimal-context stim 의 행동 포화 driver) 이
  사진에서도 유지: 인코더 식별 능력은 AUC ≈ 1.0 유지, 행동-y AUC 는 각
  stim 의 per-모델 PMR 분포에 따라 재조직. 논문 주장이 더 날카로워짐: 인코
  더는 physics-relevant 정보를 균일하게 보유; 행동 physics-mode 읽기는
  LM-측 융합에 의해 결정, 그 융합의 강도는 stim 유형에 따라 변동.
- **행동 PMR cross-stim 스토리** (통합): 비-CLIP 아키텍처가 "synth-stim
  minimality 포화" 보임 (M8c 발견), 인코더 기여 AND stim 단순성 둘 다
  포화 체제에 필요. 사진은 stim-단순성 인자 제거; 행동이 균일하게 무너짐.

## 한계

1. **모델당 n=60 사진** 은 행동-y probe 에 작음 (n_pos ≈ 20, n_neg ≈ 40).
   AUC 분산 큼; 사진 행동-y 의 모델 간 차이는 과도 해석 금지.
2. **사진은 COCO + WikiArt 의 특정 샘플** — 다른 사진 분포 (자연 장면, 웹
   이미지 등) 에 일반화는 미해결.
3. **Per-shape / per-category 분해는 사진에서 미실행** — 카테고리당 12
   사진은 sub-probe AUC 에 너무 작음.

## 헤드라인 그림

(이번 라운드 신규 그림 없음; 수치는 encoder_chain_4model.png 의 caveat 으로
들어감: 사진-측 AUC 가 M8a-측 AUC 와 인코더-LM 융합 재구성을 지지하는
방식으로 다름).

## 로드맵 함의

- **§4.5 + M9 + M6 r3 + M6 r4 + M6 r5 = 인코더-포화 스토리 완성.** 4 모델
  점 × 2 stim 유형 × 2 y 모드 = 16 AUC 셀. 깔끔한 논문급 narrative.
- **선택적 다음**: 동일-LM 인코더 스왑 (Bunny SigLIP+Phi-2 if 채팅 템플릿
  작동) 으로 LM-제어 counterfactual. 라운드 6 후보.
- **선택적 다음**: M8d 사진 probe (합성 비-볼 카테고리는 아직 M8c-style 실
  사진 등가물 없음). 낮은 우선순위.

## 산출물

- `configs/encoder_swap_internvl3_m8c{,_label_free}.py`.
- `outputs/encoder_swap_internvl3_m8c_*/predictions.{jsonl,parquet,csv}`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_vision_activations/*.safetensors`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_probe/{layer_sweep,by_object_level,by_shape}.csv`.
- `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_probe_stim_y/layer_sweep_stim_y_physical_shape_vs_abstract_shape.csv`.
- `docs/insights/m6_r5_m8c_photo_probe.md` (+ `_ko.md`).
