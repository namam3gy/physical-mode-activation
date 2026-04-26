# M4 (ST3) Insights — LM Logit Lens + Per-Layer Probes

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택 (ball → 동적 / circle → 정적 / planet → 궤도).
> - **H-boomerang** — Vision encoder 가 행동이 실패하는 곳에서도 physics-mode class 를 선형 분리 — encoder 는 알고 decoder 가 gate. (Qwen 한정: LLaVA-1.5 에서는 CLIP encoder 자체가 bottleneck 이라 반박.)
> - **H-locus** — Bottleneck 은 LM 중간 레이어 (특히 L10) 에 있음 — 더 이른 곳도, decoding head 도 아님.
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M3** — ST2 vision-encoder probing — encoder AUC ≈ 1.0 으로 factorial 축 자명 분리 ("boomerang").
> - **M4** — ST3 LM logit lens / per-layer probes — LM AUC 가 시각-토큰 위치에서 L5 부터 ~0.95 plateau.
> - **M5** — ST4 인과 localization (VTI steering / activation patching / SAE) — M5a, M5a-ext, M5b 참조.


**Probe 대상 자극** — M2 480-stim factorial. Canonical line/blank/none baseline (LM 이 기본 "abstract" 로 응답하는 low-PMR 셀):

![M4 참조 자극: line / blank / none](../figures/01_line_blank_none.png)

Sub-task 3 는 M3 boomerang 의 **내부 연장**: vision encoder 가 정보를
완벽하게 전달한다면 (M3), LM 내부에서 그 정보는 **어느 레이어까지**
살아남고, **어디서** 디코딩으로 가는 정보가 누수되는가?

원본 수치: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.csv` ·
`*.parquet`. 구현: `src/physical_mode/probing/lm.py`, driver:
`scripts/05_lm_probing.py`.

## 1. 한 문장 요약

**Qwen2.5-VL-7B 의 LM 은 visual-token 위치의 hidden state 에서 physics-mode
PMR 을 모든 5개 captured layer (5, 10, 15, 20, 25) 에서 AUC ≈ 0.94-0.95 로
선형 예측 가능하다.** 정보는 LM 을 거의 손실 없이 통과한다 — 게이팅은
**토큰 생성(디코딩) 단계** 에서 일어난다.

## 2. 무엇을 측정했나

### 2.1 Setup

- 입력: M2 가 저장한 LM hidden states at `layers=(5, 10, 15, 20, 25)` of
  Qwen2.5-VL-7B-Instruct 의 28-layer language model. 각 stimulus 당 324
  visual token × 3584 dim, bf16.
- Probe: `sklearn.LogisticRegression`, 5-fold stratified CV,
  mean-pool across visual tokens, StandardScaler.
- Logit lens: `lm_head` (unembedding projection) 을 각 layer 의 hidden state
  에 적용 → 특정 token id 집합의 logit 추적. Mean-pool across visual tokens.

### 2.2 Token 집합

- **physics verbs** (15 tokens): `fall`, `falls`, `falling`, `drop`, `drops`,
  `roll`, `rolls`, `rolling`, `bounce`, `slide`, `land`, `tumble`, `move`,
  `moving`, `orbit`.
- **geometry / static** (10 tokens): `circle`, `shape`, `line`, `drawing`,
  `image`, `figure`, `abstract`, `geometric`, `still`, `static`.
- **label** (3 tokens): `ball`, `planet`, `object`.

Tokenizer 가 single sub-token 으로 매핑할 수 있는 것만 유지.

## 3. 핵심 수치

### 3.1 Per-layer PMR probe (forced-choice target)

| layer | AUC (mean ± std) | accuracy |
|---|---|---|
| 5 | 0.939 ± 0.015 | 0.867 |
| 10 | 0.944 ± 0.006 | 0.881 |
| 15 | 0.947 ± 0.009 | 0.885 |
| **20** | **0.953 ± 0.007** | 0.885 |
| 25 | 0.944 ± 0.009 | 0.877 |

→ **모든 레이어에서 PMR 이 선형 분리됨**. Peak 은 layer 20 (0.953).
Neo et al. 2024 가 LLaVA-1.5 에서 object-specific features crystallize
layers 15-24 로 보고한 패턴과 일치 — Qwen2.5-VL-7B (28 LM layers) 에서도
중-후기 레이어에 peak.

비교:
- Vision encoder AUC @ L31: **0.944**
- LM hidden @ L20: **0.953**
- LM hidden @ L25: 0.944

→ **Vision encoder 와 LM hidden 이 거의 동일한 discriminability**.
정보가 LM 을 통과하면서 거의 손실되지 않는다.

### 3.2 Logit lens — "ball" 라벨이 L5 에서 이미 LM 을 prime

모든 480 stimulus 에 대한 평균 logit:

| layer | geometry | physics | label |
|---|---|---|---|
| 5 | 0.93 | 1.04 | 1.16 |
| 10 | 1.35 | 1.66 | 1.73 |
| 15 | 2.04 | 2.45 | 2.29 |
| 20 | 3.23 | 4.18 | 4.09 |
| 25 | 11.56 | **15.64** | 13.96 |

**L5 에서 이미 physics > geometry (1.04 > 0.93)**. 이는 "ball" 라벨이
프롬프트에 있어서 LM 이 출발부터 physics-mode 로 prime 되었기 때문.

Layer 25 에서 physics logit 이 15.64 vs geometry 11.56 → **physics
dominance 가 최종 레이어에서 극적으로 amplify** (gap 4.0). 이는 LM 의
마지막 residual updates 가 "이건 떨어지는 공" 같은 physics 서술을 강화
한다는 해석.

### 3.3 Object_level 별 physics margin

Physics margin = mean(physics logit) − mean(geometry logit):

| layer | filled | line | shaded | textured |
|---|---|---|---|---|
| 5 | 0.08 | 0.09 | 0.12 | 0.15 |
| 10 | 0.29 | 0.27 | 0.33 | 0.38 |
| 15 | 0.38 | 0.35 | 0.44 | 0.49 |
| 20 | 0.89 | 0.87 | 0.97 | 1.05 |
| 25 | 3.94 | 3.76 | 4.29 | 4.35 |

- **L5 부터 추상 → 구체 가 monotone 순서** (line/filled 가 shaded/textured 보다 낮음, L5 ~0.01 차).
- **L25 에서 최대 gap** (textured 4.35 vs line 3.76 = +0.59).
- 그러나 이 object-induced gap (0.6) 은 label-induced shift (전체를 4 unit 위로 올리는) 보다 **훨씬 작다** → **라벨이 시각 증거보다 7 배 큰 효과**.

### 3.4 Switching layer (한계)

`switching_layer = 가장 이른 레이어 where max(physics token logits) ≥ max(geometry token logits)`
기준으로 모든 480 샘플이 **layer 5 에서 switching** — 즉 captured layers 의
최저 layer 부터 이미 physics 가 우세. 이 메트릭은 현재 setup 에서
informativ하지 않다 — "ball" 라벨이 프롬프트에 있는 한 LM 은 출발부터
physics-mode prior 를 가진다.

**더 informative 하게 측정하려면**:

- captured layers 를 더 조밀하게 (0, 1, 2, 3, 4, 5...) — L5 이전에 geometry
  dominance 구간이 있을 수 있음.
- label-free prompt ("What do you see?") 로 재실행 — label prior 제거 후
  switching 위치 관측.

이 두 개는 M4 후속 (혹은 §4.9 "label 없이 프롬프트") 아이디어로 이관.

## 4. 통합 해석 — M3 + M4

**세 단계 파이프라인의 가역성**:

| 단계 | PMR discriminability |
|---|---|
| Vision encoder (M3, layer 31) | AUC **1.00** on stimulus truth, **0.944** on behavioral PMR |
| LM hidden state @ visual tokens (M4, layer 20) | AUC **0.953** on behavioral PMR |
| Behavioral output (actual decoding) | forced-choice accuracy ≈ 0.66 |

**정보는 vision encoder (0.94-1.0) → LM hidden state (0.95) 에서 거의
완벽히 보존**된다. 디스크리트 generation 에서 약 28-29 pp 의 정확도 손실이
발생 (AUC 0.95 에서 이론적으로 얻을 수 있는 correct-prediction rate 대비
실제 forced-choice 정확도).

"Boomerang" 은 이제 정확한 위치를 가진다:

```
vision encoder ───(정보 존재)──→ LM early layers ───(정보 존재)──→ LM late layers ───(정보 존재)──→ decoding ──(부분 손실)──→ token output
```

**병목은 decoding 단계 그 자체**, 혹은 더 정확히는 "LM hidden → logit 분포
→ token sampling" 의 어느 지점. 이는 activation patching (ST4) 의 직접
타겟을 제시한다: clean/corrupted 쌍을 **마지막 residual update** 또는
**logit bias** 에서 patching 하면 decoding 결정을 뒤집을 수 있을 것.

## 5. 가설 스코어카드 업데이트 (M4 이후)

| H | 이전 상태 | M4 이후 | 변화 |
|---|---|---|---|
| H-boomerang | 지지 (M3) | **확장** | 정보가 LM 을 관통해 살아남음. 게이팅이 decoding 에서 발생. |
| H7 (라벨 = physics regime) | 후보 (M2) | **지지** | Logit lens 에서 label prior 가 L5 부터 physics-geom margin 을 shift. Object_level effect 의 7배 크기. |
| H-locus (신규) | — | **후보** | 병목은 LM final layers + decoding 에 있다. ST4 에서 activation patching 으로 확정 가능. |

## 6. 논문 figure 로 사용할 값있는 plot

### Figure 4 (M4): LM 을 통과하는 정보 궤적

- X축: layer (5, 10, 15, 20, 25)
- Y축 왼쪽: probe AUC (forced-choice) — 모든 레이어 ~0.94
- Y축 오른쪽: physics margin (phys − geom logit) — 0.1 → 4.0 amplification
- Object_level 별 line plot

메시지: "정보는 LM 을 통과해 살아남는다 (AUC 평탄), 그런데 final layer
에서 physics 서술이 amplify (margin 급증). 그러나 decoding 은 그 amplification 을
완전히 사용하지 않는다 (behavior < AUC 가 보여주는 상한)."

### Figure 5 (M4 + M5 predicted): boomerang localized

Layer-wise probe AUC (vision encoder + LM) vs behavioral PMR, horizontal bars
or cumulative plot. "어디서 정보가 버려지는가" 의 visual answer.

## 7. M5 에 넘겨주는 것

- **ST4 (activation patching) 의 직접 타겟**: LM final layers (20-28) +
  decoding head. 이유: probe AUC 가 L20 에서 peak 인데 behavioral accuracy
  는 0.66 → "LM 이 정보를 갖고 있는데 output 에 반영 안 함". Patching
  개입이 가장 유효할 것.
- **실행 필요**: `capture_lm_attentions=True` 로 재-capture (M2 는 attention
  저장 안 함). Mini-factorial 로 SIP 쌍 만들어 patching 돌리면 Sub-task 4
  완성. Disk 비용: ~15 GB.

## 8. 한계

1. **Switching layer 메트릭은 label-primed 프롬프트에서 무력화됨**.
   Label-free ("What do you see?") 대체 프롬프트로 재실행 필요 → §4.9
   아이디어와 자연스럽게 연결.
2. **5개 레이어 snapshot 은 coarse**. 특히 L0-L5, L25-L27 구간이 비어있음.
   후속 run 에서 dense capture (매 레이어) 필요할 수 있음.
3. **Mean-pool 은 정보를 평균내서 단순화**. 개별 visual token 위치별 probe
   (Neo et al. 2024 heatmap 스타일) 는 아직 안 했음.
4. **Logit lens 는 `lm_head` 의 linear projection 만 사용** — 실제 LM 이
   tokens 을 생성할 때는 softmax + sampling 이 추가된다. 그 마지막 몇
   단계의 stochasticity 가 AUC 0.95 대비 행동 accuracy 0.66 gap 의 원인일
   수 있음 (T=0.7 setting).
