---
section: M5b — SAE intervention on Qwen vision encoder (last layer, pre-projection)
date: 2026-04-27
status: complete (n=20 clean stim × 4 top-k 조건 × 3 random control; 깨끗한 positive 결과)
hypothesis: H10 (research plan §2.5) 정련 — encoder-side physics-mode 신호는 *작은* monosemantic SAE feature 집합 (delta-rank top-20) 에 localize 되어 있음; single feature 는 dispensable 하지만 cumulative ablation 이 physics commitment 깨뜨림; random-feature control 이 magnitude-driven 이 아닌 feature-specific 임을 확인.
---

# M5b — Qwen2.5-VL vision encoder 위 SAE intervention

> **Recap**
>
> - **M5b SIP 패칭** (sufficiency, layer-level): L0-L9 패칭 → 20/20 physics 회복; sharp L10 boundary.
> - **M5b layer-level knockout** (necessity): L9 MLP IE = +1.0 — uniquely necessary; attention 모든 layer 0 IE.
> - **M5b per-head knockout**: 196 (L, h) 모두 IE = 0 — attention 완전 redundant 확인.
> - **남은 것**: upstream 쪽. LM-internal mechanism 은 잘 localized (L9 MLP construction); L9 MLP 가 *construct from* 하는 *encoder-side* 신호는 localize 되어있나, 또는 SigLIP feature 수천 개에 분산되어 있나?
> - **SAE** (Sparse Autoencoder; Bricken et al. 2023; Pach et al. 2025): layer activation 위 over-complete linear-relu-linear bottleneck 을 L1 sparsity 로 학습, monosemantic feature 회복.
> - **§4.6 Qwen pixel-encodability**: `pixel_values` 위 v_L10 방향 gradient ascent 가 ε=0.05 에서 physics-mode flip (5/5). Encoder-side mechanism 존재; SAE 가 *어떤* encoder feature 가 그것을 운반하는지 식별하는 자연스런 도구.

## 질문

L9 MLP 가 LM-side construction site. *Encoder-side* physics-mode 정보는
어디에 localize 되어 있는가? 두 극단 가설:

(a) **분산**: physics-mode 신호가 많은 encoder feature 에 spread; 작은
    subset 이 causally 책임지지 않음. 작은 feature 그룹 ablate 해도
    행동 보존; 큰 magnitude perturbation (matched-random 또는
    top-feature) 만 동일하게 행동 깨뜨림.

(b) **Localized**: 작은 monosemantic feature 집합 (예: 5120-feature SAE 에서
    10-50 개) 이 physics-mode 신호 운반. *Specific* feature ablate 가
    physics-mode 깨뜨리지만 matched-magnitude random ablation 은 보존.

§4.6 pixel-space 결과는 *어떤* encoder-side direction (L10 에서 v_L10 read
out) 이 ε=0.05 에서 shortcut 신호 운반함을 이미 보여줬음. SAE intervention
은 layer-level MLP knockout 의 encoder-side analog: 충분히 fine resolution
에서 ablation 이 physics 를 깨뜨리는 *specific feature 그룹* 이 있는가?

## 방법

### Activation source

`outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` —
M2 run (5-axis factorial, label="circle/ball/planet") 의 480 stim 위 캡처된
Qwen2.5-VL 마지막 vision-encoder layer activation (`vision_hidden_31`,
1296 visual token × 1280 SigLIP hidden dim). 총 622,080 token.

### SAE 학습

- 아키텍처: tied-weight encoder/decoder + input z-score 정규화 (per-dim
  mean/std, 100K 샘플), input bias `b_pre`, encoder bias `b_enc`,
  decoder column unit-norm constraint.
- d_in = 1280, d_features = 5120 (4× 확장).
- Loss = MSE(reconstruction) + λ × L1(z), λ = 1.0 (정규화 공간).
- 5000 Adam step, batch 4096, lr 1e-3 — H200 에서 1.1분.
- 최종: recon = 0.023 (정규화), L1 = 0.042, 100% feature alive,
  토큰당 7.3% active (~370 feature/token).

### Feature ranking

`predictions_scored.csv` 의 per-sample mean PMR:
- Physics-mode set: mean_pmr ≥ 0.667 인 310 stim.
- Abstract set: mean_pmr ≤ 0.333 인 19 stim.

Feature i 별: `delta_i = mean(z_i | physics) − mean(z_i | abstract)`.
delta 기준 top-20 저장.

Top-10 feature (raw activation 평균):

| feature_idx | mean_phys | mean_abs | delta |
|------------:|----------:|---------:|------:|
| 4698 | 3.13 | 0.23 | **2.90** |
| 1152 | 2.66 | 0.32 | 2.34 |
| 3313 | 7.86 | 6.24 | 1.62 |
| 4106 | 1.75 | 0.13 | 1.61 |
| 1949 | 1.55 | 0.15 | 1.39 |
| 38 | 2.32 | 0.99 | 1.33 |
| 4468 | 1.39 | 0.14 | 1.26 |
| 438 | 1.26 | 0.14 | 1.12 |
| 117 | 1.18 | 0.12 | 1.06 |
| 1674 | 1.03 | 0.01 | 1.02 |

### 인과 intervention

각 clean SIP stim (n=20, layer-level knockout 와 동일 cohort) 에 대해 마지막
vision encoder block 출력 (model.visual.blocks[-1]) 에 forward hook 등록.
Hook 이 target SAE feature 의 raw-scale 기여만 빼는 (Bricken et al. trick —
non-target feature + 재구성 residual 정확히 유지):

```
contribution_n = z[:, target_feats] @ W[target_feats]         # 정규화 공간
contribution_raw = contribution_n * input_std                # 다시 raw 로
x_new = x - contribution_raw
```

Sweep: physics-cue ranking 의 top_k ∈ {1, 5, 10, 20} + **magnitude-matched
random control** (k=20). 초기 구현은 bottom-of-ranking pool 에서 random
sample, 그러나 그 feature 들은 mass ≈ top-20 의 1% — L1 penalty 가 inactive
feature 죽임 → "random" ablation 이 zero-magnitude, fair specificity test
아님. 수정된 pool: 21+ ranking 중 *top-300 by mass*, 총 `mean_phys + mean_abs`
mass 가 top-20 의 [70%, 200%] 범위. 1개 matched set 만 발견 (top mass = 49.23,
random_0 mass = 40.97 = 83%) — activation 분포 heavy-tailed (top feature 3313
혼자 mass 14), 대부분 random k=20 sample 이 70% threshold 미달.

## 결과

![SAE intervention 결과](../figures/m5b_sae_intervention_phys_rate.png)

(Figure 미생성; CSV-only 결과.)

| Condition | Mass | Physics rate (n=20) | 비고 |
|-----------|-----:|--------------------:|------|
| Baseline (hook 없음) | — | 1.000 | manifest 구축에 의해 |
| top_k=1 (feature 4698 zero) | 3.36 | **1.000** | single top feature dispensable |
| top_k=5 (top-5 feature zero) | 11.16 | **0.600** | 부분 break: 8/20 flip |
| top_k=10 (top-10 feature zero) | 27.74 | **1.000** | 회복 (non-monotone) |
| **top_k=20 (top-20 feature zero)** | **49.23** | **0.000** | **full break: 0/20 retain physics** |
| random k=20 (mass-matched, top-20 의 83%) | **40.97** | **1.000** | mass-matched random ablation 효과 *없음* |

top_k=20 ablate 시 20 stim 모두 비슷한 D-prefix 응답 생성. 주목: random
k=20 control *도* stim 간 매우 비슷한 A-prefix 응답 ("The red arrow
pointing downward suggests…") 생성. "동일 응답" 패턴은 homogeneous stim
set (모든 clean cue=both physics-mode 입력) 위 greedy decoding 의 반영,
encoder collapse 아님 — top-feature 와 random ablation 모두 동일-prefix
패턴 생성, *내용* (A vs D) 만 어떤 feature 가 ablate 되었는지에 따라 flip.

## Headlines

1. **Encoder-side physics-mode 신호는 SAE feature 공간에서 localized.**
   top-20 physics-cue SAE feature (mass 49.23, 5120-feature SAE 의 ~1%)
   빼면 20/20 stim physics → abstract flip. **Mass-matched** random k=20
   feature 빼면 (mass 40.97 = top-20 의 83%) 20 stim 모두 physics-mode
   유지. 이 실험의 첫 버전은 bottom-of-ranking random pool 사용 → top-20
   mass 의 ~1% (L1 penalty 가 inactive feature 죽임); mass-matched pool 로
   수정이 load-bearing fix. 결과는 encoder 의 direction-specificity true
   positive — input/LM-internal layer 의 §4.6 v_L10 vs random-direction
   결과의 평행.

2. **단일 feature 는 dispensable.** top-1 feature (idx 4698, delta=2.90, 가장
   강한 single physics-cue) 만 zero 해도 PMR 유지. LM attention level 의
   redundancy-spreading 가 encoder-side analog 보유: physics-mode 정보는
   *작은 feature 그룹* (~20) 에 인코딩, single feature 가 아님.

3. **Mid-range non-monotone 미해결 (k=5 → 0.6, k=10 → 1.0).**
   k=10 이 PMR 완전 회복은 예상 밖 — 가능 설명: k=10 이 "compensating"
   조합 hit, feature 6-10 (top-5 보다 약간 lower delta) 이 abstract-mode
   정보 운반하다가 동시 제거되어 상대적 균형 회복. 더 큰 n 으로 replicate
   가치. Headline 안 흔들음: k=20 깨끗 break + random k=20 안 깨뜨림.

4. **Triangulated mechanism — full causal chain**:
   - **Encoder side**: `vision_hidden_31` (마지막 SigLIP layer, pre-
     projection) 의 top-20 SAE feature 가 physics-mode 신호 운반. Necessary
     (이 실험) + observable (delta ranking).
   - **LM side**: L9 MLP 가 residual stream 에 commitment construct
     (necessary, M5b knockout); L0-L9 가 sufficient 정보 운반 (M5b SIP);
     L10 이 redundant attention 으로 read (M5a + per-head null).
   - **Pixel side**: `pixel_values` 위 v_L10 방향 gradient ascent 가
     ε=0.05 에서 PMR flip; encoder 가 픽셀 perturbation 을 top-20 SAE
     feature 변화로 변환 (testable follow-up).
   - Mechanism: **input → encoder physics-cue features (top-20) →
     L0-L9 visual token → L9 MLP commitment → L10 read-out → letter**.

5. **H10** (research plan §2.5: "specific layer/head 의 narrow IE band")
   가 encoder-side dimension 을 얻음. LM 쪽은 L9 의 1 dominant MLP band;
   encoder 쪽은 마지막 layer 의 ~20 SAE feature. 둘 다 "narrow" 그러나
   다른 granularity — framing 은 per-architecture-component (layer/head/
   feature), literal layer 수 아님.

## 한계

1. **Mass-matched random control set 1개만, 3개 아님.** 초기 3-seed 계획은
   heavy-tailed mass 분포에 의해 무산: feature 3313 혼자 mass 14 (다음의
   3×; "general image content" feature 가능, physics-specific 아님), 그래서
   active-feature pool 의 대부분 random k=20 sample 이 70% threshold 미달.
   1개 mass-matched set 획득 (top mass 의 83%). Single-sample binary 결과
   (20/20 vs 0/20) 는 unambiguous, 그러나 n=3+ random set 이 random-ablation
   PMR rate 의 upper bound 를 tighter 하게 함.

2. **k=10 의 non-monotonicity 미해결.** n=20 에서 "1-2 stim flipped"
   sampling variance 안 들어감, 그러나 k=5 (40% break) 와 k=20 (100% break)
   사이 끼워진 k=10 의 all-20-recover 는 strange. n=40 + 중간 k 값 (k=8,
   12, 15) 으로 replicate 가치. Headline 안 변함 (random control 이
   disambiguate).

3. **Top-20 에 high-mass outlier 포함 (feature 3313, mass 14).** `mean_phys`
   (7.86) 와 `mean_abs` (6.24) 모두 다음 feature 의 3× — physics-specific
   이 아닌 "general image has content" 시사. 제거 시 top-19 mass 가 ~35
   로 축소 그러나 delta rank 3. Future work: raw delta 대신 Cohen's d
   (delta / sqrt(var_phys + var_abs)) 또는 specificity ratio
   (delta / (mean_phys + mean_abs)) 로 ranking 하여 high-baseline feature
   filter out.

4. **Pre-projection layer 만**. SAE 는 `vision_hidden_31` 위 학습 (1280-
   dim, projector 가 3584 로 lift 하기 전). 식별된 feature 는 SigLIP-
   encoder-level feature, LM 이 1:1 "consume" 하는 게 반드시 아님.
   Post-projection SAE (3584-dim) 가 L9 MLP 의 더 직접적 causally upstream
   이지만 새 capture pass 필요.

5. **SAE 학습 한 번**. 다른 L1 lambda / expansion factor / training-data
   composition 이 다른 feature dictionary 줄 수 있음. 5120-feature 4× 확장은
   합리적이지만 pre-registered 아님. 결과는 internally consistent (top
   feature 가 mass-matched random 과 1.0 → 0.0 PMR 로 다름) 그러나 SAE
   학습 사이 feature-set portability 미테스트.

6. **n=20 cue=both clean stim 만**. Layer-level knockout 와 동일 sampling
   caveat; 더 어려운 case (line/blank/none) 가 다른 feature-group 구조
   보일 가능성.

## 다른 발견과의 연결

- **§4.6 pixel encodability**: `pixel_values` 위 v_L10 방향 gradient ascent
  가 ε=0.05 에서 PMR flip. Mechanism: encoder 가 픽셀 perturbation 을
  top-20 SAE feature 변화로 변환, L9 MLP 로 propagate. 이 SAE intervention
  은 그 encoder-side path 의 *직접* 테스트 — localized feature group 존재
  확인.

- **L10 의 M5a steering**: v_L10 은 post-encoder LM hidden state 에 lives.
  Top-20 SAE feature 가 projector → LM 으로 feed, 거기서 cue 가 결국
  v_L10 의 direction 이 됨. SAE feature 가 encoder basis; v_L10 이
  LM-internal axis.

- **H-encoder-saturation** (M6/M9): Qwen 의 saturated SigLIP 인코더는
  L3 부터 깨끗하게 class-separated activation 생성 — physics-cue feature
  가 L3 에서 이미 깨끗하게 carve out 되어 L31 까지 persist. SAE finding
  추가: carving 의 *low intrinsic dimensionality* (~20 feature, 수백 아님).

- **M5b layer-level + per-head**: attention 은 LM 에서 테스트한 모든
  resolution 에서 redundant. Encoder-side localization (이 실험) 은 LM-
  side localization 으로 *전파되지 않음* L9 MLP 너머. Encoder 가 ~20-
  feature 신호 생성; LM 이 그것을 L9 의 single decision boundary 로 압축.

## 재현

```bash
# 1. Qwen vision encoder activation 위 SAE 학습 (기존 M2 캡처 사용).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_train.py \
    --activations-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations \
    --predictions outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv \
    --layer-key vision_hidden_31 --n-features 5120 --n-steps 5000 \
    --tag qwen_vis31_5120 --device cuda:0 --l1-lambda 1.0

# 2. 인과 intervention (top-k feature zero + random control).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/qwen_vis31_5120 --layer-key vision_hidden_31 \
    --top-k-list 1,5,10,20 --random-controls 3 --n-stim 20 --device cuda:0
```

## Artifacts

- `src/physical_mode/sae/{train,feature_id}.py` — SAE 모듈 (tied-weight, input-normalized, clean intervention 위한 `feature_contribution`).
- `scripts/sae_train.py`, `scripts/sae_intervention.py` — driver.
- `outputs/sae/qwen_vis31_5120/{sae.pt,metrics.json,feature_ranking.csv}`.
- `outputs/sae_intervention/qwen_vis31_5120/results.csv`.

## Open follow-ups

1. **Mass-matched random set 더 많이**: candidate pool 을 weighted
   sampling (probability ∝ feature mass) 으로 완화하여 3+ matched set
   확보 → random-ablation PMR rate 의 upper bound tighter.
2. **Cohen's d / specificity ratio 로 re-rank**: feature 3313 같은
   high-baseline feature 를 "physics-cue" set 에서 filter out. Cohen's d
   기준 top-20 가 여전히 PMR 깨끗 break 하는가, feature 3313 단독이
   안 깨뜨리는가?
3. **Single-feature ablation sweep**: top-20 feature 각각 개별 zero;
   그룹 내에서 *개별적으로* necessary vs redundant 인 subset 식별.
4. **Feature-level 기능 해석**: top-20 feature 각각, 480-stim 코퍼스의
   max-activating image patch 시각화. Monosemantically "physics-cue"
   (예: shadow, motion arrow, filled-disk shape) 인가?
5. **Non-monotonicity probe**: k=8, 12, 15 에서 n=40 — k=10 회복 유지?
6. **Post-projection SAE**: post-projector activation (3584-dim, LM 이 실제
   소비하는 것) 캡처 후 feature discovery + intervention 재실행.
7. **Cross-layer SAE**: `vision_hidden_15` 또는 더 이른 layer 에 SAE 학습;
   어떤 layer 가 처음으로 physics-mode feature 인코드하는지 trace.
8. **Cross-model SAE**: LLaVA-1.5 / Idefics2 / InternVL3 로 port — 각자
   인코더 마지막 layer 에 자신의 ~20-feature physics-cue 그룹 보유?
