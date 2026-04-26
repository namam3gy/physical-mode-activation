---
section: M5b — SIP + LM activation patching (Qwen2.5-VL)
date: 2026-04-26
status: complete (n=20 SIP pairs × 28 LM layers)
hypothesis: H10 (research plan §2.5) — "2-3 narrow layer/head 범위에서 큰 IE" → SUPPORTED with refinement: L0-L9 (early-mid 블록) full recovery; L10-L13 declining; L14+ zero
---

# M5b — Semantic Image Pairs + activation patching (Qwen2.5-VL)

> **이 문서에서 쓰는 코드 한 줄 recap**
>
> - **M2** — 480-stim 5-axis factorial; FC 프롬프트가 letter-first 응답 유도.
> - **M5a** — VTI steering: LM L10 시각 토큰에 +α·v_L10 더하면 line/blank/none 이 "정지" → physics-mode 로 flip. α=40 으로 L10 만 causal-intervention layer.
> - **M5b** — ST4 Phase 3 (이 문서): activation patching 으로 인과적으로 필요한 layer 식별.
> - **SIP** (Semantic Image Pairs, Golovanevsky et al., NAACL 2025) — 단일 시각 cue 만 다른 minimal-pair stim.
> - **IE** (Indirect Effect) — 패치로 인한 ΔP(physics-mode): P(physics | patched corrupted) − P(physics | corrupted baseline).
> - **H-locus** (M4-derived) — bottleneck 이 LM 중간 레이어 (L10) 에 있음.
> - **H10** (research plan §2.5) — "2-3 narrow layer/head 범위에서 큰 IE."

## 질문

M5a 의 runtime VTI steering 은 **L10 만이** Qwen2.5-VL 에서 α=40
intervention 으로 행동을 뒤집는 layer 임을 보였다. 자연스러운 후속:
L10 이 *유일한* causal layer 인가, 아니면 decision-lock-in 의 *경계*
이고 upstream layer 들도 cue 정보를 담는가? SIP 활성화 패칭 — physics
cue 존재 유무만 다른 — 이 두 해석을 구분.

## 방법

**SIP 구축** (`build_sip_manifest`):
- M2 forced-choice 예측 (Qwen2.5-VL, 1440 추론).
- 각 (object_level, bg_level, event_template) 에 대해 cue=both seed
  (clean: physics-mode 응답, abs_rate=0) 와 cue=none seed (corrupted:
  abstract 응답, abs_rate=1) 를 인덱스로 페어링.
- 필터: `clean abs_rate=0 AND corr abs_rate=1` (엄격하게).
- 32 candidate 페어 산출; 처음 n=20 사용.

**패칭** (`scripts/m5b_sip_activation_patching.py`):
- 각 페어 (clean_sid, corrupted_sid) 에 대해:
  1. **Cache**: forward(clean_pil) `output_hidden_states=True` 로,
     L ∈ [0..27] 의 시각 토큰 위치 h_L 추출.
  2. **Baseline**: forward(corrupted_pil) → 첫 글자 응답.
  3. **Patched**: 각 target_L ∈ [0..27] 에 대해
     `model.model.language_model.layers[target_L]` 에 forward hook —
     prefill pass 만 (matched seq_len), 시각 토큰 위치의 출력 hidden
     state 를 cached clean 값으로 교체; text 생성; 첫 글자 파싱
     (A/B/C → physics, D → abstract).
- Layer 별 IE = (patched physics 비율) − (baseline physics 비율).

총: n=20 × (1 cache + 1 baseline + 28 patched) = 600 forward pass;
H200 에서 ~8.3 분.

## 결과

![M5b SIP per-layer IE — Qwen2.5-VL 활성화 패칭](../figures/m5b_sip_per_layer_ie.png)

| Layer | IE | Patched physics 비율 (n=20) |
|---:|---:|---:|
| L0-L9 | **+1.0** | 20/20 (full recovery) |
| L10 | +0.6 | 12/20 |
| L11 | +0.6 | 12/20 |
| L12 | +0.3 | 6/20 |
| L13 | +0.1 | 2/20 |
| L14-L27 | 0.0 | 0/20 |

Baseline corrupted physics 비율: 0/20 (모두 "D — abstract"). Clean
control: 20/20 physics (SIP 구축에 의해).

## Headlines

1. **Sharp L10 boundary.** Corrupted 의 L0-L9 hidden state 를 clean 의
   값으로 패치하면 100% 페어에서 physics-mode 회복. L10-L11 패치는
   60% 회복. L13 이상은 효과 없음.

2. **L10 = "decision lock-in" 레이어.** 이는 M5a 의 reading 을 정련:
   M5a 는 L10 이 +α·v_L10 steering 으로 행동을 뒤집는 *유일* 한
   layer. 활성화 패칭은 L0-L9 *모두* 충분한 cue 정보 운반함을 보임 —
   그러나 L10-L11 에서 LM 이 결정에 commit 시작, L14+ 는 결정 fully
   baked.

3. **정보 흐름 방향**: 시각-cue 정보가 L0 에서 진입, early-mid 블록
   (L0-L9) 통과, L10-L13 에서 텍스트-측 decision token 이 읽음,
   L14 이후엔 시각 토큰 교체로 결정 뒤집기 불가.

4. **n=20 의 100% baseline-corrupted = abstract** 가 깨끗한 실험 설정.
   L0-L9 패치의 0% → 100% IE 점프가 (line, filled, shaded, textured)
   × (blank, ground, scene) × fall 전반에 걸쳐 일관됨.

## 기존 가설과의 연결

- **H-locus** (M4-derived): "bottleneck 이 LM 중간 레이어 (L10) 에 있음"
  → **정련**. L10 은 decision lock-in 이 시작되는 *경계*; *정보* 는
  L0-L9 모두에 존재. M5a 의 "single causal layer" reading 은 additive
  steering 한정이지, full-state 패칭에는 해당 안 됨.

- **H10** (research plan §2.5): "2-3 narrow layer/head 범위에서 큰 IE"
  → **부분적 지지 with revision**. 우리는 *하나의 contiguous 범위*
  (L0-L9) 에 full IE; 2-3 narrow band 가 아님. 그러나 L10-L13 declining
  + L14+ zero 가 Kaduri et al. 2024 의 broader shape ("middle ~25% of
  layers carry cross-modal flow") 와 매칭 — Qwen 28-layer LM 에서
  L7-L21 이 middle 50%, 우리의 L0-L13 은 lower-middle.

- **M5a 의 L10 특이성**: 이제 effective steering 의 *하한* 으로 해석
  가능. L10 이전은 additive steering 에 너무 이름 (representation 이
  v_L10 방향으로 crystallize 안 됨); L10 이후는 너무 늦음 (결정 commit).
  패칭은 *전체* representation 을 transplant 하므로 더 permissive.

- **Basu et al. 2024**: "LLaVA layer 1-4 의 early MLP / self-attention 에
  constraint-satisfaction 정보 저장" — broadly consistent. 우리는 정보가
  Qwen 의 L0-L9 까지 persists 함을 보임. 그들의 finer-grained MLP vs
  attention 분해는 우리 setup 에서 미해결.

## 한계

1. **단일 모델 (Qwen2.5-VL).** Cross-model SIP+patching 은 모델별
   image-token resolution + hook adaptation 필요. LLaVA-1.5,
   LLaVA-Next (AnyRes), Idefics2, InternVL3 모두 미실시.

2. **단일 intervention type (full visual-token replacement).** Plan
   §2.5 mentions: visual token patching ✓ (이거), attention knockout
   ✗, MLP replacement ✗, steering vector intervention ✓ (M5a),
   SAE intervention ✗. 5개 type 중 3개 미해결.

3. **n=20 SIP 페어.** 모두 cue-axis (none vs both) 에서. Axis-A
   (object_level: line vs textured) 와 Axis-B (bg_level: blank vs
   ground) SIP 미구축 — seed 가 다르면 confound 됨.

4. **단일 cue effect.** 우리가 측정하는 "physics-mode 정보" 는
   cast_shadow + motion_arrow vs none 구분이 specifically 인코딩하는
   것. 다른 cue 구분 (예: shading 강도) 은 다른 layer 범위에 localize
   될 수 있음.

5. **Head-level resolution 없음.** 각 패칭이 layer 전체의 시각 토큰
   hidden state 를 교체. Per-head IE 는 layer 내부 패칭 (attention
   output decomposition) 필요.

## 재현

```bash
# GPU 1 에서 (free); H200 에서 ~10분.
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_activation_patching.py \
    --n-pairs 20 --device cuda:0
```

## Artifacts

- `scripts/m5b_sip_activation_patching.py` — driver.
- `outputs/m5b_sip/manifest.csv` — 20 SIP 페어 (cue=both clean × cue=none corrupted).
- `outputs/m5b_sip/per_pair_results.csv` — 페어 × layer 의 첫 글자 응답.
- `outputs/m5b_sip/per_layer_ie.csv` — layer 별 aggregated IE.
- `docs/figures/m5b_sip_per_layer_ie.png` — IE × layer 그림.

## 후속 가능 항목

1. **Cross-model SIP+patching**. LLaVA-1.5 (32 layer, hook
   `model.language_model.model.layers[L]`) + Idefics2 + InternVL3
   패칭으로 각 모델의 "L10 equivalent" 찾기. H9 가 LM probe AUC
   plateau 가 모델별로 다름을 보였음; M5b 는 causal intervention
   layer 테스트.
2. **Attention knockout**. L10-L13 transition zone 의 specific
   (layer, head) 페어에 대해 시각 토큰 → 마지막 토큰 attention 을
   knock out, IE 측정.
3. **Multi-axis SIP**. bg_level 이 독립적으로 toggle 되는 matched-seed
   stim 생성 — 신규 stim render 실행 필요.
4. **Head ranking**. (Layer, head) 별 IE 맵. L8-L11 에서 cue → text
   decision 운반하는 1-3 head 식별.
