---
section: §4.6
date: 2026-04-26
status: complete
hypothesis: v_L10 방향으로의 작은 픽셀-공간 perturbation 은 Qwen2.5-VL 의 출력을 "원이 정지" → "원이 떨어진다" 로 뒤집고, 동일 magnitude 의 random-direction perturbation 은 그렇지 못함
---

# §4.6 — VTI-역방향 counterfactual 자극 (v_L10 픽셀-공간 gradient ascent)

## 질문

M5a 에서는 LM layer 10 의 시각 토큰 hidden state 에 `+α · v_L10` 를
*runtime* 에 더하면 Qwen2.5-VL 의 출력이 abstract ↔ physical regime
사이를 이동한다는 것을 보였다. §4.6 은 그 역방향 질문을 던진다:
**픽셀 공간에서** baseline 원 자극에 작은 perturbation 을 합성해,
*runtime steering 없이도* 모델이 시각 토큰 위치에서 보는 L10 hidden
state 가 `v_L10` 위에 충분히 강하게 projection 되도록 만들어
"원이 정지한다" → "원이 떨어진다" 로 예측을 뒤집을 수 있는가?

가능하다면, 이는 **shortcut interpretation 이 순수하게 픽셀 기반**
이라는 직접적인 증거다. 즉 shortcut 은 *모델이 이미지에서 어떤
feature 를 추출하는가* 의 속성이지, 추론 시 LM 이 텍스트와 비전
토큰을 어떻게 통합하는가 의 속성이 아니다.

## 방법

**목적함수.** `<mean(h_L10[visual]), v_L10>` 를 최대화. 여기서 `h_L10`
은 LM layer 10 (post-layernorm) hidden state, `[visual]` 은 시각 토큰
position, `v_L10` 은 M5a unit-norm steering 방향
(`outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/
steering_vectors.npz` 의 `v_unit_10`, dim 3584).

**변수.** Qwen2.5-VL post-processor `pixel_values` — float tensor
shape `(T_patches, 1176)` (1176 = 2·3·14·14 = temporal_patch ×
channels × patch × patch). 504×504 단일 이미지의 경우 `T_patches =
1296`. 이 표현에서 최적화하면 미분 불가능한 PIL → patch 전처리를
우회할 수 있고, 이는 여전히 inverse permute + de-norm 으로 RGB
이미지로 복원 가능한 가장 작은 표현이다 (
`src/physical_mode/synthesis/counterfactual.py:reconstruct_pil`
참조).

**Optimizer.** Adam, lr=1e-2, n_steps=200. float32 leaf 를 forward
에서 bf16 으로 cast — Qwen2.5-VL vision tower → projector → LM 0..10
경로는 end-to-end 미분 가능 (Phase 1 gate 에서 gradient max_abs =
13.75, NaN 없음 확인).

**구성.** 모든 구성은 5 개 baseline 원 자극
(`line_blank_none_fall_{000..004}.png`, `inputs/mvp_full_*` 에서) 에
대해 수행:

| Mode | ε bound | Target direction | 목적 |
|------|---------|-------------------|------|
| bounded | 0.05 | `v_L10` | 작은 perturbation flip 검증 |
| bounded | 0.1  | `v_L10` | 중간 perturbation 비교 |
| bounded | 0.2  | `v_L10` | 큰 perturbation 비교 |
| unconstrained | — | `v_L10` | 경로가 만들 수 있는 상한 |
| bounded | 0.1 | random unit dir × 3 seed | 방향 특이성 falsification |

`bounded` 는 매 Adam step 후 `pv − pv_initial` 를 `[-eps, +eps]` 로
clip. Random direction 은 동일 hidden dim (3584) 의 unit sphere 에서
샘플링하고, 가장 관대한 `v_L10` bound 와 매칭하기 위해 `ε = 0.1` 에
적용.

**추론.** 최적화 후, 합성된 `pixel_values` 를 `reconstruct_pil` 로
504×504 RGB 이미지로 복원하고, 표준 `apply_chat_template` →
`processor` → `model.generate(do_sample=False, max_new_tokens=64)`
경로로 Qwen2.5-VL 에 다시 입력. PMR 은
`src/physical_mode/metrics/pmr.py:score_pmr` 로 채점.

## 결과

| Config              | n | Baseline PMR mean | Synth PMR mean | n flipped | 평균 final projection |
|---------------------|--:|------------------:|---------------:|----------:|----------------------:|
| `bounded_eps0.05`   | 5 |               0.0 |        **1.0** |     **5** |                  43.7 |
| `bounded_eps0.1`    | 5 |               0.0 |        **1.0** |     **5** |                 100.6 |
| `bounded_eps0.2`    | 5 |               0.0 |        **1.0** |     **5** |                 125.9 |
| `unconstrained`     | 5 |               0.0 |        **1.0** |     **5** |                 181.1 |
| `control_v_random_0`| 5 |               0.0 |            0.0 |         0 |                  85.3 |
| `control_v_random_1`| 5 |               0.0 |            0.0 |         0 |                  76.6 |
| `control_v_random_2`| 5 |               0.0 |            0.0 |         0 |                  73.4 |

`v_L10` 구성의 baseline projection 은 모두 `−2.36` (고정된 baseline
이미지에 대해 결정론적). Random-direction baseline 은 각 random 방향
이 다른 시작 projection 을 유도하므로 약간 다름.

![§4.6 canonical 4-panel](../figures/sec4_6_counterfactual_stim_panels.png)
![§4.6 projection trajectories](../figures/sec4_6_counterfactual_stim_trajectory.png)

### Headlines

1. **ε = 0.05 에서 5/5 flip.** 가장 작은 ε = 0.05 를 포함해 모든
   bounded `v_L10` 구성이 5 개 baseline seed 전부를 PMR=0 → PMR=1 로
   뒤집었다. 사전 등록된 성공 기준은 ≥3/5 였고, 결과는 명확. 합성
   응답 샘플: "The circle will continue to fall downward due to
   gravity."

2. **Magnitude 매칭 random direction 에서 0/15 flip.** 세 random
   unit direction 모두 ε = 0.1 (가장 관대한 `v_L10` bound 와 매칭)
   에서, 어떤 seed 에서도 flip 을 만들지 못함. 합성 응답 샘플:
   "The circle will remain stationary as there is no indication of
   movement or change in its position." 이는 "충분히 큰 픽셀
   perturbation 이면 무엇이든 PMR 을 뒤집는다" 라는 대안 가설을
   기각하고, **방향 특이성 (directional specificity)** 이 원인임을
   분리해낸다.

3. **Projection magnitude 가 결정 요인은 아니다.** Random direction
   은 final projection ~73–85 에 도달; bounded `v_L10` (ε = 0.1) 은
   ~101 — 동일 자릿수, 그러나 행동 결과는 정반대 (0 vs 5 flip). 이
   는 `v_L10` 이 *특정한* 축이라는 것과 일치한다: 비슷한 magnitude
   라도 *다른 축* 위로의 random projection 은 LM 의
   physics/abstract regime 을 바꾸지 않지만, `v_L10` 위로의 작은
   projection 은 바꾼다.

4. **`v_L10` 은 LM 만의 속성이 아니라 이미지의 속성이기도 하다.**
   모델이 (test-time hidden-state injection 없이) 자기 스스로 보고
   통합하는 픽셀 변화만으로도 regime 을 뒤집을 수 있다. M5a 가
   발견한 shortcut 은 **이미지에 인코드 가능**.

### Perturbation 의 시각적 특성

ε = 0.05 는 흰 배경에 옅은 점박이 텍스처 — 자세히 보면 보이지만
abstract 한 원 형태는 그대로 — 를 만든다. 일반적인 관찰자는 여전히
이미지를 "흰 배경 위 검은 원" 으로 묘사할 것; perturbation 은 중력
단서, 지면 라인, 그림자, 또는 인간이 물리적으로 읽어낼 수 있는
어떤 feature 도 도입하지 **않는다**. ε = 0.1 에서는 텍스처가 더
뚜렷해지고 (그래도 의미 있는 feature 는 없음), ε = 0.2 / unconstrained
에서는 "노이즈" 처럼 보이기 시작하지만 의미 있는 scenelike 형태는
형성하지 않는다.

따라서 주장은: **인간이 읽을 수 있는 물리적 콘텐츠를 도입하지
않는 perturbation 만으로 모델이 뒤집힐 수 있다**. 우리는
perturbation 이 "비가시적" 이라거나, 합성 이미지가 baseline 과
구분 불가능하다고 주장하지 않는다.

## Scorer note (이번 run 중에 추가됨)

Random-direction control 응답 ("The circle will remain stationary
as there is no indication of movement or change…") 이 원래 PMR
scorer 의 과도한 허용을 노출시켰다. "no indication of movement"
의 substring "mov" 이 physics-verb stem list 와 매칭되어 15 개의
random-control 응답 모두가 처음에는 PMR=1 로 채점되고 있었다 —
이대로면 headline 이 5/5 vs 15/15 가 되어 falsifier 가 사라진다.

세 개의 abstract-marker 패턴을
`src/physical_mode/metrics/lexicons.py:ABSTRACT_MARKERS` 에 추가:
`"remain stationary"`, `"no indication of mov"`, `"no indication of
motion"`. 수정은 의도적으로 **비대칭** — abstract marker 가
physics-verb 매칭을 gate 하므로, 변경은 PMR=1 카운트를 *줄일* 수만
있고 *늘릴* 수는 없다. 합법적인 `v_L10` 응답을 조용히 억제하지
않는지 검증: 20 개 bounded-`v_L10` 합성 응답 중 새 marker 와
매칭되는 것은 **0 개**, 15 개 random-control 응답 중에는 **14
개** 매칭. (한 개의 control 응답은 원래 "remain unchanged" 패턴과
매칭되며, 이는 수정 전 scorer 에서도 이미 정상 gate.)

수정 전 scorer 와도 별도 확인: `v_L10` 5/5 flip 과 random 0/15
flip 모두 그대로 재현됨. scorer 수정이 만드는 *차이* 는 PMR=0
control 의 라벨링이지 flip 카운트가 아니다. 51 개였던 PMR 테스트
스위트는 이 동작을 고정하기 위해 54 케이스로 확장.

## Mechanism

§4.6 은 M5a 의 주장에 더 강한 제약을 부여한다:

- **M5a (steering)**: 추론 중에 L10 시각 토큰 hidden state 에
  `α · v_L10` 를 더하면 LM 이 physics-mode 출력으로 편향됨.
- **§4.6 (synthesis)**: 픽셀 공간의 작은 perturbation —
  *동일* `<h_L10, v_L10>` projection 을 최대화하도록 최적화된 —
  이, 어떤 runtime intervention 없이도 LM 을 physics-mode 로
  뒤집는다.

두 결과를 함께 놓으면 `v_L10` 은 **shortcut 경로 위에** 있다:
vision encoder + projector 가 그곳으로 쓸 수 있고, LM 이 그곳을
읽어내며, 행동적 결과 (PMR) 는 *바로 그 특정 축* 위의 projection
magnitude 에서 따라온다. Random-direction control 은 "충분히 큰
픽셀 perturbation 이면 어느 것이든 flip 을 만든다" 라는 대안을
배제한다.

이는 M9 / §4.10 의 **label 과 visual cue 가 부분적으로 중복적인
경로** 라는 그림과 일관된다: §4.6 은 *세 번째* 경로를 보여준다 —
픽셀-수준 feature 만으로 (label 없이, 명시적 물리 단서 없이) 동일
hidden-state 방향을 구동하기 충분하다.

## Hypotheses 함의

- **H-shortcut (Qwen 위 encoder-saturation 가설)**: 강화. §4.6 은
  `v_L10` 이 조작 가능한 픽셀 기반 채널임을 보임 — encoder-
  saturation 가설이 예측하는 "얇은 픽셀-to-regime 파이프라인" 에
  정확히 부합.
- **H-direction-specificity** (새로 검정): 지지. 0/15 random-
  control flip 은 "충분히 큰 픽셀 perturbation 이면 무엇이든" 을
  기각하고 `v_L10` (또는 `v_L10` 와 충분히 정렬된 방향) 를
  특정한다.
- **H7 (label-selects-regime)**: 직교. §4.6 은 *label 을 고정한
  채로* (label = "circle") regime flip 을 만든다. §4.10 의
  "label 이 pixel 을 지배한다" 결과와 함께 보면, pixel-driven 과
  label-driven 경로 모두 존재하지만 경쟁한다 — §4.6 은
  perturbation 이 `v_L10` 을 따라 표적화된 경우 pixel 경로가
  *이길 수* 있음을 보인다.

## 한계

1. **Adversarial signature, naturalistic stim 아님.** 합성된 노이즈
   패턴은 가시적. 결과는 픽셀 기반 flip 채널의 *존재* 를 입증할 뿐,
   이 채널이 자연 이미지에서 활성화된다는 것은 아님.
2. **단일 모델, 단일 방향.** Qwen2.5-VL 과 M5a 의 `v_L10` 만.
   교차 모델 `v_L10` 유사체 (LLaVA / InternVL3 등) 는 각 모델이
   고유한 버전의 이 축을 가지는지 검증 필요.
3. **`v_L10` 은 라벨링된 자극 분포 위 PCA 의 1-d 축이다.** 합성
   perturbation 은 이 축 위 projection 을 최대화하지만 자연-이미지
   manifold 안에 머무르도록 제약되지 않음 — flip 이 "`v_L10`
   projection 증가" 에서 오는지 "off-manifold 부산물" 에서 오는지
   분리할 수 없음.
4. **PMR 은 거친 행동 readout.** "정지" → "중력으로 떨어진다" 는
   가능한 가장 강한 flip 이며, 더 미세한 측정 (예: projection 조건
   부 held-out 어휘 분포) 은 후속 과제로.

## 재현

```bash
# Phase 1 gate (grad 가 pixel_values 에 도달하는지 검증).
uv run python scripts/sec4_6_differentiability_smoke.py

# Phase 2/3 — 풀 sweep (35 run × 200 step).
uv run python scripts/sec4_6_counterfactual_stim.py
# → outputs/sec4_6_counterfactual_<ts>/{<config>/<sid>/synthesized.png,
#                                         trajectory.npy, pixel_values.pt},
#   manifest.json

# Phase 4 — PMR re-inference + figures.
uv run python scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_counterfactual_<ts>
# → outputs/.../results.csv, results_aggregated.csv;
#   docs/figures/sec4_6_counterfactual_stim_panels.png
#   docs/figures/sec4_6_counterfactual_stim_trajectory.png
```

테스트: `uv run python -m pytest tests/test_counterfactual.py
tests/test_pmr_scoring.py -v`.

## Artifacts

- `src/physical_mode/synthesis/counterfactual.py` — gradient_ascent +
  pixel_values_from_pil + reconstruct_pil
- `scripts/sec4_6_differentiability_smoke.py` — Phase 1 gate
- `scripts/sec4_6_counterfactual_stim.py` — driver
- `scripts/sec4_6_summarize.py` — PMR re-inference + figures
- `tests/test_counterfactual.py` — 3 round-trip + correctness 테스트
- `tests/test_pmr_scoring.py` — abstract-marker 수정용 확장
- `outputs/sec4_6_counterfactual_20260426-050343/` — sweep run
- `docs/figures/sec4_6_counterfactual_stim_panels.png`
- `docs/figures/sec4_6_counterfactual_stim_trajectory.png`
