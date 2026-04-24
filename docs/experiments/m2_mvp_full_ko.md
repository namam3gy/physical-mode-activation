# M2 — MVP-full run (2026-04-24)

- **명령**: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`
- **자극 dir**: `inputs/mvp_full_20260424-093926_e9d79da3` (480 stimuli)
- **출력 dir**: `outputs/mvp_full_20260424-094103_8ae1fa3d`
- **모델**: `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, sdpa, **T=0.7, top_p=0.95**
- **Factorial**: 4 obj × 3 bg × 4 cue × 1 event (fall) × 10 seeds × 3 labels × 2 prompts = **2880 inferences**
- **Activation capture**: LM layers (5, 10, 15, 20, 25), hidden state 만 (attention 없음)
  → 480 `.safetensors` 파일에 5.2 GB (~11.5 MB / stimulus, 324 visual token)
- **Wall clock**: ~55 분 end-to-end (capture 포함 1.1 s / inference)
- M2 config 의 pilot 대비 차이는 `configs/mvp_full.py` 헤더와
  `docs/insights/m1_pilot_ko.md` §6 에 documented.

## 성공 기준 스코어카드 (ROADMAP M2 에서)

| 기준 | 목표 | 관측 | 상태 |
|---|---|---|---|
| object_level monotone S-curve (forced-choice) | monotone | forced: line 0.583 < filled 0.647 < shaded 0.711 < textured 0.714 | ✅ |
| 모든 object_level 에서 open-vs-forced gap | 모두 >0 | 22-32 pp (line 32, filled 29, shaded 22, textured 24) | ✅ |
| cast_shadow 단독 > none + 20 pp | +20 pp | 평균 +18.4 pp (blank +23.4, ground +18.4, scene +10.8) | ✅ (close; 엣지 조건 만족) |
| RC < 1 cell 존재 | 일부 | 103/288 cells (35.8 %) RC<1; mean RC=0.918 | ✅ |
| `outputs/*/activations/` 채워짐 | yes | 480 safetensors, LM hidden only | ✅ |

## 핵심 PMR 표

**Overall**: n=2880, PMR=0.797, hold_still=0.152, abstract_reject=0.160, GAR=0.656.

**object_level 별 (axis A)** — H1 깨끗히 지지됨:

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 720 | 0.744 | 0.193 | 0.203 | 0.594 |
| filled | 720 | 0.790 | 0.153 | 0.168 | 0.646 |
| shaded | 720 | 0.822 | 0.136 | 0.139 | 0.671 |
| textured | 720 | **0.832** | 0.126 | 0.131 | 0.713 |

4 개 수준 모두에서 monotone (중간 tie 해소). 양 끝점 gap = +8.8 pp.

**bg_level 별 (axis B)** — scene > ground > blank, pilot 의 ground 효과 재현:

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 960 | 0.669 | — |
| ground | 960 | 0.842 | 0.648 |
| scene | 960 | **0.881** | 0.664 |

Blank → scene = +21 pp (pilot 의 +36 pp blank → ground 와 비슷; T=0.7 이 delta 를 부드럽게).

**cue_level 별 (axis C)** — H6 decomposition 성공:

| cue_level | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| none | 720 | 0.540 | 0.347 | 0.479 |
| cast_shadow | 720 | **0.715** | 0.238 | 0.546 |
| motion_arrow | 720 | 0.964 | 0.031 | 0.860 |
| both | 720 | 0.969 | 0.025 | 0.738 |

Cast shadow 단독: **none 대비 +17.5 pp**. Arrow 가 0.96 에 포화; 위에 shadow
추가하면 0.97 로 약간만 nudge. Pilot 의 `arrow_shadow=1.00` 이 설명됨: arrow
가 거의 모든 일을 함; shadow 기여는 측정 가능하지만 secondary.

**Per-bg decomposition** (shadow 효과는 풍부한 배경에서 작아짐 — 포화 패턴):

| bg | none | shadow | arrow | both |
|---|---|---|---|---|
| blank | 0.287 | 0.521 (+23.4) | 0.912 | 0.954 |
| ground | 0.608 | 0.792 (+18.4) | 0.992 | 0.975 |
| scene | 0.725 | 0.833 (+10.8) | 0.988 | 0.979 |

**prompt_variant 별** — open-vs-forced gap 이 pilot 보다도 큼:

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 1440 | **0.931** | 0.002 | 0.593 |
| forced_choice | 1440 | 0.664 | 0.318 | 0.719 |

Open-ended 는 stimulus 를 abstract 로 자발적 인식 안 함 (1440 중 3개). Forced-choice
는 32 % 거부. object_level 별 gap 은 +22 pp (textured) ~ +32 pp (line) —
**더 추상적 객체일수록 큼**.

**label 별 (axis D)** — H2 직접 정량화:

| label | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| ball | 960 | **0.892** | 0.072 | 0.786 |
| circle | 960 | 0.746 | 0.186 | 0.698 |
| planet | 960 | 0.754 | 0.222 | 0.483 |

**Label × object_level 상호작용**:

| obj \\ label | circle | ball | planet |
|---|---|---|---|
| line | 0.692 | **0.846** | 0.696 |
| filled | 0.729 | **0.900** | 0.742 |
| shaded | 0.779 | **0.888** | 0.800 |
| textured | 0.783 | **0.933** | 0.779 |

**놀라운 점**: `line + ball` (PMR 0.846) > `textured + circle` (0.783) — 언어
prior 가 시각 cue 를 압도.

## 인상적 정성적 finding — 라벨이 *어떤 종류의 물리* 를 결정

같은 자극 (textured ball + ground + no cue, open-ended 프롬프트), 세 라벨:

| label | response |
|---|---|
| circle | "The circle is likely to remain static unless acted upon by an external force." |
| ball | "The ball will continue rolling down the incline." |
| planet | "The planet will continue moving along its orbital path around the Sun." |

라벨이 physics-mode on/off 만 toggle 하는 게 아님 — 모델이 적용할 *어떤 physics
regime* 인지 선택한다. `planet` 은 궤도역학 invocation (GAR=0.48), `ball` 은
gravity invocation (GAR=0.79). Figure 2 에 들어갈 paper-worthy qualitative
result.

## 가설 스코어카드 post-M2

| H | pilot status | M2 status | 변화 |
|---|---|---|---|
| H1 (S-curve) | 부분 지지 | **지지** | Middle tie 가 T=0.7 + 10 seeds 로 해소 |
| H2 (ball label) | 강한 지지 | **정량화** | +15 pp; 모든 object_level 에서 `ball > circle` |
| H3 (scene 불일치) | 미검증 | 여전히 미검증 | axis E 가 M2 에서 제외 |
| H4 (open-forced gap) | 후보 | **지지** | Gap +22 ~ +32 pp; abstraction 에 monotone |
| H5 (ground vs texture) | 일방향 | **혼재** | bg delta (+21 pp) > object delta (+9 pp); H5 지지하나 scene > ground 가 됨 |
| H6 (shadow 단독) | 분해 필요 | **지지** | Shadow +17.5 pp above none; annotation 만 아님 |

## 새 관측 (candidate 가설)

- **Per-label GAR 이 극적으로 다름** (ball 0.79 / circle 0.70 / planet 0.48).
  "Planet" 응답은 gravity 가 아닌 orbital physics 를 invoke. PMR 에 대한 label
  효과는 *binary-ish* 이지만 GAR 에 대해서는 *categorical*.
- **포화 구조**: `motion_arrow` ~≈ `both` 가 0.96-0.97. Arrow 가 dominant cue;
  shadow 의 marginal 기여는 base 가 abstract (blank bg) 일 때만 강함.
- **Open-ended 가 broken 이 아님** — 언어 prior 지배가 systematic (더 추상적
  객체에서 더 강함). Vo et al. 2025 의 "hallucinated grounding" 패턴과 일관.

## 다음 actions

- **M3 (Sub-task 2 — vision encoder probing)** unblock 됨: LM activation 캡처
  완료. Vision encoder capture 는 아직 구현 필요 (`PhysModeVLM.capture` 확장).
  Draft: vision encoder 의 3-5 layer (Qwen2.5-VL 의 SigLIP tower) 를 핵심
  factorial cell 의 ~100 stimuli 타겟 re-run 에 추가.
- **Paper 의 추가 헤드라인**: "When you call a circle a planet, it orbits"
  (label → physics-regime categorical flip). 원래 `references/project.md` 에
  없음 — pilot-to-MVP-full 단계에서 발견된 emergent finding.
  `references/roadmap.md` §4 additional ideas 에 logged.
- **Axis E (scene consistency)** 는 여전히 미검증; focused mini-experiment
  까지 deferred.
