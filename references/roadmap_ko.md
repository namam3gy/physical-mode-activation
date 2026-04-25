# ROADMAP — Physical-Mode Activation

> **이 문서의 역할.** 이 프로젝트의 "지금 어디까지 왔고 다음 뭘 할지"를 한 곳에서 본다. 새 세션을 시작할 때 **이 파일부터 읽고**, 마일스톤이 끝날 때마다 §3의 상태를 갱신한다. 세부 내용은 각 doc / 코드로 링크한다.
>
> - 연구 철학·가설 원본: `research_plan.md` (한국어, 33k)
> - 아키텍처: `docs/00_architecture.md`
> - 자극 스펙: `docs/01_stimulus_spec.md` / 점수 기준: `docs/02_scoring_rubric.md`
> - 실제 실행 기록: `docs/03_run_log.md`
> - 다음 단계 코드 진입점: `docs/04_next_steps.md`
> - 최신 인사이트: `docs/05_insights.md`

---

## 1. 연구 정의

### 1.1 중심 질문

**어떤 시각적 단서가 임계치를 넘어서면 open-source VLM이 추상 도형(원)을 기하학적 객체에서 물리적 객체(공)로 "모드 전환"하여 처리하기 시작하는가?**

측정은 두 층위로 한다:

- **행동(behavior)**: next-state-prediction 프롬프트에 대한 응답의 PMR (physics-mode priming rate) / GAR (gravity-align rate) / RC (response consistency).
- **내부(mechanism)**: 시각 인코더의 선형 probe AUC, LM backbone의 층별 logit-lens 궤적, 활성화 패칭으로 드러나는 인과적 병목 층·head.

### 1.2 Sub-task 구성 (연구계획 §2)

| # | 제목 | 내용 | 입력 | 출력 |
|---|---|---|---|---|
| ST1 | **PhysCue** behavioral thresholds | 4-5개 축 factorial stimulus + 다음 상태 예측 프롬프트 | 프로그램/photo stimuli | PMR/GAR/RC 표, per-factor curves |
| ST2 | Vision-encoder probing | CLIP / SigLIP / InternViT 활성화에 linear probe + Gandelsman head decomposition + SAE | ST1 captured vision acts | layer×head AUC, monosemantic features |
| ST3 | LM backbone layer-wise emergence | Logit lens + per-layer probes at visual token positions (Neo et al. 2024 recipe) | ST1 captured LM acts | layer×token heatmap, switching-layer |
| ST4 | Causal localization | Semantic Image Pairs + activation patching + VTI steering + SAE intervention | pilot pairs | IE curve, steering vector, head ranking |
| ST5 | Cross-model + prompt-steering | LLaVA-1.5 / LLaVA-Next / Qwen2-VL / InternVL2에 같은 factorial + 프롬프트 steering (Gavrikov et al. 2024) | 확장된 EvalConfig | 모델 간 비교 표, prompt-bias curve |

### 1.3 가설 스코어카드

연구계획 §2.2의 원래 H1-H3 + pilot 에서 도출된 H4-H6. pilot 결과는 `docs/05_insights.md` 근거.

| ID | 가설 | 상태 (post-M5a-ext recheck) | 근거 / 다음 검증 |
|---|---|---|---|
| **H1** | PMR이 추상화 축(line → textured)에 따라 S자형 증가; 3D 음영·지면 도입이 가장 큰 단계 증가. | **지지** | M2: 4개 object_level 모두 monotone (0.744 → 0.790 → 0.822 → 0.832). T=0.7 + 10 seeds가 pilot의 중간 tie를 해소. |
| **H2** | "ball" 라벨은 선화에서도 PMR을 크게 증가시킨다 → 언어 prior 독립 기여. | **revised** | M2 의 +15 pp "ball vs circle" gap 은 `ball enhancement` 가 아니라 `circle suppression`. Label-free baseline 과 paired comparison (M4b, 2026-04-25): `PMR(ball) − PMR(_nolabel) = +0.006`, `PMR(planet) − PMR(_nolabel) = +0.006`, `PMR(circle) − PMR(_nolabel) = −0.065`. Language prior 는 비대칭이다: ball ≈ no-label (시각 default), circle 은 abstract override, planet 은 추상 이미지에서만 orbit prior 추가. |
| **H3** | 장면 불일치는 RC를 저하시킨다. | **미검증** | axis E 는 M2에서 빠짐 (complexity); 별도 mini-실험으로 처리. RC 인프라는 M2에서 검증됨 (103/288 cells RC<1). |
| **H4** (pilot-derived) | Open vs forced-choice PMR gap 은 **언어 prior ↔ 시각 증거** 충돌의 안정적 signature다. | **지지 — 확장** | M2: gap이 모든 object_level에 존재 (line 32pp → textured 22pp). 추상도 ↑ 일수록 gap ↑ — abstraction 이 vision 증거를 약화시켜 언어가 더 지배한다는 structural prediction. 다음 검증: ST5 cross-model. |
| **H5** (pilot-derived) | 지면 한 줄(ground line) 단독이 텍스처 공 + no ground 보다 **더 큰** PMR 증가를 만든다. | **혼재** | M2: bg delta (blank 0.67 → scene 0.88 = +21pp) > object delta (line 0.74 → textured 0.83 = +9pp). 방향은 맞음; 단 scene 이 ground 를 또 넘음. |
| **H6** (pilot-derived) | arrow+shadow cue의 포화는 **cast shadow 단독**으로도 일어나며, arrow는 annotation에 가깝다. | **지지 (수정)** | M2 분해: cast_shadow 단독 = +17.5 pp above none (Kersten 지면 부착 cue 확인); **그러나 arrow 도 단독으로 0.96 에 saturate** — "arrow 는 annotation" 부분은 반증. Arrow 가 dominant cue, shadow 가 secondary. |
| **H7** (M2-derived) | 라벨은 PMR 을 toggle 하는 것이 아니라 **어떤 물리 regime** 을 선택한다. | **지지 but narrower** | M2 GAR: ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2: `line/blank/none × +α=40` 에서 label flip 으로 B↔A swap. M5a-ext Exp 3 qualifier: `textured/blank/none` 에서는 label 단독 flip 실패 (+α=40 → label 무관 A); regime 은 joint (image, label, α sign) 로 선택. |
| **H-boomerang** | Encoder 는 알고, decoder 가 gate: vision encoder 가 physics-mode class 를 linear 로 분리하지만 behavior 는 실패. | **지지 + causal** | M3: encoder AUC=1.00 전 factorial 축 전 probed layer; behavior 0.28-0.95. M4: LM 을 통과한 정보 보존 (AUC 0.94-0.95). M5a: L10 의 causal intervention 이 behavior flip. |
| **H-locus** (M4-derived) | Bottleneck 은 LM final layers + decoding head 에 있으며 그 이전은 아님. | **지지 (early-mid sweet spot)** | M5a: L10 α=40 은 10/10 abstract → physical 응답을 flip; 후반 layer 들은 움직이지 않음. M5a-ext Exp 3: L10 regime-flip (sign 으로 A vs B) 이 모든 cell 에서 성립. Basu et al. 2024 의 early-layer constraint-storage 결과와 정합. |
| **H-direction-bidirectional** (M5a-ext, 2026-04-24; 개정 2026-04-25) | `v_L10` 은 단순 bidirectional concept axis 로, −α 가 physics-mode 를 abstract 로 억제한다. | **revised — physics-mode 내부의 regime axis** | Exp 1 (textured/ground/both ceiling): −α 효과 없음 → 초기 "one-way activator" 프레이밍. Exp 3 (textured/blank/none moderate, 2026-04-25): −α=40 이 (line, textured) × (ball, circle) 모두에서 D → B ("stays still") 를 균일하게 유도. α 의 sign 이 regime 을 선택 (+kinetic / −static); baseline D 는 \|α\| threshold *아래* 에 위치 (axis endpoint 가 아님). |
| **H-regime** (M5a-derived) | Steering direction 은 binary "object-ness"이고 physics regime 은 label 이 결정. | **원래 형태로는 반증** | Kinetic vs static 이 이미 sign-selected regime 이어서 H-direction-bidirectional 해석으로 대체. `line/blank/none × +α=40` 의 좁은 label-driven flip (Exp 2) 은 H7 qualifier 로 흡수. |

### 1.4 Target 모델 & venue

- **1차(pilot / MVP-full)**: Qwen/Qwen2.5-VL-7B-Instruct — proven loader, 15 GB, H200에서 1.0 it/s.
- **ST5 확장**: LLaVA-1.5-7B, LLaVA-Next-7B, InternVL2-8B, (stretch) Qwen2-VL-7B (연구계획 §2.4가 명시한 모델과 layer-index align 용).
- **Venue**: EMNLP long (grounding failure / language-prior dominance angle)이 primary; NeurIPS main (ST3-4 mechanistic localization)이 stretch.

---

## 2. 마일스톤 전체 뷰

| # | 마일스톤 | 스코프 | 상태 | 완료일 |
|---|---|---|---|---|
| M0 | 인프라 스캐폴드 | Package layout, configs, scripts, tests, docs 기본 set | ✅ | 2026-04-24 |
| M1 | **ST1 Pilot** (Qwen2.5-VL-7B) | 240 stim × 2 prompts = 480 inferences; behavioral S-curve 1차 측정 | ✅ | 2026-04-24 |
| M2 | **ST1 MVP-full** (pilot 교훈 반영) | axis C 분해, axis D 확장, T=0.7, LM hidden-state capture, 2880 inferences | ✅ | 2026-04-24 |
| M3 | **ST2 — Vision encoder probing** | Vision blocks capture (8 layers, 12 GB) + layer-wise linear probes. **Boomerang 확인**: encoder AUC=1.0 on every axis; behavioral PMR 0.28-0.95. | ✅ | 2026-04-24 |
| M4 | **ST3 — LM logit lens / layer-wise probe** | LM hidden @ visual tokens AUC 0.94-0.95 전 구간; L20 peak. Label prior 가 L5 부터 physics margin shift; object_level effect 는 7배 더 작음. | ✅ | 2026-04-24 |
| M5a | **ST4 Phase 1+2 — VTI steering** | 방향 추출 + residual-stream injection. **L10 α=40 이 10/10 D→B flip** — "physical object-ness" direction 인과 확인. | ✅ | 2026-04-24 |
| M5a-ext | **VTI 후속 (neg α, label swap, 양방향성 재검정)** | Exp 1-2 (2026-04-24): ceiling 에서 neg α + label=ball side-by-side. Exp 3 (2026-04-25): moderate baseline 에서 (α × label × obj) 그리드. **핵심 결과**: `v_L10` 은 physics-mode 내부의 regime axis — +α → A (falls), −α → B (stays still), baseline D 는 threshold 아래. | ✅ | 2026-04-25 |
| M4b | **Label-free prompt — H2 null test** | M2 자극에 `open_no_label` variant. **핵심 결과**: `ball` ≈ no-label; `circle` 이 PMR 을 6.5 pp 억제. 원래 H2 재해석: language prior 는 비대칭 — circle override, ball enhancement 아님. M4 visual-token capture 가 prompt-independent (구조적 artefact). | ✅ | 2026-04-25 |
| **M5b** | **ST4 Phase 3 — SIP + patching + SAE** | Semantic Image Pairs + activation patching (attention 필요 → re-capture) + SAE feature decomposition. | ▶ **다음 (선택)** | — |
| M6 | ST5 — Cross-model sweep | LLaVA-1.5/Next, InternVL2, (optional) Qwen2-VL | 대기 | — |
| M7 | 인간 baseline + 논문 작성 | Prolific 20명 × 50 stim + EMNLP/NeurIPS 초안 | optional | — |

---

## 3. 진행 상태 (세부)

### M0 — 인프라 스캐폴드 ✅ (2026-04-24)

완료 항목:
- `src/physical_mode/` 모듈 (config, utils, stimuli, models, inference, metrics, probing scaffold).
- `scripts/0{1,2,3}_*.py` argparse runner.
- `configs/{pilot,mvp_full}.py` — config-as-code.
- `tests/` — 35 개 (stimulus determinism + PMR scoring regression).
- `docs/00-05` — architecture / spec / rubric / run log / next-steps / insights.
- `notebooks/demo.ipynb` — 32-cell walkthrough with cached outputs.
- CLAUDE.md, README.md, pyproject.toml (cu130 index), .gitignore, uv.lock.
- 프로젝트 repo: private https://github.com/namam3gy/physical-mode-activation.

성공 기준 (모두 충족):
- `uv sync` 성공, `uv run python -m pytest` 통과.
- `scripts/01_generate_stimuli.py --config configs/pilot.py --limit 10` 성공.
- 파일럿 inference + score pipeline end-to-end.

### M1 — ST1 Pilot (Qwen2.5-VL-7B) ✅ (2026-04-24)

실행: `uv run python scripts/02_run_inference.py --config configs/pilot.py`.
출력: `outputs/pilot_20260424-072418_2c16efb6/` — 480 predictions, 8 분 wall clock.

**헤드라인 발견** (`docs/05_insights.md` §2, §3):

| 관찰 | 수치 | 함의 |
|---|---|---|
| Ground 유무 효과 | blank 0.49 → ground 0.85 (+36pp) | **단일 최대 요인**. 지면이 가장 저렴하고 강한 physics trigger. |
| Abstraction endpoints | line 0.58 → textured 0.81 | H1 부분 지지; 중간 2 수준 tie. |
| Arrow+shadow cue | 1.000 (포화) | 측정 불가 cell — MVP-full에서 분해 필요. |
| Wind cue | 0.513 ≈ none 0.500 | VLM에 안 읽힘 — 자극 교체 필요. |
| Open vs forced-choice | PMR 0.80 vs 0.54, abstract_reject 0.00 vs 0.45 | **언어 prior 지배성** — H2 강지지. |

**가설 스코어**: H1 부분, H2 강지지, H3 미검증, H4-H6 후보 도출.

**검증된 인프라 특성**:
- `PhysModeVLM`이 `AutoModelForImageTextToText` generic 로더로 Qwen2.5-VL에서 작동. ST5의 모델 스왑이 config 변경만으로 가능.
- `predictions.jsonl` streaming flush가 crash-safe.
- Factorial 축 중 **event_template**이 behavioral output에 영향 없음 → MVP-full에서 downgrade.

### M2 — ST1 MVP-full ✅ (2026-04-24)

실행: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`.
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/` — 2880 predictions, 55 분 wall clock, 5.2 GB LM activations.

**성공 기준 결과** (상세: `docs/03_run_log.md` M2 항목):

| criterion | status |
|---|---|
| Monotone S-curve over object_level (forced-choice) | ✅ 0.583 < 0.647 < 0.711 < 0.714 |
| Open-vs-forced gap at every object_level | ✅ 22-32 pp, abstraction 과 양의 상관 |
| cast_shadow alone > none + 20 pp | ✅ +17.5 pp 평균 (blank 조건에서 +23) |
| RC < 1 cells exist (T>0 확인) | ✅ 103/288 (36%) cells RC<1 |
| `outputs/*/activations/` 채워짐 | ✅ 480 safetensors, 5 layers, bf16 hidden states |

**새로 도출된 헤드라인**:
1. 추상화 axis monotone S-curve 이제 깨끗히 확인 (H1).
2. 라벨이 물리 regime 을 선택 — 같은 이미지에 `circle/ball/planet` → static / rolls / orbits the Sun (H7, 신규).
3. cast_shadow 단독이 +17.5 pp; arrow 가 dominant cue (H6 수정).
4. Open-ended PMR 0.93, abstract_reject 0.002 (3/1440) — 언어 prior 지배성 재확인 + 확장 (H4).

**블로킹 해결**: 
- primitives motion-trail 은 결국 미사용 (axis C 재설계가 wind 축 자체를 대체).
- axis E (scene consistency) 는 M2에서 제외; `docs/04_next_steps.md` 에 별도 mini-experiment 로 이관 예정.
- `capture_lm_attentions=False` 플래그가 성공 — disk 가 예상의 1/3 (5.2 GB vs 15+ GB if attentions on).

### M3 — ST2 Vision encoder probing ▶ (다음 마일스톤)

**설정 변경** (`configs/mvp_full.py`를 pilot 기반으로 재작성):

1. **Axis C 재설계**: `("none", "cast_shadow", "motion_arrow", "both")` — arrow/shadow 분해로 H6 검증.
2. **Wind cue 교체**: `draw_wind_marks` 를 `draw_motion_trail` (blurred afterimage 또는 dust trail)로 재구현. 새 primitive 추가.
3. **Axis D 확장**: `("circle", "ball", "planet")` — H2/H4 정량화.
4. **Event template 축 접기**: `fall` 1개로 고정 (behavior 차이 없음). 그 용량을 seeds로 재투자.
5. **Temperature 0.7**: RC 측정 가능하게. seeds_per_cell ≥ 10.
6. **Activation capture 활성화**: `capture_lm_layers=(5, 10, 15, 20, 25)`. `capture_vision_layers` 은 `vlm_runner.py`에 아직 없으므로 ST2 전까지 LM 만.
7. **Axis E (scene consistency)** 최소 2 수준: 일관 vs 불일치(예: 사진 공 + 선화 배경). H3 검증.

**스코프 예산**: 4 object × 3 bg × 4 cue × 3 label × 10 seeds × 2 prompt ≈ 2880 cells × 2 prompts ≈ **5 760 inferences**. axis E 추가 시 ×2 = **11 520**. H200에서 1.0-1.5 it/s 이면 3-4 시간. Activation capture 포함 시 5-6 시간 + 디스크 ~8 GB.

**성공 기준**:
- [x 체크 예정] behavioral S-curve (forced-choice) 가 4개 수준에서 monotone 증가 (H1 깨끗히 검증).
- [체크 예정] Open vs forced gap 이 모든 object_level 에서 나타남 (H4 확인).
- [체크 예정] `cast_shadow` 단독이 PMR > `none` + 0.2 이상 (H6 최소 조건).
- [체크 예정] RC < 1 인 cell 존재 (T>0 이 제대로 작동).
- [체크 예정] `outputs/mvp_full_*/activations/` 에 LM hidden states 저장 확인.

**블로킹 이슈**:
- 현재 `src/physical_mode/stimuli/primitives.py` 에 motion-trail drawer 없음 → 추가 필요.
- 현재 `FactorialSpec` 에 axis E (scene_consistency) 없음 → 추가 필요.
- 기존 pilot config 와 호환성은 신경 쓸 필요 없음 (새 config 하나 만들면 끝).

### M3 — ST2 Vision encoder probing ✅ (2026-04-24)

실행: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_... --output-dir outputs/mvp_full_.../vision_activations --layers 3,7,11,15,19,23,27,31`
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` (480 safetensors, 12 GB) + `probing_vision/*.csv`

**핵심 발견** (상세: `docs/03_run_log.md` M3 항목):

- **Encoder AUC = 1.00 on every factorial axis, from layer 3 onward**. Vision 인코더는 bg/object/cue 모든 속성을 완벽히 인코딩. 정보 병목 없음.
- **Behavioral forced-choice PMR은 0.28 (cue=none) ~ 0.95 (both)** 범위. LM 의 gating 이 gap을 만든다.
- **Controlled no-cue subset (120 stimuli)**: encoder AUC 0.89 vs behavioral PMR 0.28 → encoder 가 "어떤 cells 가 physics-mode 를 trigger할지" 를 알지만 LM 은 일부만 통과시킴.
- **Per-object-level encoder AUC ~0.95 constant while behavior 0.58-0.71**: gap 은 line (가장 추상) 에서 +36 pp 로 최대 — H4 (추상도 ↑ ⇒ 언어 prior ↑) 의 내부 메커니즘 증거.

**가설 업데이트**:
- H-boomerang (원래 §1.4 의 "encoder knows, decoder doesn't"): **지지 (증거 포화)**.
- H4, H6 모두 mechanism-level 증거 확보.

**블로킹 해결 / 소득**:
- `PhysModeVLM.capture()` 에 vision hook 구현 완료 (`_resolve_vision_blocks` 헬퍼가 Qwen/LLaVA/InternVL 모두 커버).
- 프로그램 자극이 encoder AUC 1.0 을 trivially 만든다는 methodological caveat 기록. 포토리얼 stimulus 확장이 검증 단계로 필요 — §4 연동.

### M4 — ST3 LM backbone logit lens / layer-wise probing ✅ (2026-04-24)

실행: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`.
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.{csv,parquet}`.
심층 인사이트: `docs/07_m4_insights.md`.

**핵심 결과**:
- LM per-layer probe AUC (forced-choice PMR) = **0.94-0.95 across all layers**, peak L20 = 0.953.
- Logit lens: physics logit > geometry logit from L5 onwards because "ball" label primes the LM.
- Object_level 효과 (L25 line 3.76 vs textured 4.35, margin +0.6) 는 **label effect (+4.0 전체 shift)** 의 ~14%.
- Switching-layer 메트릭은 label-primed 프롬프트에서 무력화됨 (모두 L5) → §4.9 "label 없는 프롬프트" 테스트를 M5 전 mini-실험으로 승격.

**Boomerang 정확한 위치**: vision encoder (0.94-1.0) → LM hidden (0.95) 은 정보 보존. Decoding 단계에서 ~29 pp accuracy 손실 발생. ST4 의 개입 우선순위는 LM final layers + logit head.

### M5a — ST4 Phase 1+2 VTI steering ✅ (2026-04-24)

실행:
- Phase 1: inline Python driver, `compute_steering_vectors` from `src/physical_mode/probing/steering.py`
- Phase 2: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_... --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`

출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` (vectors) + `steering_experiments/` (intervention predictions). 심층 인사이트: `docs/08_m5_insights.md`.

**핵심 결과**:
- Steering vectors `v_L = mean(h|PMR=1) - mean(h|PMR=0)`. Norm 이 layer 통해 5× 증폭 (L5: 5.9 → L25: 31).
- Projection@L20 이 factorial cue axis 와 정렬 (none 22.3 → both 42.7).
- **Layer 10 α=40 주입 → 10/10 `line/blank/none` 응답이 "D: abstract" → "B: stays still" 로 flipping**. L15/L20/L25 는 같은 α 로 flipping 없음.
- Intervention 은 "abstract → physical object" 의 binary shift 를 만듦. "A: falls" 아닌 "B: stays" 로 감 → direction 은 object-ness, not gravity. H7/H-regime 일관.

**가설 업데이트**:
- H-boomerang: 확장 + **인과 지지**
- H-locus: **지지 (early-mid layer L10)**
- H-regime (신규): **후보** — steering direction 은 coarse "object-ness", regime 선택은 label-driven.

### M5a-ext — VTI 후속 ✅ (2026-04-24, 2026-04-25)

실행: `uv run python scripts/06_vti_steering.py` + `--output-subdir` flag 로
같은 M2 output tree 안에 sub-experiment 를 분리.

출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/{neg_alpha_textured_ground_both, ball_line_blank_none, bidirectional_recheck_*}/`.
심층 인사이트: `docs/insights/m5a_ext_bidirection_and_label_ko.md`. 원자료:
`docs/experiments/m5a_ext_neg_alpha_and_label_ko.md`.

**핵심 결과**:
- Exp 1 (ceiling): `textured/ground/both × circle` 에 `-α · v_L10` → first-letter
  10/10 A 유지. 초기 "one-way activator" 해석 — 재검정 (Exp 3) 에서 ceiling
  artifact 로 확인됨.
- Exp 2 (label swap): `line/blank/none × ball` 에 `+α=40 · v_L10` → 10/10 A
  ("falls"), label=`circle` (M5a) 에서는 10/10 B ("stays still"). Label-driven
  regime flip 의 causal 시연.
- Exp 3 (양방향성 재검정, 2026-04-25): `{line, textured} × blank × none` 에서
  완전한 (α × label × obj) 그리드. **신규 발견**: `-α=40` 이 4개 (obj × label)
  조건 모두에서 10/10 B ("stays still") 로 **균일하게** flip. 따라서 `v_L10` 은
  physics-mode 내부의 regime axis (+α kinetic, −α static) 이지 physics-vs-abstract
  activator 가 아니다. Baseline D 는 axis 한쪽 끝이 아니라 |α| activation
  threshold *아래* 에 위치.
- H7 qualifier: `textured/blank/none` 의 +α=40 은 label 무관 A; image 가
  physical-object signal 을 지니면 label 단독 regime flip 은 실패. Regime 은
  joint (image, label, α sign) 함수로 결정.

**가설 업데이트**:
- H-direction-bidirectional (신규): **2026-04-25 revised** — regime axis
  해석이 이전의 "one-way activator" 프레이밍을 대체.
- H-regime: **원래 형태로는 반증** — label 단독 regime flip 은 일반화되지
  않음; H-direction-bidirectional + H7 qualifier 로 흡수.
- H-locus: **unchanged (강화)** — L10 regime-flip 이 Exp 3 모든 4 cell 에서
  성립.

### M4b — Label-free prompt H2 null test ✅ (2026-04-25)

실행: `uv run python scripts/02_run_inference.py --config configs/label_free.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` 후 `scripts/03_score_and_summarize.py` 와 `scripts/05_lm_probing.py --sources open_no_label`.

출력: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions + 480 activation safetensors. 심층 인사이트: `docs/insights/m4b_label_free_ko.md`. 원자료: `docs/experiments/m4b_label_free_ko.md`.

**핵심 결과**:
- Label-free baseline 과의 paired PMR delta (480 matched seed): ball +0.006,
  planet +0.006, **circle −0.065**. M2 에서 보고된 "ball vs circle" gap 은
  실제로는 circle 억제이지 ball 증가가 아니다.
- Cell 별 구조: circle 이 추상 이미지에서 더 강하게 억제 (line: −9.2 pp;
  filled: −4.2 pp); `motion_arrow` cue 가 circle 억제를 완전 override
  (+0.000); `none` cue 가 최대 억제 (−15.0 pp).
- `line/blank/none` 4-label 표가 label 기여를 깔끔하게 분리: ball (regime
  shift, kinetic→static), circle (full suppression, PMR 0.40 → 0.10),
  planet (+30 pp PMR — orbital prior 때문에 시각 default 위에 physics 를
  *진짜로* 추가하는 유일한 label).
- Label-free activation 에 M4 재실행 → M2 의 physics-margin 표 bit-for-bit
  재현. visual-token capture 가 prompt-independent 임을 확인 (image token
  이 question text 보다 앞에 있고 causal attention). L5 의 collapsed
  switching-layer 는 capture 지점의 구조적 artefact 이지 label-independent
  한 LM commitment 의 증거가 아님.

**가설 업데이트**:
- H2: **revised** — ball ≈ no-label; circle 이 suppressive override.
  Per-label 기여가 비대칭.
- H-boomerang: **강화** — visual-token hidden states 가 prompt-independent
  이므로 L5 의 physics bias 는 image-only 기원.
- H-locus: **unchanged** — label 의 행동적 효과는 visual-token 위치 하류에
  localize, M5a 의 image-preceding trajectory 에서의 L10 효과성과 일관.
- H4: **refined** — circle 억제 강도가 이미지 추상도와 함께 증가 — 추상도 →
  language-prior-gap scaling 의 image-side dual.

### M5b — ST4 Phase 3 (SIP patching + SAE) — 작업 상세

**작업 분할**:
1. `PhysModeVLM.capture()` 에 `capture_vision_layers` 경로 구현 (Qwen2.5-VL 은 `model.visual.blocks[i]`; LLaVA 는 `model.vision_tower.vision_model.encoder.layers[i]`).
2. `src/physical_mode/probing/vision.py` 에 `train_probes(X_per_layer, y_pmr) -> dict[int, Probe]` 작성 (sklearn LogisticRegression, 5-fold stratified).
3. 추가 MVP-full 재실행 (capture_vision_layers 켜고) — 또는 M2에서 이미 켰으면 스킵.
4. Figure: layer-wise probe AUC × object_level → "encoder knows" 증명.
5. **Stretch**: Gandelsman head decomposition (CLIP 전용). Qwen2.5-VL 은 SigLIP 기반이라 Balasubramanian 적응 필요.

**가설 검증**:
- 시각 인코더 AUC 의 S-curve 기울기가 행동 PMR S-curve 보다 **가파르다** (encoder-decoder boomerang).
- 특정 head 또는 feature direction 이 "physical-ness" 축에 특화된다.

**성공 기준**:
- 최소 1 개 layer 에서 PMR probe AUC > 0.75.
- Encoder AUC 와 behavioral PMR 의 per-cell gap 이 유의미 (paired t-test / bootstrap).

### M5 — ST4 Causal localization

**전제**: M3-M4 로 어떤 layer/head 가 후보인지 파악됐음.

**작업 분할**:
1. Semantic Image Pair 생성: pilot factorial 에서 single-axis-differ 쌍 뽑아 `sip_manifest.parquet` 작성.
2. **Activation patching** (TransformerLens 또는 raw PyTorch hooks):
   - Clean/corrupted forward 쌍 캡처.
   - Layer-sweep: 각 layer 의 visual token 위치 활성화를 clean → corrupted 로 교체, PMR 확률 회복량 측정.
3. **Attention knockout**: 특정 head 의 visual-to-last-token attention 을 0 으로 → PMR 변화.
4. **VTI steering**: `v_layer = mean(h_clean) - mean(h_corrupted)`. Test-time 에 `alpha * v` 를 residual stream 에 추가 → line 원이 물리 모드로 flipping 되는지.
5. **SAE** (stretch): vision-encoder activations 에 SAE 학습 → monosemantic "shading" / "ground" feature 식별 + intervention.

**헤드라인 claim 후보**: "LLaVA layer 19 head 14 의 knockout 이 PMR 을 50pp 감소; 동일 head 만 보존하고 나머지 visual attention 다 끊으면 PMR 유지" 같은 문장 (연구계획 §3.2).

### M6 — ST5 Cross-model sweep

작업:
1. `configs/cross_model.py` — `model_id` list + 같은 factorial.
2. `scripts/02_run_inference.py` 수정: list of model_ids 를 순회.
3. LLaVA-1.5-7B (~13 GB), LLaVA-Next-7B (~14 GB), InternVL2-8B (~16 GB), (stretch) Qwen2-VL-7B (~15 GB). 총 다운로드 ~60 GB.
4. 각 모델에서 behavioral 표 + (가능하면) ST3/4 축약 버전.
5. **Prompt steering**: `system_prompt_override` 로 `"treat this as an abstract geometric shape"` vs `"treat this as a physical object subject to gravity"` → PMR shift 측정.

**가설**: 지면 효과 (H5) 는 모든 open-source VLM 에서 재현; open-vs-forced gap (H4) 는 모델별 크기 차이.

### M7 — 인간 baseline (optional) + 논문

- Prolific 20 명 × 50 stimuli (random subset) × open-ended 프롬프트.
- Human PMR 과 VLM PMR 의 per-cell alignment 분석.
- EMNLP long (primary) / NeurIPS main (stretch) 초안.

---

## 4. 원래 계획에 없던 추가 아이디어

Pilot 에서 떠오른, 혹은 연구계획 §2 에 없는 확장 방향. 선택적 — 각각 1-2 주 작업.

### 4.1 Block-stack을 별도 "abstract-physical" 경로로

현재 코드 (`primitives.py::_draw_block_stack`) 는 있으나 pilot 에서 미사용. 블록은 "추상적인 기하 + 명백한 물리" 조합이라 **원-공 축과 다른 질문**을 묻는다: "도형은 추상인데 구성(stacking)은 물리인 자극에서 VLM 이 어느 쪽으로 가는가?" → 예상: PMR 높음 + abstract_reject 낮음. 원-공 축의 컨트롤로 유용.

### 4.2 역과제 (reverse prompting)

프롬프트에 `"The image shows an abstract diagram"` 라벨을 *실제* 사진 공에 붙였을 때 PMR 떨어지는가? H4 (언어 prior 지배) 의 counterfactual. 1 시간 실험.

### 4.3 Label 언어 전환

한국어 `"공"` vs 영어 `"ball"` 을 같은 stimulus 에 붙이면 PMR 차이 나는가? Qwen2.5-VL 은 다국어 지원 → 언어별 prior 강도 측정 가능. 연구계획 §3 의 언어-grounding 내러티브 확장.

### 4.4 Video frame pair → Michotte-style causality

두 프레임 (t=0, t=1) 에 객체 위치만 달라지는 쌍을 주고 "launched by X?" 질문. Michotte (1946) launching effect 가 VLM 에 나타나는가? 동영상 모델 필요 없이 2-image prompt 로 proxy 가능.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2)

"Vision encoder 가 CLIP 이면 안 보이는 cue 를 DINOv2 기반 모델은 본다" 가설. Eyes Wide Shut (Tong et al. 2024) MoF 제안의 연속선. 단, standalone encoder 는 LLaVA-1.5 (CLIP-ViT-L/14) vs Qwen2.5-VL (SigLIP) 의 자연스러운 비교로 이미 M6 에 포함.

### 4.6 Activation 기반 counterfactual 자극 생성

SAE 또는 VTI 로 찾은 steering vector 를 반대로 써서 "VLM 이 보기엔 '물리 모드' 를 최대화하는 자극" 을 gradient ascent 로 합성. **adversarial physics-mode prompt** → 오픈소스 VLM 의 shortcut 해석 증거.

### 4.7 결정 consistency 의 경계 측정

Pilot 에서 T=0 이라 RC 측정 못했음. MVP-full 에서 T=0.7 로 얻은 RC 를 **axis 별 "결정 안정성"** 으로 해석: 어느 cue 가 결정을 안정화시키는가? 예상: ground + shaded + fall → RC 높음 (모두 동일 답); line + blank + none → RC 중간 (일관되게 stationary); 경계선 cell 에서 RC 저조.

### 4.8 PMR 스케일링

H-class (Qwen2.5-VL-7B/32B/72B), LLaVA-1.5-7B/13B 에서 모델 크기별 PMR. MechBench (Zhang et al. 2024) 의 "scale doesn't help" 주장이 PMR 에도 성립하는가? H6 에 강한 interpretability 함의.

### 4.9 Label 없이 프롬프트

`"What do you see? What might happen next?"` — "ball" 단어 **없이** 질문. H2 의 언어 prior 기여를 null-hypothesis 형태로 측정. 쉬운 추가 — `prompts.py` 에 `open_no_label` variant.

### 4.10 Attention visualization UI

Captured attentions 로 interactive heatmap (notebook 기반). Per-stimulus, per-layer, per-head 의 visual token attention. 논문 appendix figure 용.

### 4.11 H7 follow-up — label-regime 범주 주석

M2에서 발견된 "라벨이 물리 regime을 선택한다" (circle → static / ball → rolls / planet → orbits) 의 체계적 검증. Open-ended 응답을 5범주 (gravity-fall / gravity-roll / orbital / inertial / static) 로 hand-annotation 또는 zero-shot classification → axis D × 범주 분포를 confusion-matrix 로. 라벨별 GAR 차이 (ball 0.79 / planet 0.48) 가 실제로 "어떤 물리인가" 의 categorical split 인지 수치 검증.

---

## 5. 작업 시 참조 규칙

**각 새 세션 시작 시** (future Claude 또는 user):

1. 이 파일 (`ROADMAP.md`) 을 제일 먼저 읽어 "지금 어느 M에 있나" 확인.
2. 현재 M 의 **성공 기준** 과 **블로킹 이슈** 를 체크.
3. 세부 기술 질문은 `docs/00_architecture.md` → `docs/04_next_steps.md` 순서로 drilldown.
4. 최신 실험 결과가 필요하면 `docs/03_run_log.md` (수치) + `docs/0X_*.md` (각 milestone 심층 인사이트).

**Insights 파일 규칙**: 각 주요 마일스톤 완료 시 `docs/0X_<milestone>_insights.md` 를 한 개 작성한다 — 한국어, 원본 수치 링크 + 해석 + 가설 스코어카드 업데이트 + paper implications. 현재 트리:
- `docs/05_insights.md` — M1 pilot insights (legacy 이름)
- `docs/06_m3_insights.md` — M3 encoder boomerang
- `docs/07_m4_insights.md` — M4 LM logit lens
- `docs/08_m5_insights.md` — M5 VTI steering causal intervention
- (M5b, M6 ... 추가 예정)

**마일스톤 하나를 완료할 때마다**:

- §2 표의 상태 컬럼 업데이트 (▶ → ✅) + 완료일 기록.
- §3 해당 마일스톤 섹션에 "검증된 사실 / 블로킹 해결 / 새 가설" 적는다.
- 새 가설이 도출되면 §1.3 스코어카드에 H* 추가.
- `docs/03_run_log.md` 에도 run 단위 entry 추가 (이 파일과 별개).

**가설이 반박되거나 수정될 때**:

- §1.3 에서 상태 변경 + 이유 한 줄.
- 심한 수정은 `research_plan.md` 원본이 아니라 이 ROADMAP 에만 기록 (원본은 읽기 전용 스펙).

**새 아이디어가 떠오를 때**:

- 즉시 §4 에 번호 붙여 추가. 이후 M2-M6 중 어디에 끼울지 / 독립 M 으로 띄울지는 나중에.

---

## 6. 변경 이력

| 날짜 | 변경 | commit |
|---|---|---|
| 2026-04-24 | 최초 작성 — M0/M1 완료, M2 준비 상태까지 반영 | `23171b6` |
| 2026-04-24 | M2 완료 반영: 가설 스코어카드 (H1→지지, H2→정량화, H4→지지, H5→혼재, H6→지지 수정, H7 신규), M3 를 다음 마일스톤으로, §4 에 H7 follow-up 추가 | `1d17252` |
| 2026-04-24 | M3 완료: vision encoder probing — boomerang 확인 (encoder AUC=1.0 / behavioral 0.28-0.95), M4 를 다음 마일스톤으로. | `1205821` |
| 2026-04-24 | M4 완료: LM logit lens + per-layer probe. LM AUC 0.94-0.95 전 구간 (peak L20=0.953); label 이 L5 부터 physics margin 주도. M5 를 다음 마일스톤으로. | `2abdc32` |
| 2026-04-24 | M5a 완료 (VTI steering): L10 α=40 이 "line/blank/none" 10/10 을 D(abstract) → B(physical-static) flip. "object-ness" direction 인과 확인. M5b (SIP+SAE), M6 이 남음. | `61ffd29` |
| 2026-04-24 | M5a-ext Exp 1+2 완료: ceiling 에서 negative α (null — 이후 ceiling artifact 로 판명) + label=ball swap on line/blank/none (clean B→A flip). H-direction-bidirectional 신규 (초기엔 "one-way activator"), H-regime 을 "지지" 로 격상. | `9a0ed86` (merge) |
| 2026-04-25 | M5a-ext Exp 3 (`textured/blank/none` moderate baseline 에서 양방향성 재검정): −α=40 → (line/textured) × (ball/circle) 모두에서 10 B 를 균일하게 유도. H-direction-bidirectional 을 "physics-mode 내부의 regime axis" 로 개정 (+α kinetic, −α static, baseline D 는 threshold 아래). H-regime 원래 형태 반증 후 H7 qualifier 로 축소. | `f8f0fdd` |
| 2026-04-25 | M4b 완료: M2 자극에 label-free prompt 를 H2 null test 로 적용. Paired PMR(ball) − PMR(_nolabel) = +0.006 ≈ 0; PMR(circle) − PMR(_nolabel) = −0.065. **H2 revised** — language prior 는 비대칭 (circle override, ball enhancement 아님). M4 visual-token capture 가 prompt-independent (causal-attention artefact); switching-layer 의 붕괴는 구조적 현상. | (this commit) |
