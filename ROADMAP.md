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

| ID | 가설 | 상태 | 근거 / 다음 검증 |
|---|---|---|---|
| **H1** | PMR이 추상화 축(line → textured)에 따라 S자형 증가; 3D 음영·지면 도입이 가장 큰 단계 증가. | **부분 지지** | 양 끝점 (0.58 → 0.81) 맞음; 중간 tie. MVP-full에서 더 큰 n, 더 좁은 abstraction step으로 재검. |
| **H2** | "ball" 라벨은 선화에서도 PMR을 크게 증가시킨다 → 언어 prior 독립 기여. | **강하게 지지** | Open-ended PMR 0.80, abstract_reject 0.00. MVP-full에서 axis D를 {circle, ball, planet}로 확장하여 정량화. |
| **H3** | 장면 불일치는 RC를 저하시킨다. | **미검증** | Pilot 은 scene inconsistency 축 없음 + T=0 이라 RC=1 포화. MVP-full에서 `temperature=0.7`, seeds≥10, axis E 포함. |
| **H4** (pilot-derived) | Open vs forced-choice PMR gap 은 **언어 prior ↔ 시각 증거** 충돌의 안정적 signature다. 모든 object_level, 모든 모델에서 나타난다. | **후보** | ST1 full 에서 gap을 object_level 별로 쪼개 재검; ST5 에서 LLaVA/InternVL도 같은 gap 보이는지. |
| **H5** (pilot-derived) | 지면 한 줄(ground line) 단독이 텍스처 공 + no ground 보다 **더 큰** PMR 증가를 만든다. | **일방향 증거** | Pilot: bg (+36pp) > object_level endpoint gap (+23pp). 다른 모델에서도 같은 ordering이면 scene-over-object 주장. |
| **H6** (pilot-derived) | arrow+shadow cue의 포화는 **cast shadow 단독**으로도 일어나며, arrow는 annotation에 가깝다. | **분해 필요** | MVP-full axis C 재설계로 shadow/arrow 분리 → 각각의 marginal 기여 측정. |

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
| **M2** | **ST1 MVP-full** (pilot 교훈 반영) | axis C 재설계, axis D 확장, T=0.7, 활성화 capture 활성화, ~10-15k inferences | ▶ **다음** | — |
| M3 | ST2 — Vision encoder probing | M2 captured vision acts에 linear probe + Gandelsman head decomposition | 대기 | — |
| M4 | ST3 — LM logit lens / layer-wise probe | M2 captured LM acts에 logit lens + per-layer probe | 대기 | — |
| M5 | ST4 — Causal localization | SIP + activation patching + VTI steering + SAE intervention | 대기 | — |
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

### M2 — ST1 MVP-full ▶ (다음 마일스톤)

**목표**: pilot 교훈을 반영한 깨끗한 behavioral S-curve + Sub-task 2/3 용 activation 확보.

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

### M3 — ST2 Vision encoder probing

**전제**: M2 에서 vision-encoder activations 추가 capture. 현재 `PhysModeVLM.capture()` 는 LM 만 커버 → `resolve_vision_layer_path(model)` 헬퍼 + vision hooks 추가 필요.

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

### M4 — ST3 LM backbone logit lens / layer-wise probing

**전제**: M2 에서 `capture_lm_layers` 로 captured activations 존재.

**작업 분할**:
1. `src/physical_mode/probing/lm.py` 에:
   - `logit_lens_trajectory(activations, token_idx)` — 각 layer 의 hidden state 에 `model.lm_head` 적용, physics-verb tokens vs geometry-noun tokens 의 logit 추적.
   - `per_layer_pmr_probe(activations, y)` — 각 captured layer 에서 sklearn probe.
2. Figure (후보):
   - Layer × token-position heatmap: "물리 개념 부상 시공간도" (Neo et al. 2024 Fig 3 유사).
   - Per-object_level logit lens trajectory: `line` 원은 물리 동사가 후기 층까지 안 뜨고 `textured` 는 중간 층에서 뜸.
3. **Cross-model 비교** (M6 에서): LLaVA-1.5 vs Qwen2.5-VL vs InternVL2 의 switching layer 차이.

**가설**: 물리 모드는 중간 층(~15-20)에서 emerge. 연구계획 §2.4 의 Neo et al. prediction 과 align.

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

---

## 5. 작업 시 참조 규칙

**각 새 세션 시작 시** (future Claude 또는 user):

1. 이 파일 (`ROADMAP.md`) 을 제일 먼저 읽어 "지금 어느 M에 있나" 확인.
2. 현재 M 의 **성공 기준** 과 **블로킹 이슈** 를 체크.
3. 세부 기술 질문은 `docs/00_architecture.md` → `docs/04_next_steps.md` 순서로 drilldown.
4. 최신 실험 결과가 필요하면 `docs/03_run_log.md` (수치) + `docs/05_insights.md` (해석).

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
| 2026-04-24 | 최초 작성 — M0/M1 완료, M2 준비 상태까지 반영 | (this commit) |
