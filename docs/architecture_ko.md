# 아키텍처 — Physical-Mode Activation

**대상 독자**: 이 프로젝트를 작업하는 future Claude Code 세션. 코드를 만지기
전에 이 문서를 먼저 읽으십시오. 연구 spec 의 canonical 파일은
`references/project.md` (영어; 한국어 번역은 `references/project_ko.md`); 이
문서는 그 §2.2 를 코드로 번역한 *implementation* contract 이다.

## 한 단락 요약

추상화 / 배경 / 맥락 단서가 매개변수적으로 변하는 프로그램적 이미지(원, 블록 스택)
factorial 을 렌더링하고, open-source VLM 에 open-ended + forced-choice next-state
prediction 두 가지 프롬프트를 던진 후, 응답을 세 가지 행동 메트릭(PMR, GAR, RC)
으로 점수화한다. Inference 시점에 hidden state 와 attention 을 캡처할 수 있게
설계되어 있어, layer-wise probe (Sub-task 2) 와 logit-lens / causal-patching
분석 (Sub-tasks 3-4) 을 inference 재실행 없이 추가할 수 있다.

## 모듈 맵

```
src/physical_mode/
├── config.py         # EvalConfig + FactorialSpec + StimulusRow dataclass
├── utils.py          # seed, timestamp, config_hash, WORKSPACE 상수
├── stimuli/
│   ├── primitives.py # PIL 드로어: 원, 음영, 지면, 바람, 화살표, 그림자
│   ├── scenes.py     # compose(row) -> PIL.Image
│   └── generate.py   # inputs/<run_id>/{images/, manifest.parquet} 출력
├── models/
│   └── vlm_runner.py # PhysModeVLM — AutoModelForImageTextToText + hidden-state capture
├── inference/
│   ├── prompts.py    # open-ended + forced-choice 템플릿, label 매개변수화
│   └── run.py        # 메인 루프: predictions.jsonl 스트리밍 + activation 저장
├── metrics/
│   ├── lexicons.py   # PHYSICS_VERB_STEMS, DOWN_DIRECTION_PHRASES, ABSTRACT_MARKERS
│   └── pmr.py        # score_pmr / score_gar / score_rc / summarize
└── probing/          # Sub-task 2-4 모듈
    ├── vision.py     # ST2 — vision encoder linear probe
    ├── lm.py         # ST3 — LM logit lens + per-layer probe
    └── steering.py   # ST4 — VTI steering vector
```

`scripts/0{1,...,6}_*.py` 는 라이브러리에 대한 얇은 argparse wrapper. `configs/*.py`
는 `CONFIG = EvalConfig(...)` 를 노출하는 Python 파일; 스크립트가
`importlib.util.spec_from_file_location` 으로 로드해서 별도 registry 없이도
config 가 in-tree 타입을 직접 참조할 수 있다.

## Factorial 축 (`references/project.md` §2.2 에서 축소)

| 축 | 코드 | 수준 |
|---|---|---|
| A — 객체 추상화 | `object_level` | `line` · `filled` · `shaded` · `textured` · `block_stack` |
| B — 배경 | `bg_level` | `blank` · `ground` · `scene` |
| C — 맥락 cue | `cue_level` | `none` · `cast_shadow` · `motion_arrow` · `both` (legacy: `wind`, `arrow_shadow`) |
| D — 객체 라벨 (프롬프트 시점) | `label` | `circle` · `ball` · `planet` · `shape` · `object` |
| E — 장면 일관성 | (미래) | 이번 라운드에서는 조작 안 함 |

5 개의 event template (`fall`, `horizontal`, `hover`, `wall_bounce`, `roll_slope`)
이 객체의 캔버스 위 위치를 통제; `pilot.py` 와 `mvp_full.py` 는 처음 두 개만 사용.

## 데이터 흐름

```
FactorialSpec.iter()
  → StimulusRow (sample_id, factor 수준, seed)
  → render_scene(row) → PIL.Image
  → inputs/<run_id>/images/<sample_id>.png + manifest.parquet
       │
       ▼
PhysModeVLM(model_id, ...)
  for 각 (stimulus × label × prompt_variant):
    .generate(image, rendered_prompt, choice_tokens) → {raw_text, token_info, option_logits}
  for 각 stimulus (한 번만):
    .capture(image, rendered_prompt) → hidden_states + attentions
  ↓
outputs/<run_id>/
  ├── predictions.jsonl    (inference 단위 streaming)
  ├── predictions.parquet  (flat, 분석용)
  ├── predictions.csv
  ├── activations/<sample_id>.safetensors  (capture_lm_layers 가 set 된 경우만)
  └── run_meta.json
       │
       ▼
score_rows → pmr, gar, hold_still, abstract_reject
summarize  → summary_{overall, by_object_level, by_bg_level, ...}.csv
response_consistency → response_consistency.csv
       │
       ▼ (probing 후속 작업)
probing.vision   → vision encoder layer probe AUC
probing.lm       → LM-layer probe AUC, logit lens trajectory
probing.steering → VTI direction 추출 + residual-stream 주입
```

## 핵심 설계 결정

1. **Generic `AutoModelForImageTextToText`** 사용; `Qwen2_5_VLForConditionalGeneration`
   하드코딩 X. `vlm_anchroing/src/vlm_anchor/models.py:39-55` 패턴 추종. 다음 라운드
   에서 LLaVA-1.5 나 InternVL2 로의 스왑은 config 변경만으로 가능.
2. **Config 는 YAML 이 아닌 Python 파일.** Configs 는 typed literal 의 dataclass 를
   매개변수화; YAML 은 type 정보를 잃거나 schema 가 필요. eval-sufficiency 의
   `EvalConfig(...)` 패턴 직접 채택.
3. **Streamed JSONL output.** `predictions.jsonl` 은 inference 단위로 flush 되어
   6시간 run 의 3시간 지점 crash 가 완료 row 를 잃지 않게. Parquet / CSV 는 마지막에
   한 번만 materialize.
4. **Activation capture 는 optional 이고 *프롬프트-무관***. 활성화되면 `open`
   프롬프트 + `labels[0]` 으로 capture forward pass 를 실행 — 어떤 inference 프롬
   프트가 physics-mode 를 trigger 했든 모든 stimuli 에 대해 비교 가능. MVP 규모에서는
   stimulus 당 ~7 MB (5 layer × bf16), 1k-stimulus run 에서 ~7 GB.
5. **PMR scoring 은 의도적으로 단순.** Stem-prefix matching + abstract-reject
   veto. False negative 는 예상되며 `lexicons.PHYSICS_VERB_STEMS` 를 확장하여
   잡는다 — `docs/scoring_rubric.md` 참조.
6. **CI 에 model-in-the-loop test 없음.** `tests/` 는 pure-Python stimulus
   determinism 과 PMR scoring 만 cover. VLM smoke 는 `scripts/02_run_inference.py
   --limit 5` 로 수동 실행.

## Commands cheat sheet

```bash
# 처음.
uv sync

# Stimuli 생성.
uv run python scripts/01_generate_stimuli.py --config configs/pilot.py

# Smoke inference (5 stimuli).
uv run python scripts/02_run_inference.py --config configs/pilot.py --limit 5

# 전체 pilot (H200 에서 ~30-60 분).
uv run python scripts/02_run_inference.py --config configs/pilot.py

# Score + summarize.
uv run python scripts/03_score_and_summarize.py --run-dir outputs/pilot_<ts>_<hash>

# Vision-encoder activation capture (M3).
uv run python scripts/04_capture_vision.py --stimulus-dir inputs/<run> --output-dir outputs/<run>/vision_activations --layers 3,7,11,15,19,23,27,31

# LM logit lens + per-layer probe (M4).
uv run python scripts/05_lm_probing.py --run-dir outputs/<run>

# VTI steering causal intervention (M5).
uv run python scripts/06_vti_steering.py --run-dir outputs/<run> --stimulus-dir inputs/<run> --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40

# Tests.
uv run python -m pytest
```

## 다음에 할 것 (`docs/next_steps.md` 와 `references/roadmap.md` §3 참조)

- M5b: SIP activation patching + SAE feature decomposition.
- M6: cross-model sweep (LLaVA-1.5 / LLaVA-Next / InternVL2).
- Photorealistic / 3D stimuli (axis A level 5, Blender).
- Human baseline (ROADMAP §4).
