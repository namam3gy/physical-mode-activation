---
section: §4.8 — PMR scaling (Qwen2.5-VL 7B vs 32B on M2)
date: 2026-04-28
status: complete (open prompt, 1440 inference 각, H200 에서 16분 wall)
hypothesis: 스케일은 PMR 도움 안됨 — MechBench 식 "스케일링이 grounding 못 고침" 예측.
---

# §4.8 — PMR scaling: Qwen2.5-VL 7B vs 32B (M2)

## TL;DR

Qwen 32B는 open prompt 의 M2 stim 에서 Qwen 7B 와 **사실상 동일한
aggregate PMR**:

| Metric | 7B (open) | 32B (open) |
|--------|----------:|-----------:|
| **PMR (mean)** | **0.931** | **0.926** |
| abstract_reject | 0.002 | **0.065** |
| GAR | — | 0.631 |
| Hold-still | 0.152 | 0.103 |
| n | 1440 | 1440 |

Headline ΔPMR = −0.005, 노이즈 안. **5× 스케일링이 physics-mode 분류를
도와주지 않음** — MechBench 의 "더 많은 파라미터 만으로는 M2 의 'circle
/ ball / planet' 프롬프트 체제의 언어-prior 지배력을 못 고침" 가설 지지.

`abstract_reject` 35× 증가 (0.002 → 0.065) 가 유일한 의미있는 변화: 32B
는 시각 cue 가 약할 때 physics-mode framing 을 거부할 가능성이 더 높음.
이는 작지만 실재하는 per-cell discrimination 향상이지만, cue-strong
cell 이 이미 포화되어 있어서 전체 PMR 로는 translate 안됨.

## Per-axis breakdown (axis 레벨별 PMR)

### Object-level (추상화 축)
| object_level | 7B | 32B | Δ |
|--------------|----:|----:|--:|
| line         | 0.906 | 0.911 | +0.005 |
| filled       | 0.933 | 0.919 | −0.014 |
| shaded       | 0.933 | 0.922 | −0.011 |
| textured     | 0.950 | 0.950 |  0.000 |

두 모델 모두 포화; 이 프롬프트 체제에서 H1 abstraction-axis ramp 안
보임 (M2 7B 에서 이미 알려짐). 스케일도 ramp 못 노출.

### Cue-level (물리-cue 축) — 32B 가 다른 유일한 곳
| cue_level | 7B | 32B | Δ |
|-----------|----:|----:|--:|
| none         | 0.797 | **0.711** | **−0.086** |
| cast_shadow  | 0.936 | 0.994 | +0.058 |
| motion_arrow | 0.997 | 1.000 | +0.003 |
| both         | 0.992 | 0.997 | +0.005 |

`none` cell 8.6 pp 하락, `cast_shadow` 5.8 pp 상승. **32B 는 cue-
sensitive**: cue 없을 때 abstract 쪽으로 물러나고, 단일 cue 가 fire
하면 physics-mode 에 더 강하게 commit. 시각-prior 의존을 약간 개선,
그러나 프로젝트가 documenting 한 언어-prior 지배를 dissolve 하진 않음.

### Label (언어-prior 축)
| label | 7B | 32B | Δ |
|-------|----:|----:|--:|
| ball   | 0.954 | 0.933 | −0.021 |
| circle | 0.883 | **0.923** | **+0.040** |
| planet | 0.954 | 0.921 | −0.033 |

32B 에서 ball/planet 약간 하락, circle 상승 — **label gap 좁아짐**
(7B `ball − circle` = +0.071 pp; 32B = +0.010 pp). H2 언어-prior 지배가
스케일링 하에 *약화* 되지만 제거되지 않음. 그래도 cue-level 과 함께
가장 주목할 per-axis shift.

## Open questions

- **왜 스케일이 cue-level 은 도와주지만 object-level 은 안 도와주나?**
  Cue 축은 single-feature visible (그림자 blob, 화살표); object 축은
  대체로 stylistic shading. 가능성: 32B 가 discrete physics cue 를
  *통합* 할 capacity 가 더 있지만 smooth abstraction 차이에는 동일
  포화.
- **abstract_reject 점프 (0.002 → 0.065) 가 cue=none cell 에 집중?**
  per-cue PMR 하락과 매치되니 yes 가능성 큼. Cell-level pivot 이
  확인.
- **72B 는?** 미테스트 — scaling curve 확장. 32B 근처 (이 체제 포화)
  예측.

## 한계

1. Open prompt 만 — forced-choice 32B 미실행.
2. T=0.7 → RC<1 cell 존재; per-seed agreement 가 아닌 mean 보고. RC
   분석은 M2 와 동일한 형태일 것.
3. M5a steering 32B 미실행 (새 layer-aware capture 필요).
4. L10 v_L 추출 미실행 → §4.6 픽셀-인코드 가능성 cross-scale 미테스트.

## 재현

```bash
# 추론 (H200 의 single GPU bf16 — 64 GB weights + KV 들어감).
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m2_qwen_32b.py \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3
# H200 에서 1440 inference, ~16분 wall (max_new_tokens=96).

# 채점 + 요약.
uv run python scripts/03_score_and_summarize.py \
    --run-dir outputs/m2_qwen_32b_<ts>
```

## Artifacts

- `configs/m2_qwen_32b.py` — Qwen 32B M2 open-prompt config.
- `outputs/m2_qwen_32b_20260427-212653_a167494f/{predictions,
  predictions_scored,response_consistency,summary_overall,
  summary_by_*}.csv` — 전체 1440 row 추론 + 채점 + factorial 요약.
