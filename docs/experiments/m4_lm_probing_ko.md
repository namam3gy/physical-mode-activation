# M4 — LM probing (logit lens + per-layer probe) (2026-04-24)

- **명령**: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`
- **입력**: M2 가 캡처한 layer (5, 10, 15, 20, 25) 의 LM hidden state, 480 stimuli
- **Wall clock**: ~6 분 (logit-lens 480 × 5 layer = 2400 projection + model load)
- **출력**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/`
- **심층**: `docs/insights/m4_logit_lens_ko.md`.

## 핵심 finding

**LM per-layer probe AUC (target: forced-choice PMR)**:

| layer | AUC (mean ± std) |
|---|---|
| 5 | 0.939 ± 0.015 |
| 10 | 0.944 ± 0.006 |
| 15 | 0.947 ± 0.009 |
| **20** | **0.953 ± 0.007** (peak) |
| 25 | 0.944 ± 0.009 |

→ 모든 captured LM layer 가 PMR 을 AUC ~0.94-0.95 로 분리. Layer 20 peak 는
Neo et al. 2024 의 LLaVA-1.5 claim (object features 가 mid-to-late LM layer 에
crystallize) 과 일치. **정보가 LM 을 거의 손실 없이 통과**; behavioral
forced-choice accuracy (~0.66) 와 ~29 pp gap 은 discrete generation step 에서
발생할 수밖에 없음.

**Logit-lens trajectory (category 별 mean logit)**:

| layer | geometry | physics | label |
|---|---|---|---|
| 5 | 0.93 | 1.04 | 1.16 |
| 10 | 1.35 | 1.66 | 1.73 |
| 15 | 2.04 | 2.45 | 2.29 |
| 20 | 3.23 | 4.18 | 4.09 |
| 25 | 11.56 | **15.64** | 13.96 |

→ L5 부터 physics > geometry — 프롬프트의 "ball" 라벨이 *어떤 residual update
도 가기 전* LM 을 prime. Final-layer amplification (L20 → L25) 이 physics
margin 을 0.9 → 4.0 으로.

**object_level 별 physics margin** (phys − geom logit):

| layer | line | filled | shaded | textured |
|---|---|---|---|---|
| 5 | 0.09 | 0.08 | 0.12 | 0.15 |
| 20 | 0.87 | 0.89 | 0.97 | 1.05 |
| 25 | 3.76 | 3.94 | 4.29 | 4.35 |

Object 유발 margin: L25 에서 +0.6 (line → textured). **Label 유발 margin: +4.0
(모든 stimuli 에 flat bias)**. LM 내에서 label ≈ 7× object 효과. H7/H4 와 일관.

**Switching layer** 는 max-logit-per-category 사용 시 480 샘플 모두에 대해
trivially 5 — "ball" 라벨 때문에 가장 이른 captured layer 에서 이미 physics
가 앞섬. Label-primed 프롬프트에서 이 metric 은 informative 하지 않음;
label-free 프롬프트 variant 로 재검토 필요.

## 가설 update

| H | 이전 | post-M4 | 변화 |
|---|---|---|---|
| H-boomerang | 지지 (M3) | **확장** | 정보가 LM 전체를 통과해 살아남음 — 게이팅이 decoding 에서 |
| H7 (label → regime) | 후보 (M2) | **지지** | Label prior 가 L5 부터 physics margin 을 shift |
| H-locus (new) | — | **후보** | 병목은 LM final layer + decoding head. ST4 patching target. |

## Unlock

- **Sub-task 4 (M5) target 좁혀짐**: LM layer 20-27 residual stream + decoding
  head 가 intervention priority. `capture_lm_attentions=True` 로 재-capture
  하면 SIP patching 즉시 runnable.
- **§4 아이디어 update**: 4.9 "label-free prompt" 는 switching-layer metric
  을 다시 valid 하게 만들 직접 test — priority 상승.
