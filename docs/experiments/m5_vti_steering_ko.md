# M5 — VTI steering (Phase 1-2, partial Sub-task 4) (2026-04-24)

- **Phase 1 명령**: inline Python — `compute_steering_vectors` from `src/physical_mode/probing/steering.py`
- **Phase 2 명령**: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`
- **Wall clock**: ~5 분 (200 interventional inference)
- **출력**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` + `steering_experiments/`
- **심층**: `docs/insights/m5_vti_steering_ko.md`.

## 헤드라인: "physical object-ness" direction 의 causal evidence

**Phase 1 (direction)**: forced-choice PMR 라벨에서 derived 된 VTI vector.
Norm 이 LM 통해 5× 성장 (L5: 5.9 → L25: 31). L20 의 projection 은 cue axis 를
깨끗하게 tracking (none 22.3 → both 42.7). Direction 은 real 하고 factorial
aligned.

**Phase 2 (intervention) — `line/blank/none`, label `circle`, α=40 의 first-letter 분포**:

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10 D | 10 D | 10 D | 10 D | **10 B** 🔥 |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

**Layer 10 α=40 가 10/10 을 "D: abstract" 에서 "B: stays still" 로 flip**.
다른 layer 는 α=40 에서도 안 움직임. LM residual stream 의 단일 direction 이
"physical object" vs "abstract shape" 결정을 gate 한다는 첫 causal 확인.

Pre/post 샘플:
- **baseline**: "D — This is an abstract shape and as such, it does not have physical properties that would allow it to fall, move, or change…"
- **L10 α=40**: "B) It stays still. — The circle in the image appears to be floating or suspended in space without any external force acting upon it. In such a scenario, the circle would remain stationary…"

Direction 은 "object-ness" (B = physical-static 으로 flip), "gravity" (A =
falls 로 flip 해야) 가 **아님**. H7 와 일관: physics regime 은 label-driven,
direction 은 binary "abstract vs physical" split 을 인코딩.

## Caveat

- Forced-choice PMR scorer 가 option-listing ("cannot fall, move, or change
  direction") 에서 fire → PMR=1 이 noisy. First-letter flipping 이 clean
  causal signal.
- Test subset = 10 stimuli 만; filled/blank/none 등에서의 replication pending.
- α=40 만 작동했음; threshold 를 정확히 매핑하려면 finer sweep 필요.

## 가설 update

| H | 이전 | post-M5 | 변화 |
|---|---|---|---|
| H-boomerang | 확장 (M4) | **확장 + causal** | intervention 으로 causal 활성 |
| H-locus | 후보 (M4) | **지지 (early-mid L10)** | L10 이 causal sweet spot |
| H-regime (new) | — | **후보** | direction 은 object-ness 이지 which-physics 아님 |

## Unlock / deferred

- Phase 3 (SIP activation patching) 은 scope 잡았으나 실행 안 됨 —
  `capture_lm_attentions=True` 로 재-capture + patching machinery 필요.
  ROADMAP §3 M5b detail 에 있음.
- L10 residual stream 의 SAE 가 natural next step (additional idea;
  object-ness direction 을 finer feature 로 decompose).
