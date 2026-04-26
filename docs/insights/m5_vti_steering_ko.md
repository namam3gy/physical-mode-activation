# M5 (ST4) Insights — VTI Causal Steering: "physical object-ness" Direction Found

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택 (ball → 동적 / circle → 정적 / planet → 궤도).
> - **H-boomerang** — Vision encoder 가 행동이 실패하는 곳에서도 physics-mode class 를 선형 분리 — encoder 는 알고 decoder 가 gate. (Qwen 한정: LLaVA-1.5 에서는 CLIP encoder 자체가 bottleneck 이라 반박.)
> - **H-locus** — Bottleneck 은 LM 중간 레이어 (특히 L10) 에 있음 — 더 이른 곳도, decoding head 도 아님.
> - **H-regime** — Steering 방향은 binary "object-ness" — 반박됨; H-direction-bidirectional 로 대체.
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M3** — ST2 vision-encoder probing — encoder AUC ≈ 1.0 으로 factorial 축 자명 분리 ("boomerang").
> - **M4** — ST3 LM logit lens / per-layer probes — LM AUC 가 시각-토큰 위치에서 L5 부터 ~0.95 plateau.
> - **M5** — ST4 인과 localization (VTI steering / activation patching / SAE) — M5a, M5a-ext, M5b 참조.
> - **M6** — ST5 cross-model sweep — M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next) 참조.
> - **v_L10** — M5a class-mean diff (physics − abstract) 에서 유도된 layer 10 LM hidden space (dim 3584) steering 방향. Unit norm.

Sub-task 4 의 첫 deliverable. M2 captured LM activations 에서 VTI 스타일
**physics-mode direction** 을 추출하고, test-time 에 LM residual stream 에
주입 (α · v) 했을 때 **Qwen2.5-VL-7B 가 선 원을 추상 도형으로 거부하던
기본 행동이 실제로 뒤집히는가** 를 검증.

원본 수치: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` ·
`steering_experiments/`.
구현: `src/physical_mode/probing/steering.py`, `scripts/06_vti_steering.py`.

## 1. 한 문장 요약

**Qwen2.5-VL-7B 의 LM layer 10 residual stream 에 `α=40 · v_L10` 을
주입하면 `line+blank+none` 자극에 대해 10/10 응답이 "D: abstract"에서
"B: stays still"로 flipping**한다. Layer 15/20/25 에서는 같은 α 로도
움직이지 않는다. 이는 **"physical object-ness" 선형 방향이 LM early-mid
residual stream 에 존재하며, intervention 에 causally 반응한다** 는 증거.

## 2. Phase 1 — Direction 추출

### 2.1 Recipe (연구계획 §2.5 VTI)

각 captured layer L 에 대해:

```
v_L = mean_sid (mean_token h_L[sid]  |  PMR(sid) = 1)
      − mean_sid (mean_token h_L[sid]  |  PMR(sid) = 0)
```

여기서 PMR 은 forced-choice target. 480 stimuli, n_pos=312, n_neg=168.

### 2.2 결과 — 방향이 layer 따라 증폭

| layer | ‖v_L‖₂ | n_pos | n_neg |
|---|---|---|---|
| 5 | 5.88 | 312 | 168 |
| 10 | 8.16 | 312 | 168 |
| 15 | 10.33 | 312 | 168 |
| 20 | 15.43 | 312 | 168 |
| 25 | **30.98** | 312 | 168 |

→ Norm 이 layer 따라 **5× 증폭** (5.9 → 31). M4 에서 본 "physics margin
amplification" 과 일치.

### 2.3 Projection 분포 (layer 20, forced-choice peak AUC)

각 sample 의 hidden state 평균을 `v_unit_L20` 에 투사:

**Object_level 별**:
| level | mean proj | std | min | max |
|---|---|---|---|---|
| line | 30.71 | 11.21 | 5.48 | 45.44 |
| filled | 32.14 | 10.79 | 7.63 | 46.23 |
| shaded | 34.26 | 10.39 | 10.37 | 47.41 |
| textured | 35.41 | 7.36 | 19.31 | 47.90 |

→ 작지만 monotone 증가 (+4.7 from line to textured).

**Cue_level 별**:
| cue | mean proj | std | min | max |
|---|---|---|---|---|
| none | **22.31** | 9.73 | 5.48 | 35.45 |
| cast_shadow | 29.68 | 6.68 | 18.40 | 38.80 |
| motion_arrow | 37.81 | 4.89 | 28.86 | 44.88 |
| both | **42.73** | 3.04 | 37.52 | 47.90 |

→ **20-unit spread** — projection 은 cue_level 을 깨끗하게 tracking.
"physics direction" 은 factorial 에서 가장 강한 축과 정렬됨.

## 3. Phase 2 — Causal Intervention

### 3.1 Setup

- **Test stimuli**: `line / blank / none`, 10 seeds × 1 object × 1 bg × 1 cue × 1 event = 10 stimuli.
- **Prompt**: forced-choice with label = `"circle"` (baseline PMR ≈ 0 → flipping 여지 최대).
- **Layers tested**: 10, 15, 20, 25.
- **α values**: 0, 5, 10, 20, 40.
- **Temperature**: 0 (determinisitic).
- **Intervention**: `model.model.language_model.layers[L]` 의 forward-hook 이 출력 hidden_states 에 `α · v_unit_L` 더함. 모든 token position 에 균일 적용.
- **총 inference**: 10 stimuli × 4 layers × 5 α = 200. ~5 분.

### 3.2 결과 — First-letter 응답 분포

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10/10 D | 10 D | 10 D | 10 D | **10 B** 🔥 |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

**Layer 10 α=40** 에서 10/10 이 D (abstract) → B (stays still) 로 **일제히 flip**.
다른 어떤 layer 도 α=40 으로 움직이지 않는다.

### 3.3 응답 text 의 비교 (sample stimulus `line_blank_none_fall_000`)

**Baseline (α=0, 모든 layer 공통)**:

> "D — This is an abstract shape and as such, it does not have physical
> properties that would allow it to fall, move, or change in any way. The
> image is simply a representation…"

**L10 α=40 (intervention)**:

> "B) It stays still. — Justification: The circle in the image appears to
> be floating or suspended in space without any external force acting upon
> it. In such a scenario, the circle would remain stationar[y]…"

**L20 α=40**:

> "D — The image depicts an abstract shape, specifically a circle, which
> is not a physical object and therefore cannot fall, move, or change
> direction in the way that objects do. The question is based on…"

**해석 차이**:
- Baseline / 후기 레이어: "abstract shape, not a physical object" — 추상 도형 거부 모드.
- L10 α=40: "the circle... floating or suspended in space... no external force" — **물리 객체로 인식하되 "정지 상태의 물체"**. 인터벤션이 "abstract → physical object" 의 categorical shift 를 일으킴.

### 3.4 PMR 점수 caveat

Forced-choice 응답의 raw_text 가 모든 옵션을 explicit 하게 언급하기 때문에
("cannot fall, move, or change direction"), physics-verb lexicon 이 옵션
텍스트에서 hits 를 검출 → PMR=1 을 반환. 즉 M5 intervention 에서 PMR=1
은 "모델이 physics-mode 로 갔다" 의 직접 증거가 아니라 **하위 신호**.

**진짜 causal evidence 는 첫 글자 (A/B/C/D) 분포** — 이것만 보면 D→B 의
flipping 이 딱 L10 α=40 에만 나타남. PMR scorer 에 향후 정교화 필요: **옵션
list 인용을 정상 physics 서술과 구분**.

## 4. Mechanism interpretation

### 4.1 왜 L10만 steerable?

M4 에서 본 probe AUC 는 **L20 peak (0.953)**, L10 은 0.944. 그런데 intervention
은 L10 에서 훨씬 유효. 가능한 설명:

1. **Upstream 개입이 downstream 까지 번진다**. L10 에서 biased hidden state 를
   주면 이후 layer 가 그 신호를 따라 residual updates 를 해서 최종 output 이 변함.
   L20 이후에 같은 vector 를 넣으면 **이미 commit 된 representation 위에 작은 nudge** 가 되어 decoding 결과에 영향 없음.
2. **L25 direction norm = 31** vs L10 = 8. 그러나 α · v_unit 으로 normalize 했으므로 실제 magnitude 는 layer 따라 같음. 즉 "effective strength" 는 layer 의 typical activation magnitude 에 대해 상대적이다. Late layer 에서는 α=40 · v_unit 이 layer norm 에 비해 "너무 작아서" 안 밀릴 수도 있음.
3. **L10 의 direction 이 semantic bottleneck** 역할 — "abstract vs physical" 결정이 이 근처에서 이루어지고, intervention 이 그 결정을 뒤집는다.

이 중 (1) 이 기존 causal interpretability 문헌 (Basu et al. 2024, "constraint
information stored in layers 1-4" in LLaVA; Neo et al. 2024) 과 일치.

### 4.2 Direction 은 "object-ness" 이지 "gravity" 가 아님

L10 α=40 응답이 **B: stays still** 이지 **A: falls down** 이 아님. 만약 steering
vector 가 "falls/drops/rolls" 같은 gravity 개념을 인코딩했다면 A 로 flip 해야
한다. 실제로는 B (stationary physical object) 로 flip.

→ **이 방향은 "abstract vs physical object" 의 binary 구분**이지, "어떤
physics 인가" (gravity / orbit / inertia) 를 선택하는 축이 아니다. H7 (라벨이
physics regime 을 선택) 과 일관: direction 은 coarse "object-ness" 이고,
구체적 physics 서술은 **라벨** 이 결정.

이는 향후 SAE 나 finer-grained probe 로 **direction 을 여러 세부 방향** 으로
decompose 하면 "gravity 방향" 과 "object-ness 방향" 을 분리할 가능성.

## 5. 가설 스코어카드 업데이트 (M5 이후)

| H | post-M4 | post-M5 | 변화 |
|---|---|---|---|
| H-boomerang | 확장 | **확장 + 인과 지지** | 정보 존재 (M3), LM 전구간 보존 (M4), 그리고 intervention 으로 causal 활성 (M5). |
| H-locus | 후보 | **지지 (early-mid)** | L10 이 causal sweet spot. 문헌의 early-layer intervention 결과와 일관. |
| H-regime (신규) | — | **후보** | Steering direction 은 "object-ness" binary 이지 "which physics" 가 아님. Physics regime 선택은 라벨-driven. |

## 6. Paper figure 후보

### Figure 6 — Causal Steering

```
A) baseline (line/blank/none × "circle" 프롬프트):
   10/10 responses → "D: abstract"
B) with α · v_L10 injection:
   α=0, 5, 10, 20: 10/10 → D
   α=40:            10/10 → B ("stays still")
C) α=40 at L15, L20, L25:
   10/10 → D (no effect)
```

메시지 요약: "layer 10 에 physics-mode direction 을 주입하면 abstract 거부를
overrides 할 수 있다. 이 direction 은 layer 15 이후에서는 causally inactive —
representation 이 이미 commit 됨."

Paper 의 "we can steer physics-mode" claim 을 뒷받침.

## 7. 한계 · 열린 질문

1. **Test subset 이 10 stimuli 로 작음**. 다른 추상-baseline 조건 (filled+blank+none,
   line+ground+none 등) 에서도 L10 α=40 으로 flip 되는지 확인 필요.
2. **Flipping이 "B" 로만 감**. 만약 direction 이 정말 "physics-mode" 라면 일부
   sample 은 A (falls) 로도 가야 할 것. 모두 B 인 이유: label="circle" 프롬프트가
   여전히 "motion" 해석을 억제. `label="ball"` 로 테스트하면 A/B 분포가 변할지 검증 필요.
3. **α=40 이 magic number**. α sweep 을 더 조밀하게 (30, 35, 45, 50, 60) 하면
   threshold 의 정확한 위치와 saturation 관찰 가능.
4. **Negative α** (abstract-ness direction) 테스트 안 함. `textured+ground+both`
   에 `-α · v` 넣으면 physics-mode 에서 abstract-mode 로 역방향 flip 할까?
5. **Attention knockout 과 activation patching 미실행**. 본 M5 는 VTI steering
   만 커버. 원 연구계획 §2.5 의 나머지 (Semantic Image Pairs + activation patching +
   SAE) 는 다음 라운드로 이월.

## 8. M5 가 unlock 하는 것

- **Paper 의 causal claim 이 방어 가능해짐**: correlation (M3, M4) → causation (M5) 의 사슬 완성.
- **§4.2 "reverse prompting" 아이디어**: 사진 공에 `"abstract shape"` 라벨 붙이기. 본 M5 의 negative-α counterpart.
- **SAE (연구계획 §2.5 stretch)**: L10 의 residual stream 에 SAE 훈련 → "physical object-ness" 방향을 더 세밀히 decompose 가능.
- **Cross-model (M6)**: 다른 open-source VLM (LLaVA-1.5, InternVL2) 에서도 같은 "L ≈ mid-early" sweet spot 존재하는지 검증. 일관적이면 general claim; 모델마다 다르면 architecture-specific finding.
