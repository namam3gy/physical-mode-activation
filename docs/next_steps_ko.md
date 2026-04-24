# 다음 단계 — 남은 sub-task 의 code-level 진입점

`references/roadmap.md` 의 동반 문서. ROADMAP 은 milestone-level view;
이 파일은 모듈별 plug-in 상세. 작업 진행에 따라 갱신.

여기 sub-task 중 다수는 이미 완료됨 (M3, M4, M5a). 아래 목록은 남은 것을 설명.

## Sub-task 4 Phase 3 — SIP + activation patching + SAE (M5b, 선택)

**목표** (`references/project.md` §2.5). M5a steering 발견을 "L10 에 direction
존재" 에서 "특정 layer × head 조합이 causally necessary" 로 발전. activation
patching 으로 physics-mode 행동을 회복/파괴하는 최소 컴포넌트 집합 식별.

**진입점**:
- `src/physical_mode/probing/steering.py` — patching helper 추가.
- 새 `src/physical_mode/probing/patching.py` — clean/corrupted forward replay.
- 새 `scripts/07_sip_patching.py`.

**필요한 것**:
1. M2 factorial 에서 Semantic Image Pair 구성. 각 pair 는 정확히 하나의 축에서
   다른 이미지 (예: 같은 seed 의 `shaded_ground_none` vs `line_ground_none`).
   `(clean_id, corrupted_id, differing_axis)` 컬럼의 `sip_manifest.parquet`
   emit.
2. SIP 서브셋에 대해 `capture_lm_attentions=True` 로 activation 재-capture
   (~120 stimuli × 각 pair 의 양 멤버). 추가 disk: ~15 GB.
3. **Activation patching**: clean 과 corrupted 를 forward, 그 후 corrupted 를
   replay 하면서 선택한 layer 의 visual-token activation 을 clean tensor 로 교체.
   PMR-positive token probability 의 변화 ("indirect effect") 측정. raw PyTorch
   hook 사용; nnsight 도 OK 지만 필수 아님.
4. **Attention knockout**: visual token → last position 의 특정 (head, layer)
   attention 을 0 으로. Test stimuli 의 PMR delta 측정.
5. **SAE** (stretch): Pach et al. 2025 따라 M3 vision-encoder activation 에 SAE
   훈련. monosemantic "cast_shadow", "ground", "shading" feature 식별. 개별
   SAE feature 를 clamp 해서 behavioral PMR shift 측정.

**성공 기준**:
- (layer, head) 한 범위의 attention knockout 이 SIP test set 의 PMR 을 ≥ 10 pp
  drop.
- 명시적 steering vector 없이 SAE feature ablation 이 M5a 의 L10 D→B flip 을
  재현하는 것.

## Sub-task 5 — Cross-model sweep (M6)

**목표** (`references/project.md` §2.6). LLaVA-1.5-7B, LLaVA-Next-7B,
InternVL2-8B, (stretch) Qwen2-VL-7B 에서 M1-M5 재현. boomerang pattern (M3),
LM late-layer peak (M4), L10 steering sweet spot (M5a) 가 universal 인지 Qwen
specific 인지 검증.

**진입점**:
- `configs/cross_model.py` (새).
- `scripts/02_run_inference.py` — model 리스트 iterate 하도록 확장.

**필요한 것**:
1. `EvalConfig` 에 `system_prompt_override: str | None` 필드 추가 — Gavrikov
   et al. 2024 스타일 prompt steering 용 ("treat this as an abstract geometric
   shape" vs "treat this as a physical object subject to gravity").
2. model_id 리스트 iterate; output 은 `outputs/cross_model_<model>_<ts>/`.
3. 각 모델에서 ST3 축약 (LM probe at ~5 layer) + ST4 (L10 의 analog 에서 M5a
   steering) 실행.
4. Disk 예산: 모델 다운로드 총 ~60 GB (LLaVA-1.5-7B ~13 GB, LLaVA-Next-7B ~14 GB,
   InternVL2-8B ~16 GB, Qwen2-VL-7B ~15 GB).

**성공 기준**:
- `H-boomerang` 이 추가 2 개 이상 모델에서 확인 (encoder AUC 높음, behavioral
  PMR 가변).
- "L10 sweet spot" 재현, 또는 per-model sweet-spot index 가 LM depth 의 30-40 %
  근처 cluster.

## 자극 확장 (모든 라운드)

- **Photorealistic axis A level 5**: `src/physical_mode/stimuli/diffusion.py`
  추가 — `vlm_anchroing/scripts/generate_irrelevant_number_images.py` 패턴으로
  FLUX.1-schnell 호출해서 `textured_photo` variant 렌더. CLIP-similarity
  threshold 으로 "a ball" vs "a circle drawing" 필터. M3 encoder AUC = 1.0
  finding 이 프로그램 자극 ceiling 의 artifact 인지 일반화하는지 검증에 critical.
- **Blender 3D**: 정확한 광원 방향의 통제된 sphere 렌더 — shape-from-shading
  실험용. 덜 urgent; reviewer 가 요구하면.
- **Block-stack 별도 경로** (`references/roadmap.md` §4.1): MVP-full 의
  `object_levels` 에 `block_stack` 추가. "물리 객체 ≠ ball" 축 테스트.

## Behavioral / methodological 확장

- **Label-free prompt** (`references/roadmap.md` §4.9): `prompts.py` 의
  `open_no_label` variant — `{label}` slot 없이 "What do you see? What might
  happen next?" 질문. H2 prior 비활성화 + M4 switching-layer metric 에 의미
  부여 기회.
- **Reverse prompting** (`references/roadmap.md` §4.2): textured-ground-both
  자극에 `"abstract diagram"` 라벨 부착. H4 의 counterfactual.
- **Per-label regime annotation** (`references/roadmap.md` §4.11): open-ended
  응답을 5 범주 (gravity-fall / gravity-roll / orbital / inertial / static)
  로 zero-shot classify. H7 의 정량적 검증.

## 인간 baseline (선택, M7)

MVP-full set 에서 50 stimuli 샘플링, stimulus 당 20 Prolific 응답을 같은
open-ended 프롬프트로 수집, human PMR 계산. human-vs-VLM alignment 를
secondary headline figure 로 보고. paper 의 "ambitious" 버전 (NeurIPS 스코프)
에만 필수.
