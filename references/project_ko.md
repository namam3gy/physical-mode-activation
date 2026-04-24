# VLM의 "물리 모드" 전환 트리거 연구 기획: 관련 연구 조사와 실험 설계

사용자의 연구 질문 — **추상적 도형(원)이 어떤 시각적 조건에서 VLM에게 물리적 객체(공)로 처리되는가** — 는 현재 문헌에서 세 가지 희소 영역(추상–구체 축 조작, open-source VLM의 다음 상태 예측, 물리 트리거의 기계적 분석)의 교집합에 놓여 있으며, **직접적 선행 연구가 존재하지 않는다**. Pixels-to-Principles (Ballout et al., 2025)가 동일한 3개 open-source VLM 군(InternVL 2.5, Qwen2.5-VL, LLaVA-OneVision)에 대해 intuitive physics 표상을 probing하지만 사진 사실적 자극만 사용하며 추상화 수준을 조작하지 않는다. 본 보고서는 (1) 세 하위 영역의 문헌을 정리하고, (2) NeurIPS/EMNLP 수준 논문으로 발전시킬 구체적 하위 과제를 제시하며, (3) 포지셔닝과 위험 요소를 논한다.

---

## Part 1. 관련 연구 정리

### 1.1 VLM 물리 추론 벤치마크 (영역 a)

문헌은 **추상 2D 도형 물리(pre-VLM 시대)**와 **사진 사실적 3D/video 물리(현재 VLM 시대)** 두 극단으로 양분되어 있으며, **동일 물리 시나리오에서 추상화 수준을 매개변수적으로 변화시킨 연구는 없다**.

**최신 VLM 물리 벤치마크 (2023–2026)**

| 논문 | 연도/학회 | 자극 양식 | 과제 유형 | 평가 VLM | 핵심 발견 |
|---|---|---|---|---|---|
| **PhysBench** (Chow et al.) | ICLR 2025, arXiv:2501.16411 | 사진/합성 video+image 10,002항 | 4영역 QA (property/relation/scene/**dynamics**) | 75개 VLM (GPT-4o, Gemini, LLaVA, Qwen-VL, InternVL 전 계열) | VLM은 commonsense에 강하나 **dynamics에서 실패**; 오류는 perception과 knowledge 둘 다에 분포 |
| **Pixels to Principles** (Ballout, Jassim, Bruni) | arXiv:2507.16572, 2025 | GRASP+IntPhys2 video | plausibility 판단 + **중간 임베딩 probing** | **InternVL2.5, Qwen2.5-VL, LLaVA-OneVision**, Gemini | **Vision encoder는 물리 cue를 포착하지만 LLM이 활용하지 못함**; vision-language misalignment가 병목. t-SNE 군집이 projection 후 붕괴 |
| **VLM4D** (Zhou et al.) | ICCV 2025, arXiv:2508.02095 | 실사+합성 video | 병진/회전/연속성 MCQ | GPT-4o 62% vs 인간 98.8%; open-source는 더 큰 격차 | 움직임 추론에서 VLM의 거대한 공백 |
| **MechBench / "Probing Mechanical Reasoning in Large VLMs"** (Zhang et al.) | arXiv:2410.00318, 2024 | 기계 시스템 정적 이미지 (155개 실험) | 안정성·지레·관성·유체 QA | 26개 VLM (모든 대표 open-source 포함) | **규모 증가가 도움 안 됨** — 아키텍처적 한계 시사 |
| **PhysGame** (Cao et al.) | arXiv:2412.01800, 2024 | 게임 플레이 video (글리치) | 물리 위반 탐지 MCQ | LLaVA-Next-Video, Qwen2-VL, InternVL-Video 등 | Open-source가 closed-source에 크게 뒤짐 |
| **GRASP** (Jassim et al.) | IJCAI 2024, arXiv:2311.09048 | Unity 합성 video | 영속성·연속성·중력 plausibility | Video-LLaMA, VideoChat, Video-ChatGPT, LLaVA, Qwen-VL | Video-MLLM 전반 근-무작위 |
| **IntPhys 2** (Bordes et al., FAIR) | arXiv:2506.09849, 2025 | 복잡한 합성 video | VoE (영속성·불변·시공간 연속·solidity) | Frontier MLLM | 대부분 ~50% 수준 |

**고전 (abstract-shape) 벤치마크 (pre-VLM)**

Bakhtin et al. 2019의 **PHYRE**(NeurIPS), Ates et al. 2022의 **CRAFT**(Findings of ACL), Patel et al. 2022의 **CRIPP-VQA**(EMNLP), Yi et al. 2020의 **CLEVRER**(ICLR)는 모두 Box2D/CLEVR 스타일 추상 2D 도형으로 물리 이벤트를 구성했으나 **현대 VLM에 체계적으로 재평가되지 않았다**.

**정적 객체-속성 VLM 연구**: Gao et al. 2023의 **PhysObjects / PG-InstructBLIP**(arXiv:2309.02561)는 가정용품에 material·fragility·mass 주석을 달았으나 **정적 속성 QA**에 국한되며 다음 상태 예측이 아니다. Liu et al. 2024의 **PhysGen**(ECCV)은 이미지→video 생성에서 VLM이 정적 이미지로부터 물리 파라미터를 *추출 가능함*을 보여주지만 언제 이러한 모드가 활성화/비활성화되는지는 다루지 않는다.

### 1.2 추상 vs 구체 시각 인식 (영역 b)

VLM은 **순수 기하학적 도형에서 체계적으로 실패**하며, 이 실패는 reasoning 단계가 아닌 **perception 단계**에서 발생함이 여러 독립적 연구에서 수렴한다.

- **"Vision Language Models Are Blind"** (Rahmanzadehgervi, Bolton, Taesiri, Nguyen, ACCV 2024 Oral, arXiv:2407.06581) — BlindTest 7개 과제(원 겹침, 선 교차, Olympic 링 세기). GPT-4o/Gemini-1.5-Pro/Claude-3.5 평균 58%. 결정적으로 **인코더 probing은 정보가 존재함을 보여줌** — 이는 "encoder knows, decoder doesn't" 해리. 본 연구의 직접적 템플릿.
- **"Vision Language Models Are Biased"** (Vo et al., ICLR 2026 accepted, arXiv:2505.23941) — 반사실적 counting(4줄 Adidas, 5-leg 개). 정규 인스턴스 ~100% vs 반사실 ~17%. **VLM은 픽셀이 아닌 기억된 텍스트 prior에 투사함**. 본 연구의 "원에 ball 특성 투사" 가설의 최강 유사 선례.
- **MARVEL** (Jiang et al., NeurIPS 2024, arXiv:2404.13591) 및 **"Curious Case of Nonverbal Abstract Reasoning with MLLMs"** (Ahrabian et al., COLM 2024) — Raven's PM/추상 도형. MLLM의 큰 격차, perception이 reasoning보다 우세한 병목.
- **Bongard 시리즈**: Bongard-LOGO (NeurIPS 2020), Bongard-HOI (CVPR 2022), **Bongard-OpenWorld** (ICLR 2024, arXiv:2310.10207) — VLM 64% vs 인간 91%, perception이 주요 병목.
- **Sketch-photo 도메인 격차**: Sain et al. 2023 (CVPR), Efthymiadis et al. 2024 (ECCV)는 CLIP 계열이 스케치를 체계적으로 과소 표상함을 정량화. **CLIPasso** (Vinker et al., SIGGRAPH 2022)는 CLIP이 가이드 받으면 추상 스케치에 의미를 부여할 수 있음을 보여줌 — 양방향 매핑 존재.
- 기타: **IconQA** (NeurIPS D&B 2021), **MathVista** (ICLR 2024), **MMMU** (CVPR 2024), **MultiStAR** (arXiv:2505.21850)는 모두 추상 다이어그램에서 VLM의 일관된 약점을 보고.

### 1.3 인간 직관 물리학 — 인지과학 토대 (영역 c)

**Spelke의 core knowledge** (Spelke 1990, *Cog Sci*; Spelke & Kinzler 2007, *Dev Sci*): 영아는 cohesion/boundedness/rigidity/no action at a distance/continuity 원리로 객체를 파싱. **Kellman & Spelke (1983, *Cog Psych*)**: 폐쇄나 good form보다 **common motion**이 객체 통일성의 주 단서.

**Violation-of-Expectation**: Baillargeon, Spelke & Wasserman (1985, *Cognition*) drawbridge 패러다임; Margoni, Surian & Baillargeon (2024, *Psych Review*) 현대 리뷰.

**Michotte의 launching effect** (1946/1963, *Perception of Causality*): 접촉 + 정확한 시공간 contingency가 두 사각형 간 인과 지각을 즉시 유발. Leslie & Keeble (1987, *Cognition*) 6개월 영아에서도 존재. Bechlivanidis et al. (2025, *Royal Soc Open Sci*) 등록된 복제.

**Heider-Simmel animacy** (1944, *Am J Psych*): 삼각형과 원의 단순 운동이 자발적으로 agent로 지각됨. **Scholl & Tremoulet (2000, *TiCS*)**: 적절한 kinematics(자발적 운동, 관성 위반, 목표 지향적 contingency)가 빠르고 자동적이며 저항 불가능한 animacy/causality 지각을 유발.

**3D 및 객체 단서**: **Ramachandran (1988, *Nature*)** shape-from-shading + light-from-above prior로 즉각적 3D 볼록 구 지각 (180° 회전 시 오목으로 전환); **Kersten, Mamassian & Knill (1997, *Perception*)** 드리워진 그림자가 객체를 지면에 부착하고 중력 프레임 부여; **Gibson (1979)** 생태학적 접근 — 표면·지면·텍스처 기울기·optic flow가 핵심 물리 단서.

**직관 물리학 = 시뮬레이션**: **Battaglia, Hamrick & Tenenbaum (2013, *PNAS*)** "Intuitive Physics Engine" — 인간은 근사적 확률적 물리 시뮬레이터 사용. **Sanborn, Mansinghka & Griffiths (2013, *Psych Review*)** "noisy Newton". **Ullman et al. (2017, *TiCS*)** "game engine in the head". **Smith, Hamrick et al. (2024, MIT Press)** 최신 정통 리뷰 "Intuitive physics as probabilistic inference".

**AI와의 다리**: Smith et al. (2019, NeurIPS) **ADEPT** — 심층 인식 + 확률적 물리 + 입자 필터. **Piloto et al. (2022, *Nature Hum Behav*)** **PLATO** — 객체 중심 표상이 VoE 효과에 필수적임을 입증. **Yildirim et al. (2024, *Nature Hum Behav*)** — 3D shape 지각이 직관 물리와 통합됨.

**합성된 자극 강도 위계 (정적 이미지 기준, 약한 → 강한)**: 폐쇄된 윤곽 (기준선) → 선화 스타일 현실 단서 → occlusion → 원근/지면 → 사진적 재질·텍스처 → 드리운 그림자 + 지면 접촉 → **3D shape-from-shading** (가장 강력한 정적 단서). 동적 자극에서는 Michotte 접촉과 중력적 탄도가 3D shading을 능가할 가능성.

### 1.4 VLM 기계적 해석가능성 (영역 d)

**시각 인코더 분해**: **Gandelsman, Efros & Steinhardt (ICLR 2024 Oral)** "Interpreting CLIP's Image Representation via Text-Based Decomposition" — CLIP-ViT 출력을 patches × layers × heads로 분해, TextSpan으로 각 head의 부공간을 자동 레이블링(shape/color/counting/location 특화 head 발견). **Balasubramanian et al. (NeurIPS 2024)**가 DINO/DeiT로 확장.

**Sparse Autoencoder**: **Pach, Karthik et al. (NeurIPS 2025, arXiv:2504.02821)** "Sparse Autoencoders Learn Monosemantic Features in VLMs" — CLIP 활성화에 SAE 훈련, **LLM을 건드리지 않고도 CLIP feature 개입으로 LLaVA 출력을 인과적으로 조작**.

**LLaVA 내부 시각 토큰 처리**: **Neo, Ong, Torr, Geva, Krueger, Barez (arXiv:2410.07149, ICLR 2025)** "Towards Interpreting Visual Information Processing in VLMs" — 시각 토큰 위치에 logit lens + attention knockout. 발견: (1) 객체 특화 토큰이 공간적으로 지역화; 제거 시 객체 ID 정확도 70% 이상 하락; (2) 층 1-10은 광범위 맥락 처리, **층 15-24 (LLaVA-1.5 32층 중)에서 특정 객체 feature 추출**; Qwen2-VL-2B에서는 peak 층 ~25/29. **본 연구의 핵심 방법론 청사진**.

**인과 추적**: **Basu, Grayson et al. (NeurIPS 2024, arXiv:2406.04236)** "Understanding Information Storage and Transfer in MLLMs" — MultimodalCausalTrace. **LLaVA에서 제약 만족 정보는 층 1-4의 초기 MLP/self-attention에 저장**(텍스트 LLM의 중간층 MLP와 대조). 소수 시각 토큰이 이미지 정보 전달 책임.

**Symmetric Image Pairs**: **Golovanevsky, Rudman, Palit, Singh, Eickhoff (NAACL 2025, arXiv:2406.16320)** "What Do VLMs NOTICE?" — Gaussian noise의 환각적 결과(Zhang & Nanda 2024) 회피를 위한 **Semantic Image Pairs** 도입. **LLaVA self-attention은 image-grounding head가 없고 outlier suppression만** 존재 (BLIP의 cross-attention과 대조) — 아키텍처적 중요 발견.

**층별 정보 흐름**: **Kaduri, Bagon, Dekel (arXiv:2411.17491)** "What's in the Image?" — cross-modal 흐름은 **중간 ~25% 층에 집중**. **Jiang et al. (2024)** "2단계 과정": visual enrichment → semantic refinement.

**Hidden state steering**: **Liu, Ye, Zou (ICLR 2025, arXiv:2410.15778)** VTI — (hallucinated, grounded) 쌍으로부터 per-layer shift vector 계산, test-time 추가. **Li et al. (ICML 2025, arXiv:2502.03628)** VISTA — "Hidden Life of Tokens": 점진적 시각 정보 손실, early excitation, hidden genuine tokens 발견.

**인코더-디코더 해리**: **Zhang, Unell et al. (NeurIPS 2024, arXiv:2405.18415)** "Why are VLMs Bad at Image Classification?" — 분류 정보가 LLaVA 잠재 공간에 존재하나 **LM이 활용하지 않음**. **Tong et al. (CVPR 2024)** "Eyes Wide Shut" — CLIP-blind pair와 MMVP 벤치마크, MoF (CLIP+DINOv2 interleave) 제안.

**시각 토큰 중복성**: **Chen et al. (ECCV 2024 Oral)** FastV — "층 2 이후 이미지는 절반 토큰 가치"; 깊은 층의 시각 토큰 attention은 극도로 비효율. **SparseVLM** (ICML 2025), LLaVA-PruMerge 등이 후속.

**메타 리뷰**: **Yiming Liu, Zhang & Yeung-Levy (ICLR 2025 Blogpost Track)** "Mechanistic Interpretability Meets VLMs" — 현재 도구들이 물리 개념에 거의 적용되지 않았음을 기록.

### 1.5 통제된 자극 연구 (영역 e)

최소 쌍 접근의 대표작들:
- **What'sUp** (Kamath, Hessel, Chang, EMNLP 2023, arXiv:2310.19785) — on/under/left/right만 변경, 객체 고정. BLIP-VQAv2 56% vs 인간 99%.
- **Winoground** (Thrush et al., CVPR 2022) — 같은 단어 다른 순서 caption 쌍 + 두 이미지. 모든 VLM이 group score에서 chance 이하.
- **VALSE** (Parcalabescu et al., ACL 2022) — foil 기반, 6개 언어 현상.
- **CLEVR** (Johnson et al., CVPR 2017) — Blender 합성; color/shape/material/size/count 통제. CLEVR-Hyp, Super-CLEVR, CLEVRSkills(NeurIPS 2024)로 확장.
- **WHOOPS!** (Bitton-Guetta et al., ICCV 2023) — commonsense 위반 Midjourney 이미지. BLIP2-XXL explanation 27% vs 인간 95%.
- **HallusionBench** (Guan et al., CVPR 2024) — 시각 착시 및 반사실적 편집 포함.
- **NaturalBench** (Li et al., NeurIPS 2024 D&B) — 10k 짝지어진 VQA, 교대되는 답을 강제하여 vision 사용 강제.

### 1.6 물리 트리거 시각 단서 (영역 f)

- **Geirhos et al. (ICLR 2019 Oral)** "ImageNet-trained CNNs are biased towards texture" — cue-conflict 방법론의 출발점.
- **Gavrikov, Lin, Bethge, Keuper (arXiv:2403.09193, 2024)** "Are VLMs Texture or Shape Biased and Can We Steer Them?" — **VLM이 vision backbone보다 더 shape-biased**; 멀티모달 융합이 cue 선호를 변경; 언어 프롬프트로 bias steering 가능하나 인간 96% shape bias에 도달 못함. **본 연구의 직접 선례**.
- **"Shape and Texture Recognition in Large VLMs"** (arXiv:2503.23062, 2025) — LVLM은 semantic feature에 크게 의존; 클래스 연관 없는 추상 2D 도형에서 실패.
- **Garrido et al. (arXiv:2502.11831, 2025)** **V-JEPA intuitive physics** — 표상 공간 예측 목적함수가 직관 물리를 창발시킴 (IntPhys 98%); **픽셀 예측과 MLLM은 거의 무작위**. 현재 "무엇이 물리 모드를 유발하는가"에 대한 최강의 아키텍처 수준 증거.
- **Bi, Yamins, Fan et al. (*Nature Comms*, 2025)** — DNN feature space가 부드러운 물체 물리적 판단을 인간과 일치하게 인코딩.
- **Peters & Kriegeskorte (*Nature Hum Behav*, 2021)** — 객체 기반 표상 리뷰; DNN이 grouping/amodal completion 결핍.
- **"Pixels to Principles"** (Ballout et al., 2025) — 이미 언급했듯 3개 타겟 open-source VLM에서 vision-language misalignment 병목 입증, **추상화 수준 조작은 하지 않음**.

### 1.7 공백 분석 요약

본 연구는 **세 가지 희소 영역의 교집합**에 위치:
1. **(추상화 조작 × VLM)** — Geirhos식 cue-conflict가 있으나 **물리**가 아닌 분류 과제에만.
2. **(다음 상태 예측 × open-source VLM)** — PhysBench/VLM4D/GRASP는 사진 사실적 자극만; Pixels-to-Principles는 probing하나 plausibility 판단만.
3. **(기계적/probing × 물리)** — 사실상 Pixels-to-Principles만; 풍부한 방법론 도구(NOTICE의 SIP, SAE, activation patching, logit lens)가 존재하나 물리-객체 트리거에 적용된 바 없음.

동일 물리 이벤트(예: 낙하)를 **선화 → 음영 → 사진적**으로 매개변수적으로 변화시키며 VLM의 물리 모드 "임계값"을 식별하고 이를 mechanism 수준에서 localize한 연구는 **전무하다**.

---

## Part 2. 연구 개발 및 하위 과제

### 2.1 전체 페이퍼 내러티브

**중심 주장**: Open-source VLM은 시각 입력에 대해 두 가지 모드 사이를 스위칭한다 — (M1) 추상 기하학적 형태 인식, (M2) 물리적 객체 추론. 이 스위치는 (a) 특정 시각 단서 하위 집합에 의해 촉발되며 (b) 모델 내부의 식별 가능한 층·head·잠재 방향에 의해 중재된다. 본 연구는 인지과학 단서 위계를 따라 체계적으로 자극을 생성하고, 행동 임계값을 측정한 뒤, probing과 인과 개입으로 내부 메커니즘을 국소화한다.

**5개 하위 과제**: (§1) 통제된 자극 세트 PhysCue 구축 + 행동 임계값 탐지; (§2) Vision encoder probing으로 "physical-ness" 차원 분리; (§3) LLM backbone logit lens + layer-wise 창발 분석; (§4) Attention/activation patching을 통한 인과적 국소화; (§5) Cross-model 일반화 및 텍스트 프롬프트 steering.

### 2.2 Sub-task 1 — PhysCue 데이터셋과 행동 임계값

**과제 정의**. 입력: 통제된 정적 이미지; 출력: 자유형식 text response에서 추출한 next-state prediction (예: "What happens next to the object?"에 대해 "falls"/"rolls"/"stays"/"moves sideways" 등). 메트릭: (1) **물리-모드 점화율 (PMR)** — 응답에 물리 동사 존재 여부의 binary rate; (2) **중력 방향 일치율 (GAR)** — 지면이 존재할 때 하향 낙하 예측 비율; (3) **응답 일관성 (RC)** — 동일 시나리오의 여러 렌더링 간 예측 일치율.

**왜 이것이 정보적인가**. 행동 임계값 curve가 모든 후속 기계적 분석의 기준선. 단일 숫자(accuracy)가 아니라 **자극 축을 따른 스위칭 함수**를 생성하는 것이 핵심 기여.

**자극 설계 (factorial 2×2×3×3×3 = 108 조건 + 각 조건당 50 시나리오)**:
- **축 A (추상화 수준, 5 수준)**: 선 원 → 채색 원 → 음영 구(light-from-above 구배) → 텍스처 공(가죽 무늬) → 사진 공.
- **축 B (배경, 3 수준)**: 빈 배경 → 단일 수평선(지면) → 완전한 풍경 (방·야외).
- **축 C (맥락 단서, 3 수준)**: 없음 → 바람선 (측면 동선) → trajectory arrow / 드리운 그림자.
- **축 D (객체 범주 라벨, 3 수준)**: "circle" vs "ball" vs "planet" (프롬프트 통제).
- **축 E (장면 일관성)**: 일관 vs 불일치 (예: 사진 공이 선화 배경 위).

각 조건에 대해 5개 물리 이벤트 템플릿 (낙하, 경사면 구름, 벽 튕김, 공중 정지, 수평 운동).

**자극 생성**: 프로그래머틱 (matplotlib/PIL/Blender) + Midjourney/SD로 사진 사실 조건 생성; WHOOPS 스타일 이중 주석으로 품질 통제.

**프롬프트**. "The image shows {object}. Describe what will happen to the {object} in the next moment." 및 forced-choice 버전 ("Which will happen next: (A) falls down, (B) stays still, (C) moves sideways, (D) this is just an abstract shape and it doesn't move").

**가설 H1**: PMR은 추상화 수준에 따라 **S자형**으로 증가하며, 3D 음영 도입과 지면 도입이 가장 큰 단계 증가를 유발한다. H2: "ball" 라벨링은 선화에서도 PMR을 크게 증가시켜 **언어 prior의 독립 기여**를 입증한다. H3: 장면 불일치는 RC를 저하시킨다.

**난이도/실현가능성**. 낮음–중간. 자극 생성과 VLM 쿼리는 표준. 규모: 108 × 50 × 3 VLM × 2 prompt = ~32k 쿼리 (open-source inference 기준 2–3일).

### 2.3 Sub-task 2 — Vision encoder에서 "physical-ness" 축의 선형 분리

**과제 정의**. PhysCue의 각 이미지를 CLIP-ViT-L/14 (LLaVA-1.5), SigLIP (LLaVA-OneVision/Qwen2-VL), InternViT-6B (InternVL2)에 통과시켜 patch 및 CLS 활성화를 추출. **Sub-task 1의 행동 PMR 라벨**을 target으로 linear probe 훈련(layer-wise, token-wise). 메트릭: probe AUC, accuracy; token 공간 분포.

**왜 이것이 정보적인가**. Pixels-to-Principles의 "vision encoder가 물리 cue를 포착하나 LLM이 활용 못함" 주장을 추상화 축 전반에서 검증. probe이 vision encoder 전반에서 높은 AUC를 얻고도 VLM 행동이 따라가지 못한다면 **디코딩 병목**을 명시적으로 localize.

**확장 방법**:
- **Gandelsman 분해**: 각 attention head별로 "physical-ness" 기여도 계산 (text-span으로 자동 라벨링). "support plane", "3D shading", "motion blur" 같은 head 후보 식별.
- **Pach et al. 2025 SAE**: CLIP 패치 토큰에 SAE 훈련 후 추상화 축을 따라 활성화되는 monosemantic feature 후보 추출.
- **인코더-디코더 해리 지수 (EDI)**: EDI = probe AUC − downstream accuracy. 큰 EDI는 "지식은 있으나 사용되지 않음"을 정량.

**예상 발견/가설**. H4: 음영(3D) cue는 vision encoder 층 ~20+에서 선형 분리됨; 지면 존재는 더 일찍 (~층 10–15). H5: probe AUC는 추상화 수준에 S자형이나 그 기울기가 행동 S자형보다 가파름 → **디코딩 측 boomerang**. H6: 특정 head(CLIP-ViT ~L22 H7 근처, 이전 shape-선호 head들과 다름)가 physical cue에 특화.

**난이도/실현가능성**. 중간. Probe은 간단하나 headwise TextSpan은 CLIP-only 방법이며 SigLIP/InternViT에는 Balasubramanian 적응 필요.

### 2.4 Sub-task 3 — LLM backbone의 층별 물리 개념 창발 (Logit lens + cross-layer probing)

**과제 정의**. LLaVA-1.5-7B, LLaVA-Next-7B, Qwen2-VL-7B, InternVL2-8B에서 **시각 토큰 위치의 hidden state**에 logit lens 적용 + per-layer linear probe ("physical object?" 이진, "gravity direction" 4방향, "next motion verb" 5-way). Neo et al. 2024의 recipe를 정확히 따름.

**왜 이것이 정보적인가**. "물리 모드가 언제 생기는가"를 **층 수준에서** 답한다. 초기 층(1-4)에서 물리 정보가 Basu et al. 2024처럼 저장되는지, 혹은 중간 층(15-24)에서 Neo et al.처럼 창발하는지 판별.

**분석 구성**:
- **Logit lens across layers**: 각 층의 unembed 투영에서 물리 동사("fall", "roll", "bounce", "sit") vs 기하 명사("circle", "shape", "line")의 logit 궤적 추적.
- **Per-layer probe on "physics mode" binary label**: 각 층·각 시각 토큰 위치에 probe → 2D heatmap (layer × token position)이 "물리 개념의 부상 시공간도".
- **추상화 수준별 구분 분석**: 음영 구 vs 선 원의 hidden state 차이를 층별로 측정 (코사인 거리, CKA).
- **Cross-model 비교**: LLaVA-1.5(단순 MLP projector) vs LLaVA-Next(추가 해상도 토큰) vs Qwen2-VL(dynamic resolution + M-RoPE) vs InternVL2(pixel shuffle)의 층별 지도 대조.

**예상 발견**. H7: 선 원은 logit lens에서 "circle"/"shape"를 투사하며 물리 동사가 후기 층까지 부상하지 않음. H8: 음영 구 + 지면은 중간 층(~L15-20)에서 "ball"·"fall" 쌍의 로짓이 공동 상승. H9: Qwen2-VL/InternVL2가 LLaVA-1.5보다 **더 이른 층에서 물리 모드 스위칭** (더 큰 vision encoder와 더 세련된 projector 덕분).

**난이도/실현가능성**. 중간. Hidden state 후크 코드는 표준. SigLIP/InternViT 기반 VLM에서도 Neo et al. 방법이 적용됨 (Qwen2-VL에서 이미 검증).

### 2.5 Sub-task 4 — 인과적 국소화 (Semantic Image Pairs + attention patching + steering)

**과제 정의**. PhysCue로부터 **Semantic Image Pairs (SIP)** 구성 — 각 쌍은 단일 cue 축(예: 음영 유무, 지면 유무)만 차이. "clean" 이미지(물리 응답 유발)와 "corrupted" 이미지(유발하지 않음) 사이에서 **활성화 패치**를 수행 (Golovanevsky et al. NAACL 2025의 NOTICE recipe). 메트릭: 패치된 층/head의 **indirect effect** (IE), 즉 clean 응답 확률의 회복.

**왜 이것이 정보적인가**. Probing과 logit lens는 *상관*이지만 인과가 아니다. 이 하위 과제는 "음영 cue가 물리 모드를 유발하는 데 실제로 필요한 구성요소"를 식별한다. Gaussian noise 방식의 환각(Zhang & Nanda 2024) 회피를 위해 SIP 방법을 채택.

**개입 유형**:
- **Visual token patching**: corrupted의 객체 시각 토큰을 clean의 대응 토큰으로 교체, 층별 IE 측정.
- **Attention knockout**: 시각 토큰 ↔ 마지막 토큰 간 attention을 특정 head/층에서 0으로. Physics head 후보 랭킹.
- **MLP replacement**: 각 층 MLP 출력을 clean으로 복원.
- **Steering vector 개입 (VTI-style)**: (physics-mode response, non-physics response) 쌍으로부터 residual stream에 shift vector 계산. Test-time 추가로 선 원을 물리 모드로 "강제"할 수 있는지 검증.
- **SAE 개입 (Pach et al. recipe)**: CLIP SAE feature 중 "shading" / "ground plane" 방향을 찾아 증폭/억제, LLM 출력 변화 측정.

**예상 발견**. H10: 2–3개 좁은 layer·head 범위(중간 층, Kaduri et al.의 중간 25%에 일치)가 큰 IE를 보임. H11: Steering vector는 선 원에서 물리 모드를 유발 가능 → "물리성은 LLM 내에서 선형 방향으로 국소화됨". H12: LLaVA는 Golovanevsky et al.의 결과와 일관되게 visual-grounding self-attention head 부재; 물리 트리거는 **MLP 경로**에 의존할 것으로 예측.

**난이도/실현가능성**. 중간–높음. 패치 코드는 TransformerLens/nnsight로 구현. SAE 훈련은 추가 compute 필요(A100 수 일). 가장 헤드라인이 될 결과.

### 2.6 Sub-task 5 — Cross-model 비교 + 프롬프트/이미지 steering의 상호작용

**과제 정의**. LLaVA-1.5-7B/13B, LLaVA-Next-7B, Qwen-VL, Qwen2-VL-7B, InternVL2-8B/26B, (가능시 Llama-3.2-Vision) 5–7개 모델에서 Sub-tasks 1-4 축약 버전 실행. 추가로 Gavrikov et al. 2024 스타일 **프롬프트 steering**: "treat this as an abstract geometric shape" vs "treat this as a physical object subject to gravity"로 PMR 변화 측정.

**왜 이것이 정보적인가**. 발견의 일반성 vs 모델-특수성 확립. 프롬프트 steering은 **이미지 단서와 언어 단서가 독립적/상호작용적으로 물리 모드를 유발하는지** 판별.

**예상 발견**. H13: "ball" 프롬프트는 선 원의 PMR을 상당히 올리나 음영 구 + 지면의 효과에 미치지 못함 → 시각 단서가 여전히 독립적 기여. H14: 더 큰 projector (InternVL2 26B의 pixel shuffle)를 가진 모델은 이미지 단서 의존도가 더 높음. H15: Qwen2-VL의 M-RoPE가 더 풍부한 공간 단서 활용 → 지면 cue 효과가 LLaVA보다 큼.

**난이도/실현가능성**. 낮음–중간 (이미 구축된 파이프라인 재사용).

### 2.7 최소 가능 실험 vs 야심적 버전

**최소 가능 (EMNLP short 또는 workshop 급)**: Sub-task 1 (축 A + 축 B만, 54 조건) + Sub-task 2 (CLIP/SigLIP probing만) + Sub-task 3의 logit lens (LLaVA-1.5만). 1개월 1인 compute.

**표준 풀 페이퍼 (EMNLP long)**: Sub-tasks 1–3 + Sub-task 4의 SIP patching (steering/SAE 제외) + Sub-task 5의 2모델 비교. 3–4개월 1인.

**야심적 (NeurIPS)**: 전체 5개 하위 과제 + SAE 기반 물리 feature 발견 + 5개 이상 VLM 비교 + 인간 기준선 수집(Prolific, PhysCue 50개 샘플, 20명). 6개월 + 2인 compute.

### 2.8 하위 과제 간 내러티브 연결

§1은 행동 스위칭 커브를 관찰 → §2는 이 커브가 vision encoder에 이미 존재함을 보임 → §3은 LLM 내에서 언제 창발하는지 층별 지도 → §4는 어떤 구성요소가 인과적으로 담당하는지 → §5는 모델/프롬프트 일반성. 세 가지 "왜"가 차례로 답변됨: *무엇이 트리거하는가(§1)*, *어디에 국소화되는가(§2-3)*, *어떻게 개입 가능한가(§4-5)*.

---

## Part 3. 포지셔닝, 헤드라인, 위험

### 3.1 NeurIPS vs EMNLP 프레이밍

**NeurIPS (ML/해석가능성 앵글)**: 제목을 "Localizing Physical-Mode Activation in Vision-Language Models"처럼. Spelke/Michotte 인지과학을 부차로, **기계적 발견**(SAE feature, patching IE curve, steering vector)을 전면에. Track: Neurips main (Interpretability, Evaluation & Analysis). Figure 1은 PhysCue 자극 격자 + layer-wise probe heatmap. Sub-task 4 결과가 헤드라인 claim.

**EMNLP (언어/접지 앵글)**: 제목을 "When Does a Circle Become a Ball? Probing Physical-Object Reasoning Triggers in Vision-Language Models"처럼. 이미지 단서 × 언어 라벨의 상호작용(Sub-task 5)을 전면에; **vision–language grounding의 실패 모드**로 프레이밍 (Pixels-to-Principles를 직접 잇는 후속). Track: EMNLP Interpretability and Analysis of Models for NLP 또는 Resources and Evaluation. Gavrikov et al. 2024을 핵심 대화 상대로.

### 3.2 헤드라인 결과 후보

1. **"S자형 스위칭 커브"**: Open-source VLM은 5단계 추상화에서 모두 가능한 것이 아니라 음영+지면이 동시에 존재할 때 **급격한 상전이**를 보인다. 3개 모델에서 복제됨.
2. **"Encoder-decoder boomerang"**: Vision encoder probing은 선 원에서도 67% AUC로 "physicalness"를 선형 분리하나 행동 PMR은 8%에 머문다 → 실패 위치가 디코더.
3. **"Physics head localization"**: LLaVA-1.5의 layer 19, head 14 (예시)의 attention knockout이 물리 모드 PMR을 50%p 감소; 동일한 head를 활성화된 상태로 유지하면서 다른 모든 visual attention 제거시 효과 보존 → **소수 head의 인과적 필요성**.
4. **"Physics steering vector"**: (ball, circle) 쌍에서 계산한 residual stream 방향을 layer 15에 주입하면 선 원 이미지도 70% 확률로 물리 응답 유도. Shape-texture steering (Gavrikov et al.)의 물리적 대응물.

가장 바이럴할 헤드라인은 2 + 3의 조합: "VLM은 원을 공으로 보지만 말하지 않는다 — 그리고 우리는 그것을 강제할 수 있다".

### 3.3 위험 요소 및 대체 질문

**R1: 모델이 명확한 스위칭 행동을 보이지 않는 경우 (단조 증가만)**. Fallback: (a) 스위칭 *형태*의 모델 간 차이를 분석; (b) 인간 baseline과의 일치도로 재구성 ("인간과 VLM이 같은 cue에 반응하는가?" cog-sci alignment 페이퍼로 재포장). Vo et al. 2025이 유사한 전환을 보인 선례.

**R2: Vision encoder에서도 probe 분리가 안 되는 경우**. 이는 그 자체로 주요 발견 — "CLIP/SigLIP은 물리 cue를 전혀 인코딩하지 않음" (Eyes Wide Shut의 확장). DINOv2에서 반대 결과가 나온다면 MoF 제안과 직접 연결.

**R3: Patching에서 분산된 효과 (집중된 head 없음)**. Fallback: SAE 기반 feature-level 발견으로 전환. Monosemantic "support"/"gravity" feature 식별 자체가 흥미로운 결과.

**R4: 결과의 open-source VLM 특수성 (GPT-4V 일반화 불가)**. 대책: 소규모 closed-source 행동 대조(Sub-task 1만)를 포함하여 scaling 주장 가능. Pixels-to-Principles는 Gemini를 이렇게 포함.

**R5: 자극 품질 문제 (합성 vs 사진 구분이 VLM에게 domain-shift로 작용)**. WHOOPS의 교훈을 적용 — Bitton-Guetta et al.이 이미 accuracy 차이가 합성 아티팩트가 아니라 commonsense에서 기인함을 보였다. 동일한 검증을 PhysCue에 적용 (same-scene matched rendering 스타일 3개, domain shift control).

**R6: 자극의 언어적 혼재 (프롬프트가 너무 강하게 유도)**. 대책: forced-choice vs open-ended 대조; 프롬프트 신호 최소화 ("What do you see? What happens next?").

### 3.4 참신성 평가

세 영역 교집합에서 **중복되는 선행 연구 없음**. 가장 가까운 경쟁자와의 구별:

- vs **Pixels to Principles (Ballout et al. 2025)**: 동일 3개 open-source VLM, 동일 probing 접근. 차별점 — 그들은 plausibility 판단 + 사진 자극 + t-SNE 수준 분석. 본 연구는 next-state 예측 + 추상화 축 매개변수적 변화 + causal patching/steering. **Pixels-to-Principles 발견을 기반으로 *왜 misalignment가 발생하고 어떤 단서가 이를 악화시키는가*를 묻는 자연스러운 후속작**.
- vs **"VLMs are Blind" (Rahmanzadehgervi et al. 2024)**: 추상 도형에서 VLM 실패 입증 + encoder probing. 차별점 — 물리 과제 없음, steering/patching 없음. 본 연구는 "Blind" 영역과 "Physics" 영역을 연결하는 첫 다리.
- vs **"Are VLMs Shape or Texture Biased" (Gavrikov et al. 2024)**: 프롬프트 steering 방법론의 직접 선례. 차별점 — 물리 모드를 대상으로 함. 본 연구는 그들의 방법을 **새로운 차원(물리성)**으로 확장.
- vs **MechBench (Zhang et al. 2024)**: 역학 추론 VLM 벤치마크. 차별점 — 벤치마크 only; 기계적 분석 없음; 추상화 조작 없음.

**Venue 추천**. **EMNLP 2026 long** (마감 5/2026 추정)이 가장 적합 — interpretability track이 성장 중이고 Pixels-to-Principles, NOTICE, BlindTest 모두 NAACL/EMNLP/ACCV 계열. **NeurIPS 2026** (마감 5월)은 Datasets & Benchmarks 트랙에 PhysCue 단독으로 2차 제출 후보. **ICLR 2027**도 기계적 결과가 강력하면 가능.

---

## 결론

사용자가 제안한 연구 질문은 현재 VLM 문헌의 **세 공백의 교집합** — 추상–구체 자극 조작, 다음 상태 예측에서 open-source VLM, 물리 트리거의 기계적 국소화 — 에 정확히 놓인다. 인지과학 cue 위계(3D 음영 > 그림자+지면 > 재질 > 원근 > 운동 흐릿함)는 자극 설계에 **원칙적 뼈대**를 제공하며, 최근 성숙한 기계적 해석가능성 도구(Neo et al. logit lens + attention knockout, Golovanevsky SIP, Pach SAE, VTI/VISTA steering)는 **methodological lift가 사실상 0**임을 의미한다. Pixels-to-Principles의 최근 vision-language misalignment 발견은 본 연구의 이상적 도약대이며, Gavrikov et al.의 shape-texture steering 패러다임은 "physics-mode steering"으로의 자연스러운 확장을 제공한다. 최소 가능 실험(1개월)이 EMNLP short로 방어 가능하고, 야심적 버전(6개월)은 NeurIPS main track 경쟁력을 확보할 수 있다. 가장 큰 위험 — 명확한 스위칭이 관찰되지 않을 경우 — 에 대해서도 단조 변화 자체와 인간 baseline 대조로 재프레이밍하는 fallback이 존재하여 연구 투자 수익이 안정적이다.