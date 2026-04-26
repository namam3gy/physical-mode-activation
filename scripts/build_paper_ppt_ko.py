"""Paper-style PPT (Korean) — introduction → related work → problem →
implementation → experiments → conclusion.

Companion to `build_review_ppt.py` (which is timeline-style: M1 → M9 →
§4.X). This deck reorganizes the same material into a paper-talk
structure for advisor / collaborators / paper-shop presentation. All
text is Korean (per user request); per-slide detailed explanations
(for domain newcomers) live in `docs/review_ppt/physical_mode_paper_ko.md`.

Usage:
    uv run python scripts/build_paper_ppt_ko.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# Import shared layout helpers
sys.path.insert(0, str(Path(__file__).parent))
from _ppt_layout import (  # noqa: E402
    SLIDE_W, SLIDE_H, ACCENT, GRAY_DARK, GRAY_MID, GRAY_LIGHT, WHITE,
    add_text_box, add_bullets, add_figure, add_caption, add_title_bar,
    add_footer, add_table, new_slide,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

# Korean-supporting font name. PowerPoint / Keynote / LibreOffice will
# auto-fall back to system Korean fonts if this exact name is missing.
FONT = "Malgun Gothic"
FOOTER_TEXT = "VLM 의 추상-물리 shortcut — 5-model 실험과 픽셀-인코드 가능성"


# ============================================================================
# 슬라이드 정의
# ============================================================================


def slide_01_title(prs):
    s = new_slide(prs)
    add_text_box(s, Inches(0.5), Inches(1.6), Inches(12.3), Inches(1.5),
                 "오픈소스 VLM 은 언제 원을 공으로 보는가?",
                 size=42, bold=True, color=ACCENT, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(3.0), Inches(12.3), Inches(0.8),
                 "5 개 오픈소스 VLM 에서 추상→물리 shortcut 의\n"
                 "행동 / 인과 / 픽셀 수준 분석",
                 size=22, color=GRAY_DARK, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(0.5),
                 "Qwen2.5-VL · LLaVA-1.5 · LLaVA-Next · Idefics2 · InternVL3",
                 size=15, color=GRAY_MID, align=PP_ALIGN.CENTER, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.4),
                 "논문 발표용 / 2026-04-26",
                 size=14, color=GRAY_MID, align=PP_ALIGN.CENTER, font_name=FONT)
    return s


# ----- Section 1: Introduction -----

def slide_02_motivation(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "동기 — VLM 의 shortcut 문제",
                      subtitle="흰 배경의 검은 원을 보고 \"공이 떨어진다\" 라고 말하는 모델",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.1), Inches(7.5), Inches(5.0), [
        "프롬프트는 open-ended (\"이 다음에 무슨 일이 일어날까?\").",
        ("이미지에는 중력 단서, 지면, 그림자, 텍스처가 *전혀 없음*. "
         "사람은 \"흰 배경 위 검은 원\" — 추상 도형이라고 묘사.", 14),
        ("그런데 Qwen2.5-VL: \"공이 중력으로 떨어진다.\" 같이 *물리* "
         "용어로 응답.", 15),
        ("이게 shortcut: 시각 증거는 abstract / physical 둘 다와 호환되지만, "
         "모델이 한 쪽으로 collapse.", 15),
        ("기존 문헌 (Eyes Wide Shut, Tong et al. 2024 등) 이 anecdotal "
         "하게 보고했지만 *어디서 일어나는지* 는 미해결.", 14),
    ], font_name=FONT)
    add_figure(s, FIG_DIR / "01_line_blank_none.png",
               Inches(8.5), y + Inches(0.5), w=Inches(4.5))
    add_caption(s, Inches(8.5), Inches(5.6), Inches(4.5),
                "본 연구의 baseline 자극: line/blank/none. "
                "Qwen 응답 → 물리, 사람 → 추상.", font_name=FONT)
    return s


def slide_03_question(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "본 연구의 핵심 질문",
                      subtitle="언제 / 어디서 / 어떻게 — 세 가지 차원에서 shortcut 을 localize",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.2), Inches(12.3), Inches(5.5), [
        ("[행동] 어떤 자극에서, 어떤 모델에서, shortcut 이 얼마나 강한가? "
         "어떤 행동 시그니처가 있는가?", 18),
        ("[메커니즘] 모델 내부에서 추상→물리 결정이 *어디서* 일어나는가? "
         "특정 LM 레이어 / 방향이 행동을 인과적으로 결정하는가?", 18),
        ("[픽셀] shortcut 이 이미지 자체에 *인코드 가능* 한가? "
         "픽셀에 작은 perturbation 을 가해서 행동을 뒤집을 수 있는가? "
         "이게 가능하다면, 어떤 architecture 에서 가능한가?", 18),
    ], font_name=FONT)
    return s


def slide_04_contributions(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "주요 기여 3 가지", font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.1), Inches(12.3), Inches(5.5), [
        ("**Architecture-level reframe**: 5 model × 3 stim source × "
         "bootstrap CI 로, 행동 PMR ceiling 이 *encoder representational "
         "capacity 만으로* 결정되지 않음을 disconfirm. 결정자는 "
         "**joint encoder + LM (architecture 수준)**.", 16),
        ("**Causal localization**: Qwen2.5-VL 의 LM layer 10 에 단일 선형 "
         "방향 `v_L10` 이 존재. forward-hook 으로 +α=40 더하면 line/blank/none "
         "stim 의 10/10 이 \"D: abstract\" → \"B: stays still\" flip. "
         "다른 layer 는 같은 α 로 안 움직임.", 16),
        ("**Pixel encodability — encoder-saturation 특이적**: Qwen 에서 "
         "픽셀-공간 gradient ascent (ε=0.05) 로 5/5 PMR flip. 매칭 magnitude "
         "random direction 은 0/15. **그러나 cross-model null**: 4 비-Qwen "
         "모델에 transfer 안 되고, LLaVA-1.5 자체 v_L10 으로도 0/5 flip "
         "(projection 은 Qwen 매칭). 픽셀-인코드 가능성은 saturated "
         "architecture 의 속성.", 15),
    ], font_name=FONT)
    return s


# ----- Section 2: Related Work -----

def slide_05_related_shortcut(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "관련 연구 — VLM grounding-failure / shortcut",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.1), Inches(12.3), Inches(5.5), [
        ("**Eyes Wide Shut** (Tong et al., 2024) — VLM 이 놓치는 visual "
         "primitives. MoF (Mixture-of-Features) 제안.", 15),
        ("**Vision-language grounding failure** 분석 (다수). 행동적 "
         "관찰 위주 — 어디서 실패하는지 알지만 *왜* 그곳에서 *어떻게* "
         "실패하는지 메커니즘 분석 부재.", 15),
        ("**Language-prior dominance** 가 VLM benchmark 에서 anecdotal 하게 "
         "보고 — circle vs ball label 이 같은 이미지에서 다른 응답을 유도, "
         "본 연구의 H7 (label-selects-regime) 의 사전 관찰.", 15),
        ("**한계**: 기존 연구는 행동 수준 (output 통계) 에서 정지. *어느 "
         "레이어 / 방향* 이 결정하는지의 mechanism-level localization 은 "
         "VLM 에 거의 적용 안 됨.", 15),
    ], font_name=FONT)
    return s


def slide_06_related_probing(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "관련 연구 — Probing / Causal interpretability",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.1), Inches(12.3), Inches(5.5), [
        ("**Linear probing** (Alain & Bengio; Belrose et al.) — hidden "
         "state 가 어떤 정보를 인코드하는지 logistic regression 으로 측정.", 15),
        ("**Logit lens** (nostalgebraist; tuned lens — Belrose et al.) — "
         "각 레이어에서 vocab projection 으로 \"중간 결정\" 추적.", 15),
        ("**Activation patching / SIP** (Wang et al.; Conmy et al.) — "
         "clean run 의 hidden state 를 corrupted run 에 patch 하여 "
         "indirect effect 측정. LM 에 적용; VLM 에는 부분적.", 15),
        ("**VTI steering vectors / class-mean directions** (Burns et al. 등). "
         "physics-mode = 1 vs = 0 의 hidden state 평균 차이로 \"physics-mode "
         "방향\" 추출, runtime intervention.", 15),
        ("**Adversarial / feature visualization** (Goodfellow et al.; "
         "Madry et al.; Olah et al.). 본 §4.6 은 후자에 가까움 — class-mean "
         "방향을 target 으로 한 픽셀-공간 feature visualization.", 15),
    ], font_name=FONT)
    return s


# ----- Section 3: Problem & Definitions -----

def slide_07_problem(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "문제 정의 — 두 차원 측정",
                      subtitle="행동 (PMR / GAR / RC) + 메커니즘 (probe / steering / pixel-encoding)",
                      font_name=FONT)
    add_text_box(s, Inches(0.5), y + Inches(0.1), Inches(6.0), Inches(0.5),
                 "행동 (Behavioral)", size=18, bold=True, color=ACCENT, font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.6), Inches(6.0), Inches(2.5), [
        ("PMR — 응답에 물리 동사 포함 비율 (\"falls\", \"rolls\")", 13),
        ("GAR — gravity-align rate (PMR 의 부분, 하방 운동)", 13),
        ("RC — response consistency (T=0.7 seed 간 일관성)", 13),
    ], font_name=FONT)
    add_text_box(s, Inches(7.0), y + Inches(0.1), Inches(6.0), Inches(0.5),
                 "메커니즘 (Mechanistic)", size=18, bold=True, color=ACCENT, font_name=FONT)
    add_bullets(s, Inches(7.0), y + Inches(0.6), Inches(6.0), Inches(2.5), [
        ("Vision encoder probe AUC — 인코더가 추상 vs 물리 cell 을 "
         "선형 분리할 수 있는가?", 13),
        ("LM logit lens — 각 LM 레이어에서 visual-token 위치의 hidden "
         "state 가 PMR 을 예측하는가?", 13),
        ("Causal intervention — `α · v_L10` 을 forward hook 으로 더하면 "
         "출력이 뒤집히는가?", 13),
        ("Pixel encoding — 픽셀 공간 gradient ascent 가 행동을 뒤집는가?", 13),
    ], font_name=FONT)
    return s


def slide_08_stim_design(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "자극 설계 — M2 5축 factorial (2880 stim)",
                      subtitle="object_level × bg_level × cue_level × event × seed",
                      font_name=FONT)
    figs = ["01_line_blank_none.png", "02_line_ground_none.png",
            "03_shaded_ground_none.png", "04_textured_ground_arrow_shadow.png",
            "05_filled_blank_wind.png", "06_textured_blank_none.png"]
    labels = ["line/blank/none\n(가장 추상)", "line/ground/none\n(+ 지면)",
              "shaded/ground/none\n(3D 공)",
              "textured/ground/\narrow_shadow\n(최대 cue)",
              "filled/blank/wind\n(바람 cue)",
              "textured/blank/none\n(텍스처만)"]
    fig_w = Inches(2.0); gap = Inches(0.15)
    cols = 3
    for i, (f, lbl) in enumerate(zip(figs, labels)):
        col = i % cols; row = i // cols
        x = Inches(0.5) + (fig_w + gap) * col
        yy = y + Inches(0.1) + Inches(2.4) * row
        add_figure(s, FIG_DIR / f, x, yy, w=fig_w)
        add_text_box(s, x, yy + Inches(2.0), fig_w, Inches(0.4), lbl,
                     size=10, color=GRAY_MID, align=PP_ALIGN.CENTER, font_name=FONT)
    add_bullets(s, Inches(7.5), y + Inches(0.2), Inches(5.5), Inches(5.0), [
        ("**4 object_level**: line / filled / shaded / textured "
         "— 추상화 축", 14),
        ("**3 bg_level**: blank / ground / scene", 14),
        ("**4 cue_level**: none / cast_shadow / motion_arrow / both", 14),
        ("**3 event**: fall / rise / horizontal", 14),
        ("**10 seeds** × 3 prompts (open / FC / label-free) × 3 labels "
         "(circle / ball / planet) = 2880 추론", 13),
        ("M2 = 본 연구의 핵심 자극 set. M8a (5 도형), M8d (3 카테고리), "
         "M8c (실사진) 가 외부 타당성 확장.", 13),
    ], font_name=FONT)
    return s


def slide_09_metrics(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "메트릭 정의 (정량 측정)",
                      subtitle="모두 응답 텍스트 위 rule-based scorer; bootstrap 5000-iter CI",
                      font_name=FONT)
    rows = [
        ["Metric", "정의", "예시"],
        ["PMR(_label)", "physics-mode reading rate. label 프롬프트 (\"the ball...\") 응답에서 물리 동사 포함 비율.", "ball: 0.86"],
        ["PMR(_nolabel)", "label 없는 open-ended 프롬프트 (\"What do you see?\") 응답에서의 PMR.", "Qwen: 0.94"],
        ["GAR", "gravity-align rate — 응답에서 *하방* 운동 (falls, drops) 비율 (PMR 의 부분).", "ball: 0.79"],
        ["RC", "response consistency — T=0.7 sampling N seed 간 *같은 PMR call* 비율.", "0.918 (M2 평균)"],
        ["Paired-delta H2", "PMR(label) − PMR(_nolabel) — label 효과 (per stim 매칭).", "LLaVA ball: +0.475"],
        ["Class-mean v_L", "mean(h_L | PMR=1) − mean(h_L | PMR=0). \"physics-mode 방향\" 추정.", "Qwen v_L10 dim 3584"],
        ["v_L projection", "샘플 hidden h_L 을 v_L 단위 벡터에 사영. \"physics-mode 강도\".", "M2 cue=both 평균 42.7"],
    ]
    add_table(s, Inches(0.5), y + Inches(0.1), Inches(12.5), Inches(5.5), rows,
              header_color=ACCENT, font_size=11, font_name=FONT)
    return s


def slide_10_models(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "테스트 모델 — 5 개 오픈소스 VLM",
                      subtitle="동일 generic loader (AutoModelForImageTextToText) 로 cross-model 비교",
                      font_name=FONT)
    rows = [
        ["Model", "Vision encoder", "LM", "이미지 처리"],
        ["Qwen2.5-VL-7B", "SigLIP", "Qwen2-7B", "동적 504×504"],
        ["LLaVA-1.5-7B", "CLIP-ViT-L/14", "Vicuna-7B (LLaMA-2)", "고정 336×336"],
        ["LLaVA-Next-7B", "CLIP-ViT-L/14", "Mistral-7B", "AnyRes 5-tile"],
        ["Idefics2-8B", "SigLIP-SO400M", "Mistral-7B", "384×384"],
        ["InternVL3-8B", "InternViT-300M", "InternLM3-8B", "동적 448×448"],
    ]
    add_table(s, Inches(0.5), y + Inches(0.1), Inches(12.5), Inches(3.3), rows,
              header_color=ACCENT, font_size=14, font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0), [
        ("Encoder 계열: SigLIP × 2 (Qwen, Idefics2 SO400M), CLIP × 2 "
         "(LLaVA-1.5, LLaVA-Next), InternViT × 1 (InternVL3).", 13),
        ("LM 계열: Qwen2 × 1, Vicuna × 1, Mistral × 2, InternLM3 × 1.", 13),
        ("Cross-encoder + cross-LM 비교가 architecture-level 효과를 "
         "분리해내는 데 사용됨 (§4.5 Idefics2 swap, M6 r6 LLaVA-1.5 ↔ "
         "LLaVA-Next).", 13),
    ], font_name=FONT)
    return s


# ----- Section 4: Implementation -----

def slide_11_pipeline(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "파이프라인 — stim → inference → score → probe",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.2), Inches(12.3), Inches(5.5), [
        ("**[1] Stimulus generation** — `scripts/01_generate_stimuli.py` "
         "가 EvalConfig + FactorialSpec 으로부터 PIL stim 생성 (PIL/matplotlib).", 14),
        ("**[2] Inference** — `scripts/02_run_inference.py` 가 generic "
         "AutoModelForImageTextToText 로 모든 모델 실행. "
         "predictions.{jsonl, parquet, csv} 저장. crash-safe streaming flush.", 14),
        ("**[3] Activation capture** (옵션) — `capture_lm_layers=(5,10,15,20,25)` "
         "와 `capture_vision_layers=(3,7,11,15,19,23)` forward hook 으로 "
         "LM / vision 활성화 safetensors 저장.", 14),
        ("**[4] Scoring** — `physical_mode.metrics.pmr.score_pmr()` rule-"
         "based scorer (다국어 stem; abstract marker gating). hand-validation "
         "5-6% disagreement.", 14),
        ("**[5] Probing** — Linear logistic regression on captured activations "
         "with `behavioral_y` (PMR call) or `stim_y` (factorial label).", 14),
        ("**[6] Causal intervention** — `scripts/06_vti_steering.py` 와 "
         "`scripts/sec4_6_counterfactual_stim.py` 가 forward-hook + "
         "pixel-space gradient ascent 적용.", 14),
    ], font_name=FONT)
    return s


def slide_12_capture_probe(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "활성화 캡처 + 선형 프로빙",
                      subtitle="\"인코더가 추상 vs 물리를 선형 분리하는가?\" / \"LM 의 어느 레이어가?\"",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.1), Inches(12.3), Inches(2.8), [
        ("Vision encoder 의 selected layers (3, 7, 11, 15, 19, 23) 에 "
         "forward hook → 시각 토큰 위치의 hidden state mean-pool → "
         "로지스틱 회귀 (sklearn) → 5-fold AUC.", 14),
        ("LM 의 selected layers (5, 10, 15, 20, 25) 에 동일 처리, "
         "*시각 토큰 위치만* 마스킹.", 14),
        ("Y-target 두 가지: **behavioral_y** (per-stim PMR call) 또는 "
         "**stim_y** (factorial cell label, 예: 도형 이름).", 14),
    ], font_name=FONT)
    add_figure(s, FIG_DIR / "encoder_chain_5model.png",
               Inches(2.5), y + Inches(2.5), w=Inches(8.0))
    add_caption(s, Inches(2.5), Inches(6.5), Inches(8.0),
                "5 모델 vision encoder probe AUC chain (M8a stim).",
                font_name=FONT)
    return s


def slide_13_causal_intervention(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "인과 개입 — VTI steering + §4.6 픽셀 ascent",
                      font_name=FONT)
    add_text_box(s, Inches(0.5), y + Inches(0.1), Inches(12.3), Inches(0.5),
                 "M5a — runtime VTI steering (forward hook)",
                 size=18, bold=True, color=ACCENT, font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.6), Inches(12.3), Inches(2.0), [
        ("**v_L 추출**: v_L = mean(h_L | PMR=1) − mean(h_L | PMR=0); "
         "v_unit_L = v_L / ||v_L||.", 13),
        ("**개입**: model.model.language_model.layers[L] 의 forward "
         "출력 hidden_states 에 `α · v_unit_L` 더함 (모든 token position 균일).", 13),
        ("L ∈ {10, 15, 20, 25}, α ∈ {0, 5, 10, 20, 40, −α}, T=0.", 13),
    ], font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(0.5),
                 "§4.6 — pixel-space gradient ascent",
                 size=18, bold=True, color=ACCENT, font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.5), [
        ("**최적화 변수**: post-processor `pixel_values` tensor "
         "(Qwen: T_patches × 1176; LLaVA-1.5: 1×3×336×336).", 13),
        ("**목적함수**: `<mean(h_L10[visual]), v_L10>` 최대화. "
         "Adam, lr=1e-2, n_steps=200. float32 leaf → bf16 cast 로 "
         "vision tower → projector → LM 0..10 미분 가능.", 13),
        ("**제약**: L_∞-bounded (`δ.clamp_(-eps, eps)`) ε ∈ {0.05, 0.1, "
         "0.2} or unconstrained. Random unit-direction control 매칭 ε=0.1.", 13),
    ], font_name=FONT)
    return s


# ----- Section 5: Experiments / Results -----

def slide_14_pmr_ladder(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (1) — 5-model M2-stim PMR 사다리",
                      subtitle="동일 480 stim × 3 라벨 + label-free; 5000-iter bootstrap CI",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "m2_cross_model_pmr_ladder.png",
               Inches(0.5), y + Inches(0.1), w=Inches(8.5))
    rows = [
        ["Model", "PMR(_nolabel)", "Cluster"],
        ["LLaVA-1.5", "0.383 [0.34, 0.43]", "**Floor (CLIP+Vicuna)**"],
        ["LLaVA-Next", "0.790 [0.75, 0.83]", "Mid (CLIP+Mistral+AnyRes)"],
        ["Qwen2.5-VL", "0.938 [0.92, 0.96]", "Saturated"],
        ["Idefics2", "0.967 [0.95, 0.98]", "Saturated"],
        ["InternVL3", "0.988 [0.98, 1.00]", "Saturated"],
    ]
    add_table(s, Inches(9.2), y + Inches(0.5), Inches(3.8), Inches(3.5), rows,
              header_color=ACCENT, font_size=11, font_name=FONT)
    add_text_box(s, Inches(9.2), Inches(5.5), Inches(3.8), Inches(1.5),
                 "**핵심**: 같은 인코더 계열 (CLIP) 에서도 PMR 0.18 ↔ 0.70 — "
                 "encoder family 단독으로 결정되지 않음.\n\n"
                 "→ Architecture-level reframe.",
                 size=12, color=GRAY_DARK, font_name=FONT)
    return s


def slide_15_h1_ramp(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (2) — H1 abstraction ramp 모델별",
                      subtitle="LLaVA-1.5 만 깨끗한 monotone S-curve; 다른 모델은 천장",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "m2_cross_model_h1_ramp.png",
               Inches(0.5), y + Inches(0.1), w=Inches(8.5))
    add_bullets(s, Inches(9.2), y + Inches(0.2), Inches(3.8), Inches(5.0), [
        ("LLaVA-1.5: 0.51 → 0.81 (range +0.30, 가장 깔끔)", 13),
        ("LLaVA-Next: range +0.14 (부분 saturate)", 13),
        ("Qwen / Idefics2 / InternVL3: range ≤ +0.09 (천장)", 13),
        ("**해석**: H1 ramp 는 *unsaturated 인코더* 에서만 측정 가능. "
         "포화된 모델은 axis 가 압축됨.", 13),
    ], font_name=FONT)
    return s


def slide_16_h2_paired(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (3) — H2 paired-delta: 3가지 architecture-conditional 패턴",
                      subtitle="PMR(label) − PMR(_nolabel) 의 모델별 부호 패턴이 saturation 과 일치",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "m2_cross_model_h2_paired_delta.png",
               Inches(0.5), y + Inches(0.1), w=Inches(7.5))
    add_bullets(s, Inches(8.3), y + Inches(0.1), Inches(4.7), Inches(5.5), [
        ("**LLaVA-1.5 / LLaVA-Next** (unsaturated CLIP): 모든 라벨 양수. "
         "ball Δ=+0.475 (LLaVA-1.5) / +0.190 (LLaVA-Next).", 13),
        ("**Qwen / Idefics2** (포화 SigLIP-계열): 비대칭 — 비-물리 라벨 "
         "(circle, planet) 이 baseline *아래로* 억제 \"circle override\".", 13),
        ("**InternVL3** (완전 포화): 모든 Δ ≈ 0.", 13),
        ("→ H2 가 \"항상 라벨이 PMR 을 더한다\" 가 아님. **encoder "
         "saturation 이 어느 패턴이 적용되는지를 결정**.", 14),
    ], font_name=FONT)
    return s


def slide_17_encoder_probes(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (4) — Vision encoder probe (5-model AUC chain)",
                      subtitle="비-CLIP cluster ≥ 0.88; CLIP-ViT-L 만 그 아래",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "encoder_chain_5model.png",
               Inches(0.5), y + Inches(0.1), w=Inches(8.5))
    add_bullets(s, Inches(9.2), y + Inches(0.5), Inches(3.8), Inches(5.0), [
        ("Qwen SigLIP: AUC 0.99", 12),
        ("Idefics2 SigLIP-SO400M: 0.93", 12),
        ("InternVL3 InternViT: 0.89", 12),
        ("LLaVA-Next CLIP-ViT-L: 0.77", 12),
        ("LLaVA-1.5 CLIP-ViT-L: 0.73", 12),
        ("**Stim-defined Y target 으로 모든 인코더 AUC = 1.0** — 인코더 "
         "표현능력은 균일.", 12),
        ("**인코더 차이는 PMR 차이를 직접 설명하지 못함** (0.77 vs 0.93 "
         "vs 행동 PMR 0.18 vs 0.88).", 12),
    ], font_name=FONT)
    return s


def slide_18_m5a_steering(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (5) — M5a VTI causal steering: L10 only flips",
                      subtitle="line/blank/none 10/10 baseline → L10 α=40 만에서 \"D: abstract\" → \"B: stays still\"",
                      font_name=FONT)
    add_text_box(s, Inches(0.5), y + Inches(0.1), Inches(7.5), Inches(0.5),
                 "First-letter 응답 분포 (forced-choice)",
                 size=16, bold=True, color=ACCENT, font_name=FONT)
    rows = [
        ["layer", "α=0", "α=5", "α=10", "α=20", "α=40"],
        ["10", "10 D", "10 D", "10 D", "10 D", "**10 B**"],
        ["15", "10 D", "10 D", "10 D", "10 D", "10 D"],
        ["20", "10 D", "10 D", "10 D", "10 D", "10 D"],
        ["25", "10 D", "10 D", "10 D", "10 D", "10 D"],
    ]
    add_table(s, Inches(0.5), y + Inches(0.6), Inches(7.5), Inches(2.5), rows,
              header_color=ACCENT, font_size=14, font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.5), [
        ("L10 α=40 응답 예: \"It stays still — the circle appears to be "
         "floating or suspended in space without any external force...\" "
         "→ 추상 도형이 *물리 객체* 로 인식됨 (정지 상태의).", 13),
        ("M5a-ext Exp 3: −α=40 도 D → B 로 flip. v_L10 은 단방향 activator 가 "
         "아니라 **regime axis** (+α 동적, −α 정적).", 13),
        ("→ shortcut 의 *causal locus* 가 LM mid-layer (L10) 에 단일 "
         "선형 방향으로 존재.", 13),
    ], font_name=FONT)
    add_figure(s, FIG_DIR / "01_line_blank_none.png",
               Inches(8.5), y + Inches(0.6), w=Inches(4.0))
    add_caption(s, Inches(8.5), Inches(4.0), Inches(4.0),
                "Steered 자극: line/blank/none.", font_name=FONT)
    return s


def slide_19_sec46_qwen(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (6) — §4.6 픽셀-공간 gradient ascent (Qwen)",
                      subtitle="ε=0.05 에서 5/5 v_L10 flip; 매칭 random 0/15 — 방향 특이성",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "sec4_6_counterfactual_stim_panels.png",
               Inches(0.5), y + Inches(0.1), w=Inches(12.0))
    add_caption(s, Inches(0.5), Inches(5.5), Inches(12.0),
                "baseline → ε=0.05 → ε=0.1 → unconstrained. 작은 ε 에서 "
                "abstract 원의 형태는 보존; 픽셀 노이즈만 추가됨에도 모델은 "
                "\"공이 떨어진다\" 라고 응답함.", font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(6.0), Inches(12.3), Inches(1.0), [
        ("응답: \"The circle will continue to fall downward due to gravity.\" "
         "vs random control: \"The circle will remain stationary as there is "
         "no indication of movement.\" → 방향 특이성, magnitude 가 아님.", 13),
    ], font_name=FONT)
    return s


def slide_20_sec46_cross_null(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (7) — §4.6 cross-model NULL: pixel-encodability 는 saturation 특이적",
                      subtitle="LLaVA-1.5 자체 v_L10 으로도 0/5 PMR flip, projection 은 매칭",
                      font_name=FONT)
    rows = [
        ["테스트", "v_L10 flip", "projection 변화", "해석"],
        ["§4.6 (Qwen)", "5/5 at ε=0.05", "8 → ~100", "성공 — Qwen 은 픽셀 인코드 가능"],
        ["Transfer (Qwen → LLaVA)", "0/5", "(transfer)", "Adversarial 모델별 — 일반적"],
        ["Transfer (Qwen → LLaVA-Next)", "0/5", "(transfer)", "동일"],
        ["Transfer (Qwen → Idefics2)", "0/5", "(transfer)", "동일"],
        ["Transfer (Qwen → InternVL3)", "0/5 (음의 transfer)", "1.0 → 0.2 unconstrained", "큰 ε 가 saturated 출력 교란"],
        ["**LLaVA-1.5 자체 v_L10**", "**0/5 모든 ε**", "**8 → 150-200 (Qwen 매칭)**", "**Projection vs Behavior 분리 — pixel-encodability 는 Qwen 만**"],
    ]
    add_table(s, Inches(0.3), y + Inches(0.1), Inches(12.7), Inches(4.0), rows,
              header_color=ACCENT, font_size=11, font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(2.0), [
        ("Gradient ascent 자체는 LLaVA-1.5 에서 정상 작동 — projection 8→200 "
         "도달, gradient 흐름 검증됨. 하지만 LM 출력은 변하지 않음.", 13),
        ("**해석**: Qwen 의 saturated SigLIP 이 LM 이 직접 읽는 픽셀-to-L10 "
         "채널을 만듦. LLaVA-1.5 의 unsaturated CLIP 은 그 채널이 없음.", 13),
        ("→ **H-shortcut** 은 Qwen-scoped. M9 PMR-천장 / §4.7 결정-안정성 "
         "천장과 평행한 **세 번째 saturation 시그니처**.", 13),
    ], font_name=FONT)
    return s


def slide_21_external_validity(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (8) — 외부 타당성 (M8a / M8d / M8c)",
                      subtitle="비-원 도형 / 비-공 카테고리 / 실사진",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "m8a_shape_grid.png",
               Inches(0.4), y + Inches(0.1), w=Inches(4.2))
    add_caption(s, Inches(0.4), Inches(4.4), Inches(4.2),
                "M8a 5×4: 5 도형 × 4 추상화 레벨", font_name=FONT)
    add_figure(s, FIG_DIR / "m8d_full_scene_samples.png",
               Inches(4.8), y + Inches(0.1), w=Inches(4.2))
    add_caption(s, Inches(4.8), Inches(4.4), Inches(4.2),
                "M8d: 카테고리 × 이벤트 (car / person / bird)", font_name=FONT)
    add_figure(s, FIG_DIR / "m8c_photo_grid.png",
               Inches(9.2), y + Inches(0.1), w=Inches(3.8))
    add_caption(s, Inches(9.2), Inches(4.4), Inches(3.8),
                "M8c: 60 실사진 × 5 카테고리", font_name=FONT)
    add_bullets(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0), [
        ("**M8a strict scoring**: Qwen 1/4 (천장), LLaVA 4/4 — 비대칭 "
         "그 자체가 saturation 가설의 cross-shape 검증.", 12),
        ("**M8d**: LLaVA 3/3 H7 PASS (car/person/bird). Qwen ceiling-flat.", 12),
        ("**M8c**: 사진은 인코더 격차를 *압축* (모든 모델 PMR [0.18, 0.67] "
         "수렴) — saturation 의 co-factor 는 \"input-context simplicity\" 도 포함.", 12),
    ], font_name=FONT)
    return s


def slide_22_multilingual(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결과 (9) — Multilingual labels (§4.3): 5 모델 × 한·일 언어",
                      subtitle="한국어 (공/원/행성), 일본어 (ボール/円/惑星) 라벨로 H2 cross-language 테스트",
                      font_name=FONT)
    add_figure(s, FIG_DIR / "sec4_3_korean_vs_english_cross_model.png",
               Inches(0.5), y + Inches(0.1), w=Inches(7.5))
    add_bullets(s, Inches(8.3), y + Inches(0.1), Inches(4.7), Inches(5.5), [
        ("Cross-label ordering 한국어에서 4/5 모델 보존 (planet > ball > circle).", 12),
        ("LLaVA-1.5 가 가장 큰 swing (avg |Δ| = 0.11) — Vicuna 의 한국어 "
         "coverage 가 가장 약함.", 12),
        ("**일본어** 는 다른 mechanism 노출: Qwen genuinely engages JA "
         "(label-echo 85-91%); LLaVA-1.5 internally translates kanji to "
         "English; **Idefics2 falls back to Chinese on 惑星** (24%).", 12),
        ("Scorer 다국어 확장 (KO/JA/CN substring matching, 51→54 케이스).", 12),
    ], font_name=FONT)
    return s


# ----- Section 6: Discussion -----

def slide_23_architecture_reframe(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "Discussion (1) — Architecture-level reframe",
                      subtitle="3가지 saturation 시그니처가 동일 architectural 속성의 다른 측면",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.2), Inches(12.3), Inches(5.5), [
        ("**Signature 1 — PMR ceiling** (M9 paper Table 1): 행동 PMR 이 "
         "non-CLIP cluster [0.80, 0.92] 와 CLIP cluster [0.14, 0.37] 로 "
         "분리. 사진에서는 [0.18, 0.67] 로 수렴.", 16),
        ("**Signature 2 — Decision-stability ceiling** (§4.7): non-CLIP "
         "모델 (Qwen / Idefics2 / InternVL3) 이 cue 발화 시 5 seed "
         "모두 같은 PMR call 로 수렴 (RC ≈ 1.0). CLIP-기반 모델은 강한 "
         "cue 에서도 seed-level variance 보유.", 16),
        ("**Signature 3 — Pixel encodability** (§4.6 cross-model, NEW): "
         "Qwen 은 픽셀-공간 gradient ascent 로 PMR flip 가능 (ε=0.05 5/5). "
         "LLaVA-1.5 는 projection 이 매칭되어도 0/5 flip — pixel-to-L10 "
         "channel 이 없음.", 16),
        ("3가지 signature 가 **동일한 architecture-level saturation 속성** "
         "의 다른 표현. \"encoder representational capacity 만으로 결정\" "
         "이라는 단순 인코더 가설은 disconfirm — joint encoder + LM 이 "
         "결정자.", 15),
    ], font_name=FONT)
    return s


def slide_24_limitations(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "Discussion (2) — 한계 및 미해결 질문", font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.2), Inches(12.3), Inches(5.5), [
        ("**LM-only counterfactual 부재**: LLaVA-1.5 → LLaVA-Next 의 0.52 "
         "PMR jump 는 4축 confound (AnyRes / projector / training / LM family). "
         "동일 인코더 LM-swap 은 future work.", 14),
        ("**v_L10 은 1-d class-mean 축**: SAE / multi-axis decomposition 으로 "
         "pixel-encodable 한 *다른* 방향이 있을 수 있음 — LLaVA 에서도.", 14),
        ("**§4.6 cross-model 부분 검증**: LLaVA-Next / Idefics2 / InternVL3 "
         "는 M2 stim 위 v_L10 이 class-imbalanced (n_neg = 9, 5, 1) 라 "
         "per-model gradient ascent 안 함. 더 어려운 stim (M8a / M8c) 으로 "
         "검증 가능.", 14),
        ("**M5b 보류**: SIP / activation patching / SAE 메커니즘 분석 — "
         "다음 paper-section gap.", 14),
        ("**Single-task evaluation**: \"next-state-prediction\" 만 검증. "
         "다른 shortcut 행동 (counting / spatial / causality) 은 미검증.", 14),
        ("**Human baseline 미수집**: M7 Prolific (20 raters × 50 stim) 예산 "
         "있음, 미실시.", 14),
    ], font_name=FONT)
    return s


# ----- Section 7: Conclusion -----

def slide_25_conclusion(prs):
    s = new_slide(prs)
    y = add_title_bar(s, "결론 — 3 가지 paper-grade 기여",
                      subtitle="VLM 의 추상→물리 shortcut 을 행동 / 인과 / 픽셀 3차원으로 localize",
                      font_name=FONT)
    add_bullets(s, Inches(0.5), y + Inches(0.2), Inches(12.3), Inches(5.5), [
        ("**[기여 1] Architecture-level reframe**. 5 모델 × 3 stim source × "
         "bootstrap CI 로, 행동 PMR ceiling 이 encoder representational "
         "capacity *만으로* 결정되지 않음을 disconfirm. 2-CLIP-point insight "
         "(LLaVA-1.5 vs LLaVA-Next) 가 가장 깨끗한 disconfirmer.", 16),
        ("**[기여 2] 인과 localization**. Qwen2.5-VL 의 LM L10 단일 선형 "
         "방향 v_L10 이 forward-hook 으로 행동을 인과적으로 뒤집음 (10/10 "
         "α=40). M5a-ext: v_L10 은 regime axis (+ kinetic / − static), "
         "binary toggle 아님.", 16),
        ("**[기여 3] Pixel encodability — saturation 특이적**. Qwen "
         "ε=0.05 픽셀 perturbation 으로 5/5 PMR flip + 매칭 random 0/15. "
         "**§4.6 cross-model null**: 픽셀-인코드 가능성은 Qwen 의 saturated "
         "SigLIP 의 속성. M9 PMR-ceiling / §4.7 결정-안정성 ceiling 과 "
         "평행한 **세 번째 saturation 시그니처**.", 16),
    ], font_name=FONT)
    return s


def slide_26_qa(prs):
    s = new_slide(prs)
    add_text_box(s, Inches(0.5), Inches(2.5), Inches(12.3), Inches(1.2),
                 "감사합니다", size=60, bold=True, color=ACCENT,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(4.0), Inches(12.3), Inches(0.8),
                 "Q & A · 토론 · 다음 단계 우선순위",
                 size=22, color=GRAY_DARK, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.4),
                 "Repo: github.com/namam3gy/physical-mode-activation",
                 size=14, color=GRAY_MID, align=PP_ALIGN.CENTER, font_name=FONT)
    add_text_box(s, Inches(0.5), Inches(6.0), Inches(12.3), Inches(0.4),
                 "동반 자료: docs/review_ppt/physical_mode_paper_ko.md "
                 "(슬라이드별 상세 설명)",
                 size=12, color=GRAY_MID, align=PP_ALIGN.CENTER, font_name=FONT)
    return s


# ============================================================================

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        ("Title", slide_01_title),
        ("Motivation", slide_02_motivation),
        ("Question", slide_03_question),
        ("Contributions", slide_04_contributions),
        ("Related — shortcut", slide_05_related_shortcut),
        ("Related — probing", slide_06_related_probing),
        ("Problem & metrics", slide_07_problem),
        ("Stim design", slide_08_stim_design),
        ("Metrics", slide_09_metrics),
        ("Models", slide_10_models),
        ("Pipeline", slide_11_pipeline),
        ("Capture + probe", slide_12_capture_probe),
        ("Causal intervention", slide_13_causal_intervention),
        ("PMR ladder", slide_14_pmr_ladder),
        ("H1 ramp", slide_15_h1_ramp),
        ("H2 paired", slide_16_h2_paired),
        ("Encoder probes", slide_17_encoder_probes),
        ("M5a steering", slide_18_m5a_steering),
        ("§4.6 Qwen", slide_19_sec46_qwen),
        ("§4.6 cross-model null", slide_20_sec46_cross_null),
        ("M8 external validity", slide_21_external_validity),
        ("§4.3 multilingual", slide_22_multilingual),
        ("Discussion 1 — reframe", slide_23_architecture_reframe),
        ("Discussion 2 — limits", slide_24_limitations),
        ("Conclusion", slide_25_conclusion),
        ("Thanks / Q&A", slide_26_qa),
    ]

    total = len(builders)
    for i, (name, fn) in enumerate(builders, 1):
        s = fn(prs)
        if i > 1 and i < total:
            add_footer(s, i, total, FOOTER_TEXT, font_name=FONT)
        print(f"  [{i:2d}/{total}] {name}")

    out_dir = PROJECT_ROOT / "docs" / "review_ppt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "physical_mode_paper_ko.pptx"
    prs.save(str(out_path))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
