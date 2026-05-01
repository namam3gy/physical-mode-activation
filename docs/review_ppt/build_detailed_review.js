// Detailed comprehensive review deck — Physical-Mode Activation in VLMs
// 2026-05-01 from-scratch build. Targets ~45 slides covering: behavioral
// results, mechanistic chain, pixel-space, FAILURES (M-PSwap NaN, Qwen×MCQ
// NULL, LLaVA-A bias, zombie hang), Pillar B journey, M5b round 2 deep dive.
//
// Run: NODE_PATH=$(npm root -g) node docs/review_ppt/build_detailed_review.js

const PPTX = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "docs", "figures");
const OUT = path.join(__dirname, "physical_mode_detailed_review_ko.pptx");
const fig = (n) => path.join(FIG, n);

const C = {
  bgDark: "0E1E3D", bgLight: "F4F6FB",
  primary: "0B3D6B", primaryLight: "1F5F95",
  accent: "F4A23C", accentSoft: "FFE9CC",
  teal: "1C7293", ink: "1A2233", inkSoft: "4A5568",
  paper: "FFFFFF", divider: "C9D4E5",
  good: "2C7A4F", bad: "B5371C", warm: "C4621D",
  cardCircle: "E8EFF7", cardBall: "FFF0DE",
  fail: "8B1A1A", failSoft: "FBE5E5",
};
const FONT = "Calibri";

const pres = new PPTX();
pres.layout = "LAYOUT_WIDE";
pres.title = "원이 공으로 보이는 순간 — 종합 검토 (디테일 버전)";
pres.author = "thyun.park";

const W = 13.333, H = 7.5;
const TOTAL = 45;

function bg(s, c) { s.background = { color: c }; }
function pageNum(s, n, dark = false) {
  s.addText(`${n} / ${TOTAL}`, {
    x: W - 1.2, y: H - 0.45, w: 1.0, h: 0.3,
    fontSize: 10, fontFace: FONT,
    color: dark ? "B7C2D6" : C.inkSoft, align: "right",
  });
}
function footer(s, dark = false) {
  s.addText("원 → 공: VLM의 추상-물리 shortcut 종합 검토 (실험 디테일 + 실패 사례 포함)", {
    x: 0.5, y: H - 0.45, w: 11, h: 0.3,
    fontSize: 9.5, fontFace: FONT,
    color: dark ? "8FA0BD" : C.inkSoft, italic: true,
  });
}
function stripe(s, label, color = C.accent) {
  s.addShape("rect", { x: 0.5, y: 0.4, w: 0.16, h: 0.45,
    fill: { color }, line: { color } });
  s.addText(label, {
    x: 0.75, y: 0.38, w: 8, h: 0.5,
    fontSize: 11, fontFace: FONT, bold: true, color: C.primary, charSpacing: 4,
  });
}
function title(s, t, sub) {
  s.addText(t, {
    x: 0.5, y: 0.85, w: 12.3, h: 0.85,
    fontSize: 26, fontFace: FONT, bold: true, color: C.primary,
  });
  if (sub) s.addText(sub, {
    x: 0.5, y: 1.6, w: 12.3, h: 0.5,
    fontSize: 13, fontFace: FONT, color: C.teal, italic: true,
  });
  s.addShape("rect", { x: 0.5, y: 2.1, w: 1.0, h: 0.04,
    fill: { color: C.accent }, line: { color: C.accent } });
}
function content() { const s = pres.addSlide(); bg(s, C.bgLight); return s; }
function box(s, x, y, w, h, opts) {
  s.addShape(opts.round ? "roundRect" : "rect", {
    x, y, w, h, rectRadius: opts.round || 0,
    fill: { color: opts.fill || C.paper },
    line: { color: opts.line || C.divider, width: opts.lw || 1 },
  });
  if (opts.title) s.addText(opts.title, {
    x: x + 0.2, y: y + 0.1, w: w - 0.4, h: 0.4,
    fontSize: opts.titleSize || 13, fontFace: FONT, bold: true,
    color: opts.titleColor || C.primary,
  });
}
function table(s, x, y, w, headers, rows, colW, opts = {}) {
  const fs = opts.fontSize || 10;
  const rowH = opts.rowH || 0.4;
  // header
  let cx = x;
  headers.forEach((h, i) => {
    s.addText(h, {
      x: cx, y: y, w: colW[i], h: 0.35,
      fontSize: fs, fontFace: FONT, bold: true, color: C.primary,
    });
    cx += colW[i];
  });
  s.addShape("rect", {
    x: x, y: y + 0.32, w: colW.reduce((a, b) => a + b), h: 0.02,
    fill: { color: C.divider }, line: { color: C.divider },
  });
  rows.forEach((r, i) => {
    cx = x;
    const ry = y + 0.4 + i * rowH;
    r.forEach((cell, j) => {
      const cellOpts = typeof cell === "object" ? cell : { text: cell };
      s.addText(cellOpts.text, {
        x: cx, y: ry, w: colW[j], h: rowH,
        fontSize: cellOpts.fontSize || fs - 0.5, fontFace: FONT,
        bold: cellOpts.bold || false,
        color: cellOpts.color || C.ink,
      });
      cx += colW[j];
    });
  });
}

// ========================================================================
// SLIDE 1 — Title
// ========================================================================
{
  const s = pres.addSlide();
  bg(s, C.bgDark);
  s.addShape("rect", { x: 0, y: 0, w: W, h: 0.18,
    fill: { color: C.accent }, line: { color: C.accent } });

  s.addText("원이 공으로 보이는 순간", {
    x: 0.5, y: 1.2, w: W - 1, h: 1.1,
    fontSize: 48, fontFace: FONT, bold: true, color: "FFFFFF", align: "center",
  });
  s.addText("VLM의 추상→물리 shortcut을 행동·메커니즘·픽셀 수준에서 분석", {
    x: 0.5, y: 2.45, w: W - 1, h: 0.5,
    fontSize: 18, fontFace: FONT, italic: true, color: "DEE6F2", align: "center",
  });
  s.addShape("rect", { x: W/2 - 0.6, y: 3.15, w: 1.2, h: 0.04,
    fill: { color: C.accent }, line: { color: C.accent } });

  s.addText("종합 검토 (디테일 버전)", {
    x: 0.5, y: 3.5, w: W - 1, h: 0.5,
    fontSize: 24, fontFace: FONT, bold: true, color: C.accent, align: "center",
  });
  s.addText([
    { text: "본 deck 은 ", options: { fontSize: 13 } },
    { text: "M0–M9 + §4 series + Pillar A + Pillar B (M-LMSwap, M-PSwap) + Post-proj round 2 + B5 (Qwen 32B)", options: { fontSize: 13, bold: true, color: C.accent } },
    { text: " 까지의 모든 실험을 ", options: { fontSize: 13 } },
    { text: "성공·실패·디테일", options: { fontSize: 13, bold: true, color: "FFFFFF" } },
    { text: " 모두 다룹니다.", options: { fontSize: 13 } },
  ], {
    x: 1.0, y: 4.2, w: W - 2, h: 0.6,
    fontFace: FONT, color: "DEE6F2", align: "center",
  });

  s.addText("5개 오픈소스 비전-언어 모델 비교 + Vicuna-vs-Mistral 통제 실험", {
    x: 0.5, y: 5.0, w: W - 1, h: 0.4,
    fontSize: 14, fontFace: FONT, color: "B7C2D6", align: "center",
  });
  s.addText("Qwen2.5-VL-7B/32B · LLaVA-1.5 · LLaVA-Next · Idefics2 · InternVL3  +  M-LMSwap A/B (CLIP+Vicuna / CLIP+Mistral)", {
    x: 0.5, y: 5.4, w: W - 1, h: 0.4,
    fontSize: 12, fontFace: FONT, italic: true, color: "8FA0BD", align: "center",
  });

  s.addText("2026년 5월 1일 · thyun.park · 45 슬라이드", {
    x: 0.5, y: 6.7, w: W - 1, h: 0.3,
    fontSize: 11, fontFace: FONT, color: "8FA0BD", align: "center", italic: true,
  });
  pageNum(s, 1, true);
}
// ========================================================================
// SLIDE 2 — The Hook
// ========================================================================
{
  const s = content();
  stripe(s, "S T O R Y   1 — 놀라운 관찰");
  title(s, "사람이 보면 \"그냥 동그라미\"", "그런데 AI는 \"공이 떨어진다\"라고 답한다.");

  s.addImage({ path: fig("01_line_blank_none.png"), x: 0.6, y: 2.6, w: 3.4, h: 3.4 });
  s.addText("Figure 1.  baseline 자극 (line · blank · 단서 없음)\n중력·지면·텍스처·움직임 단서 모두 0", {
    x: 0.4, y: 6.0, w: 3.8, h: 0.6,
    fontSize: 10, fontFace: FONT, color: C.inkSoft, italic: true, align: "center",
  });

  box(s, 4.5, 2.6, 4.0, 1.5, { round: 0.08, fill: C.cardCircle, line: C.divider, title: "사람의 응답" });
  s.addText("\"흰 배경 위 검은 원이 그려져 있다.\"\n→ 추상적 도형. 운동 단서 없음.", {
    x: 4.7, y: 3.05, w: 3.6, h: 1.0, fontSize: 12, fontFace: FONT, color: C.ink });

  box(s, 8.7, 2.6, 4.2, 1.5, { round: 0.08, fill: C.cardBall, line: C.warm, title: "Qwen2.5-VL의 응답", titleColor: C.warm });
  s.addText("\"The ball will fall down due to gravity.\"\n→ 갑자기 \"공\"·\"중력\"·\"낙하\".", {
    x: 8.9, y: 3.05, w: 4.0, h: 1.0, fontSize: 12, fontFace: FONT, color: C.ink });

  box(s, 4.5, 4.3, 8.4, 2.2, { round: 0.08, fill: C.paper, line: C.accent, lw: 2, title: "관찰 — 무엇이 이상한가?", titleColor: C.accent });
  s.addText([
    { text: "1. 이미지에 ", options: { fontSize: 12 } },
    { text: "중력 단서·지면·텍스처가 전혀 없다", options: { fontSize: 12, bold: true, color: C.bad } },
    { text: ".\n2. 모델은 추상↔물리 두 해석 중 ", options: { fontSize: 12 } },
    { text: "한쪽으로만 무너진다 (collapse)", options: { fontSize: 12, bold: true, color: C.bad } },
    { text: ".\n3. 이런 ", options: { fontSize: 12 } },
    { text: "shortcut", options: { fontSize: 12, bold: true } },
    { text: "은 알려져 있지만 ", options: { fontSize: 12 } },
    { text: "어디서·왜 일어나는지는 미해결", options: { fontSize: 12, italic: true } },
    { text: ".\n4. 모델별로 강도가 다르다 — ", options: { fontSize: 12 } },
    { text: "PMR_nolabel: LLaVA-1.5 0.18 → InternVL3 0.99 (5.5×)", options: { fontSize: 12, bold: true, color: C.warm } },
    { text: ".", options: { fontSize: 12 } },
  ], { x: 4.7, y: 4.75, w: 8.0, h: 1.7, fontFace: FONT, color: C.ink, paraSpaceAfter: 4 });

  s.addText("프롬프트:  \"What do you see in this image, and what might happen next?\"  · M2 stim (n=480, 4×3×4×10 factorial)", {
    x: 0.5, y: 6.7, w: 12.5, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.inkSoft, italic: true });
  footer(s); pageNum(s, 2);
}

// ========================================================================
// SLIDE 3 — Project journey timeline
// ========================================================================
{
  const s = content();
  stripe(s, "S T O R Y   2 — 프로젝트 여정");
  title(s, "8일간의 실험 여정 (2026-04-24 → 2026-05-01)",
    "milestone × 실패 × 분기 결정을 시간순으로 한눈에.");

  // Vertical timeline
  const events = [
    { date: "04-24", color: C.good, label: "M0–M5a 한 번에 통과", desc: "Infra → M1 pilot → M2 MVP-full → M3 encoder → M4 LM → M5a VTI L10 α=40 10/10 flip" },
    { date: "04-25", color: C.good, label: "M5a-ext + M6 r1 + M8 외부 타당성", desc: "regime axis (±α) 확인, LLaVA-1.5 floor 발견 (PMR 0.18), M8a/M8d/M8c (5도형/3카테고리/실사진)" },
    { date: "04-26", color: C.primary, label: "M5b SIP + 5-model M2 cross-model + §4.6 Qwen pixel", desc: "L9 MLP IE=+1.0, 5-model PMR ladder 0.18→0.99, 픽셀 v_L10 ascent 5/5 flip" },
    { date: "04-27", color: C.primary, label: "M5b per-head knockout + §4.6 5-model n=10 sweep", desc: "196 (L,h) cells all IE=0, Idefics2 0/9 layers anomaly emerges, SAE Cohen's d ranking" },
    { date: "04-28", color: C.warm, label: "Audit + 5-model 시그니처 lock", desc: "M5a cross 3/4 flip, M5b SAE 5-model (3 break, 2 NULL), §4.8 7B vs 32B, M4 LM probe AUC ladder" },
    { date: "04-29", color: C.fail, label: "M-PSwap NaN 미해결 → backlog. Pillar B M-LMSwap 채택", desc: "Day-0 + canonical 2-stage Option 1, scaffold 작성, smoke 50-step PASS" },
    { date: "04-30", color: C.warm, label: "Variant A Stage 1 + Stage 2 학습 (~24h)", desc: "step1000 → step21000 LCS-558K + LLaVA-Instruct-665K LoRA, 학습 NaN 없음" },
    { date: "05-01", color: C.accent, label: "Regression split + post-proj round 2 + B5 Qwen 32B", desc: "step9000 gate FAIL, step21000 baseline=0.000 PASS, regime-cross ladder 5-model, 32B post-proj k=40 break" },
  ];

  let y = 2.5;
  events.forEach((e, i) => {
    s.addShape("rect", { x: 0.7, y: y, w: 0.7, h: 0.5,
      fill: { color: e.color }, line: { color: e.color } });
    s.addText(e.date, { x: 0.7, y: y, w: 0.7, h: 0.5,
      fontSize: 11, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(e.label, { x: 1.55, y: y, w: 5.0, h: 0.25,
      fontSize: 11.5, fontFace: FONT, bold: true, color: C.ink });
    s.addText(e.desc, { x: 1.55, y: y + 0.22, w: 11.5, h: 0.3,
      fontSize: 9.5, fontFace: FONT, color: C.inkSoft, italic: true });
    y += 0.55;
  });

  s.addText("색상 — 초록=성공, 파랑=메커니즘, 주황=warning/audit, 빨강=실패, 액센트=현재 진행", {
    x: 0.5, y: 7.0, w: 12.5, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.inkSoft, italic: true });
  footer(s); pageNum(s, 3);
}

// ========================================================================
// SLIDE 4 — 3-axis question + measurement positions
// ========================================================================
{
  const s = content();
  stripe(s, "S T O R Y   3 — 3축 질문");
  title(s, "WHEN · WHERE · HOW — 같은 자극·같은 5 모델로 동시에 답한다",
    "행동 (PMR/GAR/RC) → 메커니즘 (probe/steer/ablate) → 픽셀 (gradient ascent).");

  const axes = [
    { tag: "WHEN", ko: "언제 (행동)", color: C.primary,
      tools: "PMR · GAR · RC · paired-Δ · open vs FC · _nolabel",
      results: "5-model PMR ladder 0.18 → 0.99\nH1 ramp / H2 3 patterns / M9 robust" },
    { tag: "WHERE", ko: "어디서 (메커니즘)", color: C.teal,
      tools: "linear probe · logit lens · VTI steering · SAE ablation · SIP patching · MLP/attention knockout",
      results: "L10 α=40 flip 10/10 / L9 MLP IE=+1.0\n~30 SAE feature break / 196 (L,h)=0" },
    { tag: "HOW", ko: "어떻게 (픽셀)", color: C.warm,
      tools: "v_L gradient ascent on pixels · matched-mass random control",
      results: "Qwen broad shortcut / LLaVA-Next L20+L25\nIdefics2 0/9 → perceiver candidate" },
  ];

  axes.forEach((a, i) => {
    const x0 = 0.5 + i * 4.3;
    box(s, x0, 2.7, 4.0, 4.3, { round: 0.1, fill: C.paper, line: a.color, lw: 2 });
    s.addShape("rect", { x: x0, y: 2.7, w: 4.0, h: 0.55,
      fill: { color: a.color }, line: { color: a.color } });
    s.addText(a.tag, { x: x0 + 0.15, y: 2.72, w: 1.5, h: 0.5,
      fontSize: 16, fontFace: FONT, bold: true, color: "FFFFFF", charSpacing: 4 });
    s.addText(a.ko, { x: x0 + 1.7, y: 2.78, w: 2.2, h: 0.4,
      fontSize: 14, fontFace: FONT, bold: true, color: "FFFFFF", align: "right" });

    s.addText("도구", { x: x0 + 0.2, y: 3.4, w: 3.6, h: 0.3,
      fontSize: 11, fontFace: FONT, bold: true, color: a.color });
    s.addText(a.tools, { x: x0 + 0.2, y: 3.7, w: 3.6, h: 1.4,
      fontSize: 11, fontFace: FONT, color: C.ink });

    s.addText("주요 결과", { x: x0 + 0.2, y: 5.1, w: 3.6, h: 0.3,
      fontSize: 11, fontFace: FONT, bold: true, color: a.color });
    s.addText(a.results, { x: x0 + 0.2, y: 5.4, w: 3.6, h: 1.5,
      fontSize: 11, fontFace: FONT, color: C.ink });
  });

  footer(s); pageNum(s, 4);
}

// ========================================================================
// SLIDE 5 — VLM pipeline + 5 model zoo
// ========================================================================
{
  const s = content();
  stripe(s, "M E T H O D — V L M   p i p e l i n e");
  title(s, "VLM 파이프라인 + 5-model zoo",
    "측정은 5단계 각각에서: 인코더 (probe/SAE) → projector (post-proj SAE) → LM (logit lens/steering/SIP) → 응답 (PMR).");

  // pipeline boxes
  const stages = [
    { x: 0.5, label: "이미지", color: C.inkSoft },
    { x: 2.7, label: "Vision Encoder", color: C.teal },
    { x: 5.1, label: "Projector", color: C.primaryLight },
    { x: 7.5, label: "Language Model", color: C.primary },
    { x: 10.1, label: "응답 (text)", color: C.warm },
  ];
  stages.forEach((st, i) => {
    s.addShape("roundRect", { x: st.x, y: 2.65, w: 2.0, h: 0.7, rectRadius: 0.05,
      fill: { color: st.color }, line: { color: st.color } });
    s.addText(st.label, { x: st.x, y: 2.65, w: 2.0, h: 0.7,
      fontSize: 12, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    if (i < stages.length - 1) {
      s.addText("→", { x: st.x + 2.0, y: 2.65, w: 0.2, h: 0.7,
        fontSize: 18, fontFace: FONT, bold: true, color: C.ink, align: "center", valign: "middle" });
    }
  });
  // measurement labels under each stage
  const meas = [
    { x: 2.7, t: "M3 probe AUC\nM5b encoder SAE" },
    { x: 5.1, t: "M5b post-proj SAE\n(round 2)" },
    { x: 7.5, t: "M4 logit lens\nM5a VTI · M5b SIP\nMLP/head knockout" },
    { x: 10.1, t: "PMR · GAR · RC\nopen vs FC · _nolabel" },
  ];
  meas.forEach((m) => {
    s.addText(m.t, { x: m.x, y: 3.45, w: 2.0, h: 0.6,
      fontSize: 9.5, fontFace: FONT, italic: true, color: C.inkSoft, align: "center" });
  });

  // 5 model table
  table(s, 0.5, 4.4,
    12.3,
    ["모델", "Vision Encoder", "Projector", "LM", "PMR_nolabel", "역할"],
    [
      ["Qwen2.5-VL-7B/32B", "SigLIP-400M / SO400M", "merger MLP", "Qwen2", "0.94", "메인 — 인과 anchor (B5: 32B 추가)"],
      ["LLaVA-1.5-7B", "CLIP-ViT-L-336", "2-layer MLP", "Vicuna-7B", "0.18", "Floor — S-curve 가장 깨끗 / encoder-NULL"],
      ["LLaVA-Next-7B", "CLIP-ViT-L-336 AnyRes", "2-layer MLP", "Mistral-7B", "0.79", "Mid — LM-side flip / encoder-NULL"],
      ["Idefics2-8B", "SigLIP-SO400M", { text: "perceiver-resampler", color: C.warm, bold: true }, "Mistral-7B", "0.97", "유일 perceiver — §4.6 0/9 의 핵심"],
      ["InternVL3-8B-hf", "InternViT-300M", "MLP (pixel-shuffle)", "InternLM3-8B", "0.99", "비-CLIP 비교점 (super-saturated)"],
      ["M-LMSwap A (학습완료)", "CLIP-ViT-L-336", "2-layer MLP (fresh)", { text: "Vicuna-7B", color: C.accent, bold: true }, "0.87 (드리프트)", "Pillar B 통제 비교축 1"],
      ["M-LMSwap B (대기중)", "CLIP-ViT-L-336", "2-layer MLP (fresh)", { text: "Mistral-7B-Instruct", color: C.accent, bold: true }, "(미학습)", "Pillar B 통제 비교축 2 — A↔B Δ-PMR"],
    ],
    [2.5, 2.4, 2.0, 1.7, 1.4, 2.3], { fontSize: 10, rowH: 0.36 });

  s.addText("핵심 비교쌍 — LLaVA-1.5 vs LLaVA-Next: 같은 CLIP인데 PMR 0.18 ↔ 0.79.  Pillar B 의 동기. (슬라이드 34)", {
    x: 0.5, y: 7.0, w: 12.5, h: 0.3,
    fontSize: 11, fontFace: FONT, italic: true, color: C.warm });
  footer(s); pageNum(s, 5);
}

// ========================================================================
// SLIDE 6 — Stim design: abstraction ladder + 5-axis factorial
// ========================================================================
{
  const s = content();
  stripe(s, "M E T H O D — s t i m u l u s   d e s i g n");
  title(s, "Stim 디자인 — \"같은 동그라미를 다른 옷\" 입혀서 측정",
    "axis A: 추상화 4단계 · axis B: 배경 3단계 · axis C: 단서 4단계 · axis D: 라벨 4단계 · seeds 10 = 1,440 ~ 2,880 stim/모델.");

  // Abstraction ladder visual
  box(s, 0.5, 2.55, 12.3, 1.6, { round: 0.08, fill: C.paper, line: C.divider, title: "axis A — 추상화 사다리 (왼쪽 추상 ↔ 오른쪽 물리)" });
  const ladImgs = [
    { f: "01_line_blank_none.png", t: "line — 선만" },
    { f: "05_filled_blank_wind.png", t: "filled — 회색 채움" },
    { f: "03_shaded_ground_none.png", t: "shaded — 3D 셰이딩" },
    { f: "04_textured_ground_arrow_shadow.png", t: "textured — 가죽/점박이" },
  ];
  ladImgs.forEach((im, i) => {
    s.addImage({ path: fig(im.f), x: 0.8 + i * 3.0, y: 2.95, w: 1.0, h: 1.0 });
    s.addText(im.t, { x: 0.8 + i * 3.0, y: 4.0, w: 2.5, h: 0.2,
      fontSize: 10, fontFace: FONT, bold: true, color: C.ink });
  });

  // Factorial table
  box(s, 0.5, 4.3, 6.0, 2.7, { round: 0.08, fill: C.cardCircle, line: C.primary, title: "5-axis factorial 정의" });
  s.addText([
    { text: "object_level (A) ", options: { fontSize: 10.5, bold: true } },
    { text: "× 4:  line / filled / shaded / textured\n", options: { fontSize: 10.5 } },
    { text: "bg_level (B) ", options: { fontSize: 10.5, bold: true } },
    { text: "× 3:  blank / ground / scene\n", options: { fontSize: 10.5 } },
    { text: "cue_level (C) ", options: { fontSize: 10.5, bold: true } },
    { text: "× 4:  none / cast_shadow / motion_arrow / both\n", options: { fontSize: 10.5 } },
    { text: "event_template ", options: { fontSize: 10.5, bold: true } },
    { text: "× 3:  fall / horizontal / rise\n", options: { fontSize: 10.5 } },
    { text: "label (D) ", options: { fontSize: 10.5, bold: true } },
    { text: "× 4:  circle / ball / planet / _nolabel\n", options: { fontSize: 10.5 } },
    { text: "seeds_per_cell ", options: { fontSize: 10.5, bold: true } },
    { text: "× 10\n", options: { fontSize: 10.5 } },
    { text: "\n총합:  4×3×4×3×10 × 4-label = ", options: { fontSize: 10.5 } },
    { text: "2,880 추론/모델", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: " (M2)", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 4.7, w: 5.6, h: 2.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  // External validity
  box(s, 6.7, 4.3, 6.1, 2.7, { round: 0.08, fill: C.accentSoft, line: C.warm, title: "외부 타당성 확장 (M8 · §4)" });
  s.addText([
    { text: "M8a — ", options: { fontSize: 10.5, bold: true } },
    { text: "5 도형 (circle/square/triangle/hexagon/polygon) × 4 추상화\n", options: { fontSize: 10.5 } },
    { text: "M8d — ", options: { fontSize: 10.5, bold: true } },
    { text: "3 카테고리 (car/person/bird) — 비-공 H7 검증\n", options: { fontSize: 10.5 } },
    { text: "M8c — ", options: { fontSize: 10.5, bold: true } },
    { text: "60 실사진 (COCO + WikiArt) — 인코더 격차 압축 효과\n", options: { fontSize: 10.5 } },
    { text: "M8e — ", options: { fontSize: 10.5, bold: true } },
    { text: "cross-source consolidation\n", options: { fontSize: 10.5 } },
    { text: "§4.3 — ", options: { fontSize: 10.5, bold: true } },
    { text: "한국어/일본어/중국어 라벨 (5-model)\n", options: { fontSize: 10.5 } },
    { text: "§4.6 — ", options: { fontSize: 10.5, bold: true } },
    { text: "픽셀 공간 counterfactual stim 생성\n", options: { fontSize: 10.5 } },
    { text: "§4.8 — ", options: { fontSize: 10.5, bold: true } },
    { text: "Qwen 7B vs 32B scaling (open + FC)", options: { fontSize: 10.5 } },
  ], { x: 6.9, y: 4.7, w: 5.7, h: 2.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 6);
}

// ========================================================================
// SLIDE 7 — Metrics
// ========================================================================
{
  const s = content();
  stripe(s, "M E T H O D — m e t r i c s");
  title(s, "측정 지표 — PMR / GAR / RC / paired-Δ / v_L direction",
    "응답 텍스트 위에서 자동 채점한 6개 지표 (사람 검수 ~5 % 불일치). 최근 발견: binary PMR 의 한계.");

  table(s, 0.5, 2.6, 12.3,
    ["지표", "정의", "어디 쓰는가", "주의사항"],
    [
      [{ text: "PMR", bold: true, color: C.primary }, "응답에 falls / rolls / bounces 등 물리 동사가 들어간 비율", "메인 지표 — 모든 행동·메커니즘 결과의 y축", "lexicon stem 매칭 (\"continu\" 등 false positive 가능)"],
      [{ text: "PMR(_nolabel)", bold: true, color: C.primary }, "라벨 없는 open-ended 프롬프트의 PMR", "모델 자체 편향 측정 (label confound 제거)", "M9 bootstrap CI 의 핵심 입력"],
      [{ text: "GAR", bold: true, color: C.teal }, "Gravity-Align Rate — 물리 응답 중 \"하방으로\" 답한 비율", "중력 정합도 — H7 라벨-regime 분석", "ball/circle/planet GAR 패턴이 H7 evidence"],
      [{ text: "RC", bold: true, color: C.teal }, "Response Consistency — T=0.7 N seed PMR call 일관도", "결정 안정성 (§4.7 axis별 RC)", "n=5 seed × 5-model bootstrap"],
      [{ text: "H2 paired-Δ", bold: true, color: C.warm }, "PMR(label) − PMR(_nolabel)", "라벨 prior 가 PMR 을 얼마나 끌고 가는지", "3 모델별 부호 패턴 (positive/asymmetric/0)"],
      [{ text: "v_L direction", bold: true, color: C.warm }, "L 레이어 hidden state 의 mean(physics=1) − mean(physics=0)", "VTI steering · SAE feature ranking · 픽셀 ascent", "saturated 모델은 n_neg<5 → 안 잡힘"],
    ],
    [2.0, 4.5, 3.3, 2.5], { fontSize: 10, rowH: 0.55 });

  box(s, 0.5, 6.0, 12.3, 1.0, { round: 0.08, fill: C.failSoft, line: C.fail, lw: 1, title: "최근 발견 — binary PMR 의 한계 (M5b round 2, 2026-05-01)", titleColor: C.fail });
  s.addText("Idefics2 post-proj k=20 → \"disappear\" (PMR=0), k=40+ → \"continue to expand outward\" (PMR=1, BUT \"continu\" stem 매칭으로 인한 scoring artifact). " +
    "Text 는 fall→hit→roll 로 명확히 이동하지만 binary PMR 은 모든 motion verb 를 동등 점수.  " +
    "→ paper draft 에 regime-shift score (text-distance) 를 PMR 보조 지표로 추가 권장.", {
    x: 0.7, y: 6.4, w: 11.9, h: 0.55,
    fontSize: 10, fontFace: FONT, color: C.ink, italic: true });

  footer(s); pageNum(s, 7);
}

// ========================================================================
// SLIDE 8 — M2 PMR ladder (5-model)
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #1 — P M R   l a d d e r");
  title(s, "M2 PMR(_nolabel) 사다리 — 같은 480 자극, 5.5 × 격차",
    "인코더만으로 설명 안 되는 격차. 같은 CLIP-ViT-L 안에서도 LLaVA-1.5 0.18 vs LLaVA-Next 0.79.");

  s.addImage({ path: fig("m2_cross_model_pmr_ladder.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  table(s, 8.3, 2.6, 4.7,
    ["모델", "PMR_nolabel [95% CI]", "Tier"],
    [
      ["LLaVA-1.5", "0.18 [0.14, 0.21]", { text: "Floor", bold: true, color: C.good }],
      ["LLaVA-Next", "0.79 [0.75, 0.83]", { text: "Mid", bold: true, color: C.warm }],
      ["Qwen2.5-VL", "0.94 [0.92, 0.96]", { text: "Saturated", bold: true, color: C.bad }],
      ["Idefics2", "0.97 [0.95, 0.98]", { text: "Saturated", bold: true, color: C.bad }],
      ["InternVL3", "0.99 [0.98, 1.00]", { text: "Super", bold: true, color: C.fail }],
    ],
    [1.6, 2.0, 1.1], { fontSize: 11, rowH: 0.45 });

  box(s, 8.3, 5.0, 4.7, 2.1, { round: 0.08, fill: C.cardCircle, line: C.primary, title: "왜 인코더 단독 결정자가 아닌가" });
  s.addText([
    { text: "1. ", options: { fontSize: 10 } },
    { text: "같은 CLIP-ViT-L-336 ", options: { fontSize: 10, bold: true } },
    { text: "인코더에서 PMR 0.18 ↔ 0.79.\n", options: { fontSize: 10 } },
    { text: "2. M3 stim-defined Y 로 측정 시 ", options: { fontSize: 10 } },
    { text: "5 인코더 모두 AUC = 1.0", options: { fontSize: 10, bold: true, color: C.teal } },
    { text: ".\n3. 즉 인코더 ", options: { fontSize: 10 } },
    { text: "표현 능력은 균일", options: { fontSize: 10, bold: true } },
    { text: ", 차이는 LM 의 \"읽는 방식\".\n4. → ", options: { fontSize: 10 } },
    { text: "Encoder knows, decoder gates", options: { fontSize: 10, bold: true, italic: true, color: C.warm } },
    { text: " (boomerang).", options: { fontSize: 10 } },
  ], { x: 8.5, y: 5.45, w: 4.3, h: 1.6, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 8);
}

// ========================================================================
// SLIDE 9 — H1 ramp 5-model
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #2 — H 1   r a m p");
  title(s, "H1 — 추상화 사다리에 따른 PMR S-curve (모델별)",
    "LLaVA-1.5 만 깨끗한 +0.30 ramp.  나머지 4 모델은 첫 단계부터 천장에 붙어 측정 헤드룸 없음.");

  s.addImage({ path: fig("m2_cross_model_h1_ramp.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  table(s, 8.3, 2.6, 4.7,
    ["모델", "line", "textured", "Δ"],
    [
      ["LLaVA-1.5", "0.51", "0.81", { text: "+0.30", bold: true, color: C.good }],
      ["LLaVA-Next", "0.65", "0.79", { text: "+0.14", color: C.warm }],
      ["Idefics2", "0.88", "0.97", { text: "+0.09", color: C.bad }],
      ["Qwen", "0.89", "0.94", { text: "+0.05", color: C.bad }],
      ["InternVL3", "0.97", "0.99", { text: "+0.02", color: C.fail }],
    ],
    [1.5, 0.9, 1.2, 1.0], { fontSize: 11, rowH: 0.45 });

  box(s, 8.3, 5.0, 4.7, 2.1, { round: 0.08, fill: C.accentSoft, line: C.warm, title: "관측" });
  s.addText("• Ramp 측정성은 ", { x: 8.5, y: 5.45, w: 4.3, h: 0.3, fontSize: 10, fontFace: FONT });
  s.addText([
    { text: "encoder saturation 과 반비례", options: { fontSize: 10, bold: true } },
    { text: " — saturated 모델은 천장 효과로 H1 검증 불가.\n• ", options: { fontSize: 10 } },
    { text: "M8a (5 도형) ", options: { fontSize: 10, bold: true } },
    { text: "에서도 동일 패턴 — Qwen 3/5 fail (square/triangle 천장), LLaVA 4/5.\n• → \"H1 은 ", options: { fontSize: 10 } },
    { text: "unsaturated-only AND shape-axis-only", options: { fontSize: 10, italic: true, color: C.warm } },
    { text: "\" 로 정밀화 (H 카드 수정).", options: { fontSize: 10 } },
  ], { x: 8.5, y: 5.65, w: 4.3, h: 1.4, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 9);
}

// ========================================================================
// SLIDE 10 — H2 paired-Δ 3 patterns
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #3 — H 2   3   p a t t e r n s");
  title(s, "H2 paired-Δ — 라벨 효과의 부호 자체가 architecture-conditional",
    "LLaVA family: 모든 라벨 양수 (classical H2). Qwen/Idefics2: 비-물리 라벨이 baseline 아래로 (\"circle override\"). InternVL3: ≈ 0.");

  s.addImage({ path: fig("m2_cross_model_h2_paired_delta.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  box(s, 8.3, 2.6, 4.7, 4.5, { round: 0.08, fill: C.paper, line: C.divider, title: "3가지 architecture-conditional 패턴" });
  s.addText([
    { text: "① Unsaturated CLIP — ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "LLaVA-1.5 / Next\n", options: { fontSize: 11 } },
    { text: "  ball Δ = +0.475, planet +0.244, circle +0.173\n", options: { fontSize: 10 } },
    { text: "  → 모든 라벨이 PMR 을 끌어올림 (classical H2)\n\n", options: { fontSize: 10, italic: true } },

    { text: "② Saturated SigLIP — ", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "Qwen / Idefics2\n", options: { fontSize: 11 } },
    { text: "  ball Δ ≈ 0, planet/circle 음수 (baseline 아래로 억제)\n", options: { fontSize: 10 } },
    { text: "  → \"circle override\" — 비-물리 라벨이 PMR 을 깎음\n\n", options: { fontSize: 10, italic: true } },

    { text: "③ Super-saturated — ", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "InternVL3\n", options: { fontSize: 11 } },
    { text: "  모든 라벨 Δ ≈ 0 (천장 효과)\n", options: { fontSize: 10 } },
    { text: "  → 라벨이 더 끌어올릴 헤드룸이 없음\n\n", options: { fontSize: 10, italic: true } },

    { text: "교훈:  H2 는 \"라벨이 항상 PMR 을 더한다\" 가 아님.\n", options: { fontSize: 10 } },
    { text: "라벨 효과의 부호 자체가 ", options: { fontSize: 10, italic: true } },
    { text: "encoder saturation 상태에 따라 결정", options: { fontSize: 10, italic: true, bold: true, color: C.warm } },
    { text: ".", options: { fontSize: 10 } },
  ], { x: 8.5, y: 3.05, w: 4.3, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 10);
}

// ========================================================================
// SLIDE 11 — M9 generalization audit
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #4 — M 9   r o b u s t n e s s");
  title(s, "M9 generalization audit — 같은 클러스터링이 자극을 바꿔도 보존",
    "3 모델 × 3 자극 source × 5000-iter bootstrap CI. 합성 도형 / 사진 / 카테고리에서 동일 분리 패턴.");

  table(s, 0.5, 2.6, 12.3,
    ["자극 source", "non-CLIP cluster", "CLIP-LLaVA-1.5", "분리 여부"],
    [
      ["M8a 5-도형 합성", "[0.80, 0.92]", "[0.14, 0.21]", { text: "완전 분리", bold: true, color: C.good }],
      ["M8d 3-카테고리 (car/person/bird)", "[0.84, 0.92]", "[0.18, 0.36]", { text: "완전 분리", bold: true, color: C.good }],
      ["M8c 60 실사진", "[0.28, 0.55]", "[0.18, 0.42]", { text: "수렴 (overlap)", bold: true, color: C.warm }],
    ],
    [3.0, 2.6, 2.6, 4.1], { fontSize: 11, rowH: 0.5 });

  box(s, 0.5, 4.3, 12.3, 2.7, { round: 0.08, fill: C.cardCircle, line: C.primary, title: "발견 + H7 evidence" });
  s.addText([
    { text: "1. 합성 자극 (M8a/M8d): ", options: { fontSize: 11, bold: true } },
    { text: "non-CLIP vs CLIP cluster 가 95% CI 로 완전 분리.\n", options: { fontSize: 11 } },
    { text: "2. 사진 (M8c): ", options: { fontSize: 11, bold: true } },
    { text: "모든 모델 [0.18, 0.67] 로 수렴 — \"풍부한 cue\" 가 saturation 천장을 깨뜨림.\n", options: { fontSize: 11 } },
    { text: "3. ", options: { fontSize: 11, bold: true } },
    { text: "사진은 라벨 효과 (H7) 도 절반으로 — synthetic stim 의 minimality 가 라벨-regime selection의 co-factor.\n", options: { fontSize: 11 } },
    { text: "4. M8d H7 — ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "LLaVA car +0.525, person +0.138, bird +0.550 ", options: { fontSize: 11 } },
    { text: "(PMR_regime physical−abstract).\n", options: { fontSize: 11 } },
    { text: "  → 본 프로젝트의 ", options: { fontSize: 11, italic: true } },
    { text: "가장 강한 cross-category H7 evidence", options: { fontSize: 11, italic: true, bold: true, color: C.good } },
    { text: ". \"라벨이 regime 을 선택한다\" 는 카테고리-general claim 으로 일반화.", options: { fontSize: 11, italic: true } },
  ], { x: 0.7, y: 4.75, w: 11.9, h: 2.1, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 11);
}

// ========================================================================
// SLIDE 12 — §4.8 Qwen 7B vs 32B + B5.1 today
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #5 — § 4 . 8   s c a l i n g");
  title(s, "Qwen 7B vs 32B PMR scaling — \"scale doesn't fix grounding\"",
    "5× 모델 키워도 aggregate PMR 안 움직임 (MechBench-style). 단, cue=none 셀에서만 scale 이 도움.  + 오늘 B5.1: 32B FC label-free PMR 0.873.");

  table(s, 0.5, 2.6, 12.3,
    ["지표", "Qwen 7B", "Qwen 32B", "Δ", "해석"],
    [
      ["aggregate PMR (open, M2)", "0.931", "0.926", "−0.005", "5× scaling 으로 변화 없음"],
      ["cue=none PMR", "0.797", "0.711", { text: "−8.6 pp", bold: true, color: C.good }, "32B 가 weak-cue 에서 더 잘 abstain"],
      ["abstract_reject rate", "0.002", "0.065", { text: "35×", bold: true, color: C.good }, "32B 가 \"이건 그림이다\" 로 응답"],
      ["H2 (ball − circle)", "+0.071", "+0.010", { text: "halved", bold: true, color: C.warm }, "라벨 prior 효과 약화"],
      [{ text: "FC PMR_nolabel (B5.1 today)", color: C.accent }, "(7B FC 별도)", { text: "0.873", bold: true, color: C.accent }, "—", "FC prompt 에서도 32B 천장 유지"],
    ],
    [3.6, 1.5, 1.5, 1.5, 4.2], { fontSize: 10.5, rowH: 0.4 });

  box(s, 0.5, 4.85, 6.0, 2.1, { round: 0.08, fill: C.paper, line: C.good, title: "Scale 이 도움이 되는 곳", titleColor: C.good });
  s.addText("• cue=none cell (5% of M2) 에서만 PMR 이 떨어진다.\n" +
    "• 그곳은 ", { x: 0.7, y: 5.3, w: 5.6, h: 0.8, fontSize: 10.5, fontFace: FONT });
  s.addText([
    { text: "visual prior 가 가장 약한 곳", options: { fontSize: 10.5, bold: true } },
    { text: " — scale 이 grounding 을 강화.\n• 32B 는 \"abstract_reject\" 응답이 35× 증가 → \n  \"이미지에 정보가 부족하다\" 라고 명시적으로 언급.\n• ", options: { fontSize: 10.5 } },
    { text: "→ scaling 이 visual-prior 약한 곳에서만 grounding 향상", options: { fontSize: 10.5, italic: true, color: C.good } },
    { text: ".", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 5.5, w: 5.6, h: 1.4, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 4.85, 6.1, 2.1, { round: 0.08, fill: C.paper, line: C.fail, title: "Scale 이 안 도움 되는 곳", titleColor: C.fail });
  s.addText("• Aggregate PMR 0.926 ≈ 7B 0.931 — saturation 그대로.\n" +
    "• shaded/textured cue 가 강한 cell 에서는 7B/32B 모두 천장.\n" +
    "• ", { x: 6.9, y: 5.3, w: 5.7, h: 0.7, fontSize: 10.5, fontFace: FONT });
  s.addText([
    { text: "Stem reading: 단순 scaling 으로는 architectural saturation 미해결", options: { fontSize: 10.5, bold: true, color: C.fail } },
    { text: ".\n• MechBench (Zhang+ 2024) 의 \"scale 으로 안 풀림\" 과 일치.", options: { fontSize: 10.5 } },
  ], { x: 6.9, y: 5.7, w: 5.7, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 12);
}

// ========================================================================
// SLIDE 13 — §4.3 Korean/Japanese label
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #6 — § 4 . 3   m u l t i l i n g u a l");
  title(s, "§4.3 한국어/일본어 라벨 swap — model-specific 패턴",
    "5-model × {ball/공/ボール × circle/원/円 × planet/행성/惑星}. 4/5 모델에서 ordering 보존, LLaVA-1.5 가 가장 큰 swing.");

  table(s, 0.5, 2.6, 12.3,
    ["모델", "EN ordering", "KO swing", "JA 특이사항"],
    [
      ["Qwen2.5-VL", "ball > planet > circle", "minor (±0.05)", "정상"],
      ["LLaVA-1.5", "ball >> circle ~ planet", { text: "ball KO 0.16 → 공 0.42 (+26pp!)", bold: true, color: C.warm }, "한국어 SFT 약함의 시그널"],
      ["LLaVA-Next", "ball > circle > planet", "minor", "정상 (Mistral)"],
      ["Idefics2", "ball > circle > planet", "minor", { text: "JA 惑星 → Chinese fallback 24%", bold: true, color: C.fail }],
      ["InternVL3", "saturated all", "≈ 0", "InternLM3 강함"],
    ],
    [2.0, 2.5, 4.2, 3.6], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 5.4, 12.3, 1.6, { round: 0.08, fill: C.paper, line: C.teal, title: "발견" });
  s.addText([
    { text: "1. ", options: { fontSize: 11 } },
    { text: "라벨 ordering 은 4/5 모델에서 언어 invariant", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " — H7 (라벨이 regime 을 선택) 이 multilingual generalize.\n", options: { fontSize: 11 } },
    { text: "2. LLaVA-1.5 의 한국어 \"공\" 응답이 영어 \"ball\" 보다 26 pp 높은 PMR — ", options: { fontSize: 11 } },
    { text: "Vicuna 의 한국어 SFT 약점이 visual grounding 약점을 노출", options: { fontSize: 11, bold: true } },
    { text: ".\n", options: { fontSize: 11 } },
    { text: "3. ", options: { fontSize: 11 } },
    { text: "Idefics2 일본어 \"惑星\" → 중국어 응답 24%", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: " — Mistral 의 일본어 SFT 가 약해서 한자를 중국어로 인식.", options: { fontSize: 11 } },
  ], { x: 0.7, y: 5.85, w: 11.9, h: 1.1, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 13);
}

// ========================================================================
// SLIDE 14 — M4b/M4c label-free findings
// ========================================================================
{
  const s = content();
  stripe(s, "B E H A V I O R   #7 — l a b e l - f r e e   p r o m p t s");
  title(s, "M4b / M4c — 라벨 confound 제거하고 다시 측정",
    "M4b: open prompt 라벨 떼니 \"ball ≈ no-label / circle suppress\". M4c: FC 도 같은 패턴 + planet-suppress 추가.");

  box(s, 0.5, 2.6, 6.1, 4.3, { round: 0.08, fill: C.paper, line: C.primary, title: "M4b — open_no_label (Qwen, M2 stim)" });
  s.addText([
    { text: "프롬프트 변경:\n", options: { fontSize: 11, bold: true } },
    { text: "  기존: \"What will the [ball/circle/planet] do?\"\n", options: { fontSize: 10 } },
    { text: "  M4b:  \"What will happen next?\"  (라벨 0)\n\n", options: { fontSize: 10 } },
    { text: "결과 (Qwen):\n", options: { fontSize: 11, bold: true } },
    { text: "  ball:        baseline +0.000  (≈ no-label)\n", options: { fontSize: 10 } },
    { text: "  circle:    baseline −0.065  (suppressed)\n", options: { fontSize: 10 } },
    { text: "  planet:   baseline +0.012  (≈ no-label)\n\n", options: { fontSize: 10 } },
    { text: "Reframe (M4b):\n", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "원래 H2 (\"ball 라벨이 PMR 끌어올림\") 는 ", options: { fontSize: 10 } },
    { text: "Qwen 에서 정확히 반대", options: { fontSize: 10, bold: true, color: C.warm } },
    { text: ":\nball 은 baseline 과 같고, circle 만 baseline 아래로.\n→ 언어 prior 의 ", options: { fontSize: 10 } },
    { text: "asymmetric 영향: 양수가 아니라 음수 노이즈", options: { fontSize: 10, italic: true } },
    { text: ".", options: { fontSize: 10 } },
  ], { x: 0.7, y: 3.05, w: 5.7, h: 3.85, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.3, { round: 0.08, fill: C.paper, line: C.teal, title: "M4c — forced_choice_no_label (Qwen + LLaVA)" });
  s.addText([
    { text: "프롬프트:\n", options: { fontSize: 11, bold: true } },
    { text: "  Q: 다음 중 무엇이 일어날까요?\n", options: { fontSize: 10 } },
    { text: "  A) the depicted object will fall down\n", options: { fontSize: 10 } },
    { text: "  B) ... will stay still\n", options: { fontSize: 10 } },
    { text: "  C) ... will rise\n", options: { fontSize: 10 } },
    { text: "  D) abstract / not enough info\n\n", options: { fontSize: 10 } },
    { text: "결과 (Qwen):  M4b 패턴 재현 + ", options: { fontSize: 11, bold: true } },
    { text: "planet-suppress 추가\n", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "  → option set bias: 행성 regime 은 D 로 collapse\n\n", options: { fontSize: 10 } },
    { text: "결과 (LLaVA-1.5):  ", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "\"A\" 만 반환 (477/480)\n", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "  → first-token logit-ratio 도 \"A\"-bias 확인.\n", options: { fontSize: 10 } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "Vicuna 의 model-level pathology", options: { fontSize: 10, italic: true, color: C.fail } },
    { text: ", greedy 차원 아님.\n", options: { fontSize: 10 } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "FC 분석에서 LLaVA family 제외", options: { fontSize: 10, italic: true, bold: true } },
    { text: " (M6 r2c).", options: { fontSize: 10 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 3.85, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 14);
}

// ========================================================================
// SLIDE 15 — M3 vision encoder probe AUC ladder
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #1 — M 3   e n c o d e r   p r o b e");
  title(s, "M3 — Encoder knows, decoder gates (boomerang)",
    "Stim-defined Y 로 측정 시 5 인코더 모두 AUC = 1.0. Behavioral-Y 로 측정 시 ladder 가 PMR 사다리와 일치.");

  s.addImage({ path: fig("encoder_chain_5model.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.4 });

  table(s, 8.3, 2.6, 4.7,
    ["모델", "behavior-Y AUC", "stim-Y AUC", "PMR"],
    [
      ["Qwen SigLIP", "0.99", "1.0", "0.94"],
      ["Idefics2 SigLIP-SO", "0.93", "1.0", "0.97"],
      ["InternVL3 InternViT", "0.89", "1.0", "0.99"],
      ["LLaVA-Next CLIP", "0.81", "1.0", "0.79"],
      ["LLaVA-1.5 CLIP", "0.73", "1.0", "0.18"],
    ],
    [1.8, 1.4, 1.0, 0.5], { fontSize: 10.5, rowH: 0.4 });

  box(s, 8.3, 4.85, 4.7, 2.2, { round: 0.08, fill: C.paper, line: C.teal, title: "결론 — boomerang" });
  s.addText([
    { text: "• Stim-Y AUC 모두 1.0 → 인코더 ", options: { fontSize: 10.5 } },
    { text: "표현 능력은 균일", options: { fontSize: 10.5, bold: true } },
    { text: ".\n• Behavior-Y AUC 가 PMR 사다리와 일치 → 차이는 ", options: { fontSize: 10.5 } },
    { text: "LM-side gating", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: ".\n• \"인코더가 본 것\" ↔ \"LM 이 \n   읽은 것\" 의 분기:\n", options: { fontSize: 10.5 } },
    { text: "  encoder knows, decoder gates", options: { fontSize: 10.5, italic: true, bold: true, color: C.warm } },
    { text: ".\n• 단, ", options: { fontSize: 10.5 } },
    { text: "LLaVA-1.5 에서 boomerang 부재", options: { fontSize: 10.5, italic: true, color: C.fail } },
    { text: " — 그곳은 인코더가 bottleneck.", options: { fontSize: 10.5 } },
  ], { x: 8.5, y: 5.3, w: 4.3, h: 1.7, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 15);
}

// ========================================================================
// SLIDE 16 — §4.5 cross-encoder swap
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #2 — § 4 . 5   e n c o d e r   s w a p");
  title(s, "§4.5 — SigLIP-SO400M + Mistral-7B (Idefics2 vibe) replicates",
    "Qwen-style 결과를 SigLIP-SO 위에서 재현 (controlled encoder family swap). H-encoder-saturation causal evidence.");

  box(s, 0.5, 2.6, 6.0, 4.3, { round: 0.08, fill: C.paper, line: C.primary, title: "실험 설계" });
  s.addText([
    { text: "통제: ", options: { fontSize: 11, bold: true } },
    { text: "encoder family 만 swap, 나머지 동일.\n\n", options: { fontSize: 11 } },
    { text: "비교 1:  Qwen2.5-VL-7B (SigLIP)\n", options: { fontSize: 10.5 } },
    { text: "비교 2:  Idefics2-8B (SigLIP-SO400M + perceiver-resampler + Mistral-7B)\n\n", options: { fontSize: 10.5 } },
    { text: "측정 항목:\n", options: { fontSize: 11, bold: true } },
    { text: "• Vision encoder probe AUC\n", options: { fontSize: 10.5 } },
    { text: "• Behavioral PMR + H7 패턴\n", options: { fontSize: 10.5 } },
    { text: "• 후속: M5a steering + M5b SAE\n\n", options: { fontSize: 10.5 } },
    { text: "예상:\n", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "encoder-saturation hypothesis 가 맞다면\n", options: { fontSize: 10.5 } },
    { text: "Idefics2 도 PMR ≥ 0.9 + circle override 패턴.\n", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 3.05, w: 5.6, h: 3.8, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.3, { round: 0.08, fill: C.cardCircle, line: C.good, title: "결과 — H-encoder-saturation 통과", titleColor: C.good });
  s.addText([
    { text: "1. Idefics2 vision probe AUC: ", options: { fontSize: 11, bold: true } },
    { text: "0.93", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " (Qwen 0.99 와 동일 saturation tier)\n", options: { fontSize: 10.5 } },
    { text: "2. Idefics2 PMR_nolabel: ", options: { fontSize: 11, bold: true } },
    { text: "0.97 ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "(Qwen 0.94 와 같은 tier)\n", options: { fontSize: 10.5 } },
    { text: "3. H2 paired-Δ: ", options: { fontSize: 11, bold: true } },
    { text: "circle override 재현", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " (planet/circle < 0)\n", options: { fontSize: 10.5 } },
    { text: "4. M5a L25 steering: ", options: { fontSize: 11, bold: true } },
    { text: "10/10 flip", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " (Qwen L10 의 Idefics2 layer-equivalent)\n\n", options: { fontSize: 10.5 } },
    { text: "→ ", options: { fontSize: 11 } },
    { text: "Encoder family (SigLIP) 가 결정자", options: { fontSize: 11, bold: true, color: C.good } },
    { text: ".  CLIP 와 SigLIP 사이의 격차가 PMR 사다리 의 주요 원인.\n", options: { fontSize: 10.5 } },
    { text: "→ Causal evidence at the encoder-family level", options: { fontSize: 10.5, italic: true } },
    { text: ".", options: { fontSize: 10.5 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 3.8, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 16);
}

// ========================================================================
// SLIDE 17 — M5b SAE encoder ablation round 1
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #3 — M 5 b   r o u n d   1");
  title(s, "M5b — Encoder SAE feature ablation (round 1, 2026-04-28)",
    "5120-feature SAE × top-k Cohen's d 랭크된 feature ablation.  3 of 5 모델에서 PMR 1.0 → 0.0 깨짐.");

  s.addImage({ path: fig("m5b_sae_intervention_cross_model.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.4 });

  table(s, 8.3, 2.6, 4.7,
    ["모델", "Layer", "Break at k =", "Verdict"],
    [
      ["Qwen2.5-VL", "L31 (last)", { text: "20 (0.4%)", bold: true, color: C.good }, "★★★ clean"],
      ["Idefics2", "L26", { text: "160 (3.5%)", color: C.good }, "★ break"],
      ["InternVL3", "L23 (-1)", { text: "160 (3.9%)", color: C.good }, "★ break"],
      ["LLaVA-1.5", "L22 (-2)", { text: "NULL ≤ 800", color: C.fail, bold: true }, "✗ encoder NULL"],
      ["LLaVA-Next", "L22 (-2)", { text: "NULL ≤ 160", color: C.fail, bold: true }, "✗ encoder NULL"],
    ],
    [1.5, 1.0, 1.4, 0.8], { fontSize: 10, rowH: 0.5 });

  box(s, 8.3, 5.4, 4.7, 1.7, { round: 0.08, fill: C.failSoft, line: C.fail, title: "Round 1 reading (당시)", titleColor: C.fail });
  s.addText([
    { text: "비-CLIP: ~30 SAE feature 로 ", options: { fontSize: 9.5 } },
    { text: "encoder 안에 \"physics-mode\" 표현 국소화", options: { fontSize: 9.5, bold: true } },
    { text: ".\nCLIP family: encoder-side 에 없음 → ", options: { fontSize: 9.5 } },
    { text: "LM-side direction 만 라우팅?", options: { fontSize: 9.5, bold: true, color: C.warm } },
    { text: "\n* 2026-05-01 round 2 가 이 reading 을 reframe ", options: { fontSize: 9.5, italic: true, color: C.warm } },
    { text: "(슬라이드 30-32)", options: { fontSize: 9.5, italic: true } },
    { text: ".", options: { fontSize: 9.5 } },
  ], { x: 8.5, y: 5.85, w: 4.3, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 17);
}

// ========================================================================
// SLIDE 18 — M4 LM logit lens cross-model
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #4 — M 4   L M   l o g i t   l e n s");
  title(s, "M4 cross-model — LM probe AUC ladder (5-model × 5-layer)",
    "M2 cross-model captures 재활용 (no new inference). 가장 놀라운 결과: Idefics2 LM AUC 0.995 > vision AUC 0.93.");

  s.addImage({ path: fig("m4_lm_probing_cross_model.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.4 });

  table(s, 8.3, 2.6, 4.7,
    ["모델", "Vision AUC", "LM AUC", "방향성"],
    [
      ["Idefics2", "0.93", { text: "0.995", bold: true, color: C.warm }, { text: "LM > Vision", bold: true, color: C.warm }],
      ["Qwen2.5-VL", "0.99", "0.96", "≈"],
      ["LLaVA-Next", "0.81", "0.79", "≈"],
      ["LLaVA-1.5", "0.73", "0.76", "≈"],
      ["InternVL3", "untestable", "untestable", "(n_neg=1)"],
    ],
    [1.5, 1.2, 1.1, 0.9], { fontSize: 10, rowH: 0.4 });

  box(s, 8.3, 4.85, 4.7, 2.3, { round: 0.08, fill: C.cardCircle, line: C.warm, title: "Idefics2 의 의미", titleColor: C.warm });
  s.addText([
    { text: "• Vision AUC 0.93 ≤ LM AUC 0.995\n", options: { fontSize: 10 } },
    { text: "• Perceiver-resampler 가 정보를 ", options: { fontSize: 10 } },
    { text: "오히려 더 잘 보존", options: { fontSize: 10, bold: true, color: C.warm } },
    { text: " (strip ✗)\n", options: { fontSize: 10 } },
    { text: "• 그런데 §4.6 픽셀 ascent 는 0/9 → ", options: { fontSize: 10 } },
    { text: "정보 LM 도달 ≠ 픽셀-공간 routability", options: { fontSize: 10, bold: true, italic: true } },
    { text: "\n", options: { fontSize: 10 } },
    { text: "• Forward 통과 OK, ", options: { fontSize: 10 } },
    { text: "inverse pixel→v_L 차단", options: { fontSize: 10, bold: true } },
    { text: "\n• → 슬라이드 28-29 에서 perceiver hypothesis 정밀화\n• ", options: { fontSize: 10 } },
    { text: "M5a 도 10/10 flip 으로 forward 작동 확인", options: { fontSize: 10, italic: true } },
    { text: " (슬라이드 21).", options: { fontSize: 10 } },
  ], { x: 8.5, y: 5.3, w: 4.3, h: 1.85, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 18);
}

// ========================================================================
// SLIDE 19 — M5a VTI L10 (Qwen)
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #5 — M 5 a   V T I   L 1 0");
  title(s, "M5a (Qwen) — \"한 레이어에 + α·v_L10 더하면 응답이 뒤집힌다\"",
    "이미지 한 픽셀 안 건드리고 LM L10 hidden state 에 한 줄 추가만으로 추상 → 물리 collapse.");

  box(s, 0.5, 2.6, 6.0, 4.3, { round: 0.08, fill: C.paper, line: C.primary, title: "v_L10 정의 + 개입" });
  s.addText([
    { text: "v_L10 정의:\n", options: { fontSize: 11, bold: true } },
    { text: "  hidden_L10[physics=1].mean()\n", options: { fontSize: 10 } },
    { text: "  − hidden_L10[physics=0].mean()\n", options: { fontSize: 10 } },
    { text: "  ≈ \"physics-mode 방향\"\n\n", options: { fontSize: 10 } },
    { text: "개입:\n", options: { fontSize: 11, bold: true } },
    { text: "  hidden_L10 ← hidden_L10 + α·v_L10\n", options: { fontSize: 10 } },
    { text: "  α 스케일 0, 5, 10, 20, 40\n\n", options: { fontSize: 10 } },
    { text: "자극: ", options: { fontSize: 11, bold: true } },
    { text: "line/blank/none (가장 추상한 원, baseline PMR ≈ 0.05)\n\n", options: { fontSize: 10 } },
    { text: "라벨: ", options: { fontSize: 11, bold: true } },
    { text: "circle (라벨도 abstract — 모든 prior 가 D 쪽)", options: { fontSize: 10 } },
  ], { x: 0.7, y: 3.05, w: 5.6, h: 3.8, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.3, { round: 0.08, fill: C.cardCircle, line: C.good, title: "결과 — α=40 에서 10/10 D→B flip", titleColor: C.good });
  s.addText([
    { text: "α = 0 (baseline):\n", options: { fontSize: 10.5, bold: true } },
    { text: "  \"This is just a circle on a white background.\"\n", options: { fontSize: 9.5, italic: true } },
    { text: "  → 추상 (D), 10/10\n\n", options: { fontSize: 9.5 } },
    { text: "α = 10 ~ 20:\n", options: { fontSize: 10.5, bold: true } },
    { text: "  여전히 추상 (D).\n\n", options: { fontSize: 9.5, italic: true } },
    { text: "α = 40:\n", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "  \"It stays still — the circle appears to be floating in space\n   without external force.\"\n", options: { fontSize: 9.5, italic: true } },
    { text: "  → ", options: { fontSize: 9.5 } },
    { text: "물리·정지 (B), 10/10 flip!", options: { fontSize: 9.5, bold: true, color: C.good } },
    { text: "\n\n다른 레이어 (L15·L20·L25): 같은 α 에서 변화 없음.\n", options: { fontSize: 10.5 } },
    { text: "→ L10 만의 ", options: { fontSize: 10.5 } },
    { text: "narrow band intervention", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: ".", options: { fontSize: 10.5 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 3.8, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 19);
}

// ========================================================================
// SLIDE 20 — M5a-ext bidirectional regime axis
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #6 — M 5 a - e x t");
  title(s, "M5a-ext — v_L10 은 \"물리/추상\" axis 가 아니라 regime axis",
    "+α: 동적 물리 (떨어진다). −α: 정적 물리 (떠 있다 / 정지). baseline D 는 |α| 임계값 아래에 있는 \"미정\" 상태.");

  table(s, 0.5, 2.6, 12.3,
    ["α 부호 / 크기", "응답 예시", "regime", "텍스트 카테고리"],
    [
      ["α = 0", "\"This is just a circle on white background.\"", "abstract / undecided", { text: "D", bold: true } ],
      ["α = +40", "\"The circle will continue falling downward due to gravity.\"", "kinetic physics", { text: "A — falls", bold: true, color: C.good }],
      ["α = +40 (label=ball)", "\"The ball will roll along the surface.\"", "kinetic physics", { text: "A — falls", bold: true, color: C.good }],
      ["α = −40", "\"The circle remains stationary, suspended in midair.\"", "static physics", { text: "B — stays", bold: true, color: C.warm }],
      ["α = −40 (label=ball)", "\"The ball is floating without external force.\"", "static physics", { text: "B — stays", bold: true, color: C.warm }],
    ],
    [2.0, 5.0, 2.5, 2.8], { fontSize: 10, rowH: 0.55 });

  box(s, 0.5, 5.55, 12.3, 1.5, { round: 0.08, fill: C.paper, line: C.warm, title: "재해석 — H-direction-bidirectional → H-regime-axis", titleColor: C.warm });
  s.addText([
    { text: "초기 가설: ", options: { fontSize: 11 } },
    { text: "v_L10 이 \"object-ness\" axis (+ → 물체 / − → 추상)", options: { fontSize: 11, italic: true } },
    { text: ".\n수정: ", options: { fontSize: 11 } },
    { text: "v_L10 은 regime axis (+ → kinetic / − → static)", options: { fontSize: 11, italic: true, bold: true, color: C.warm } },
    { text: ", baseline D 는 |α| 임계값 아래의 \"undecided\" 상태.\n", options: { fontSize: 11 } },
    { text: "→ 양쪽 부호 모두 ", options: { fontSize: 11 } },
    { text: "physics-mode 활성화", options: { fontSize: 11, bold: true } },
    { text: ", 부호가 ", options: { fontSize: 11 } },
    { text: "어떤 physics regime", options: { fontSize: 11, bold: true } },
    { text: " 인지 결정.", options: { fontSize: 11 } },
  ], { x: 0.7, y: 6.0, w: 11.9, h: 1.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 20);
}

// ========================================================================
// SLIDE 21 — M5a cross-model 3 of 4 flip
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #7 — M 5 a   c r o s s - m o d e l");
  title(s, "M5a runtime steering — 3 of 4 testable models flip 10/10",
    "각 모델 자체 v_L_per_model + 자체 α dynamic range. LLaVA-1.5 만 0/10 (encoder bottleneck).");

  table(s, 0.5, 2.6, 12.3,
    ["모델", "Layer", "α", "Baseline cell", "결과"],
    [
      ["Qwen2.5-VL", "L10", "40", "line/blank/none circle", { text: "10/10 flip ✅", bold: true, color: C.good }],
      ["LLaVA-Next", "L20 / L25", "10 / 15-20", "line/blank/both", { text: "10/10 flip ✅", bold: true, color: C.good }],
      ["Idefics2", "L25", "20", "line/blank/none", { text: "10/10 flip ✅", bold: true, color: C.good }],
      ["LLaVA-1.5", "L25 (sweep α=0~60)", "—", "line/blank/none", { text: "0/10 ✗", bold: true, color: C.fail }],
      ["InternVL3", "—", "—", "baseline=1.0", { text: "untestable (ceiling)", bold: true, color: C.inkSoft }],
    ],
    [2.0, 1.7, 1.2, 3.0, 4.4], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 5.4, 6.0, 1.7, { round: 0.08, fill: C.cardCircle, line: C.good, title: "Idefics2 결과 텍스트", titleColor: C.good });
  s.addText([
    { text: "α = 0 (baseline): ", options: { fontSize: 10.5, bold: true } },
    { text: "\"It is unclear what will happen — the image is just an arrow and a circle.\"  (10/10 abstract)\n\n", options: { fontSize: 10, italic: true } },
    { text: "α = 20 @ L25: ", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "\"The tip of the arrow will hit the center of the circle.\"  (10/10 physics)", options: { fontSize: 10, italic: true } },
  ], { x: 0.7, y: 5.85, w: 5.6, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 5.4, 6.1, 1.7, { round: 0.08, fill: C.failSoft, line: C.fail, title: "LLaVA-1.5 — encoder bottleneck", titleColor: C.fail });
  s.addText([
    { text: "α 0~60 sweep, layer L20/L25 모두 ", options: { fontSize: 10 } },
    { text: "0/10 flip", options: { fontSize: 10, bold: true, color: C.fail } },
    { text: ".\n", options: { fontSize: 10 } },
    { text: "v_L 방향 자체는 잡히지만 (ratio 정확), L 위에서 흔들어도 응답이 안 바뀜.\n→ ", options: { fontSize: 10 } },
    { text: "encoder-side 에 인코딩이 너무 약해서 LM 이 흔들리지 않는 cell", options: { fontSize: 10, italic: true } },
    { text: ".\n→ §4.6 weak-shortcut 결과와 일치 (L25 only, n=10 에서 4/10).", options: { fontSize: 10 } },
  ], { x: 6.9, y: 5.85, w: 5.7, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 21);
}

// ========================================================================
// SLIDE 22 — M5b SIP + activation patching
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #8 — M 5 b   S I P");
  title(s, "M5b SIP — 어디까지 patching 하면 physics regime 이 회복되는가",
    "Qwen: L0-L9 patching → IE = +1.0 (20/20 회복). L10-L11 partial. L14+ zero. 수직 절벽 like 위치 결정.");

  s.addImage({ path: fig("m5b_sip_per_layer_ie.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  table(s, 8.3, 2.6, 4.7,
    ["Layer 범위", "Qwen IE (n=20)", "LLaVA-1.5 IE (n=15)"],
    [
      ["L0 - L9", { text: "+1.0", bold: true, color: C.good }, "(separate run)"],
      ["L10 - L11", { text: "+0.6", color: C.warm }, "—"],
      ["L14+", { text: "0", color: C.fail }, "0"],
      ["L20 (LLaVA lock)", "—", { text: "+1.0 (62.5% depth)", bold: true, color: C.good }],
    ],
    [1.6, 1.5, 1.6], { fontSize: 10, rowH: 0.45 });

  box(s, 8.3, 4.85, 4.7, 2.3, { round: 0.08, fill: C.cardCircle, line: C.primary, title: "Cross-model 비교" });
  s.addText([
    { text: "• ", options: { fontSize: 10 } },
    { text: "Qwen: L10 (relative depth 36 %)", options: { fontSize: 10, bold: true } },
    { text: "\n• ", options: { fontSize: 10 } },
    { text: "LLaVA-1.5: L20 (relative depth 62.5 %)", options: { fontSize: 10, bold: true } },
    { text: "\n• Curve shape 는 동일 (수직 절벽)\n• Locus 는 model-specific\n→ \"transition layer\" 가 ", options: { fontSize: 10 } },
    { text: "model-specific 이지만 sharp 하다", options: { fontSize: 10, italic: true, color: C.warm } },
    { text: ".\n→ paper 기여 2 (causal localization) 의 Qwen-only 였던 것이 ", options: { fontSize: 10 } },
    { text: "LLaVA-1.5 까지 cross-model 확장", options: { fontSize: 10, bold: true, color: C.good } },
    { text: ".", options: { fontSize: 10 } },
  ], { x: 8.5, y: 5.3, w: 4.3, h: 1.85, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 22);
}

// ========================================================================
// SLIDE 23 — M5b MLP knockout (L9)
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   #9 — M 5 b   M L P   k n o c k o u t");
  title(s, "M5b — L9 MLP 가 단독으로 physics-mode 결정 (sufficient + necessary)",
    "MLP knockout (necessity): L9 만 IE = +1.0, attention knockout 모든 28 레이어에서 IE = 0.");

  s.addImage({ path: fig("m5b_knockout_per_layer_ie.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  box(s, 8.3, 2.6, 4.7, 4.5, { round: 0.08, fill: C.paper, line: C.warm, title: "Triangulation — Qwen 인과 chain", titleColor: C.warm });
  s.addText([
    { text: "1. Encoder ", options: { fontSize: 10.5 } },
    { text: "~30 SAE feature ", options: { fontSize: 10.5, bold: true } },
    { text: "carry physics signal\n", options: { fontSize: 10.5 } },
    { text: "    (M5b round 1)\n\n", options: { fontSize: 9.5, italic: true, color: C.inkSoft } },
    { text: "2. ", options: { fontSize: 10.5 } },
    { text: "L0-L9 visual tokens transport ", options: { fontSize: 10.5, bold: true } },
    { text: "the signal\n", options: { fontSize: 10.5 } },
    { text: "    (M5b SIP)\n\n", options: { fontSize: 9.5, italic: true, color: C.inkSoft } },
    { text: "3. ", options: { fontSize: 10.5 } },
    { text: "L9 MLP constructs the commitment", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "\n    (MLP knockout, +1.0)\n\n", options: { fontSize: 9.5, italic: true, color: C.inkSoft } },
    { text: "4. ", options: { fontSize: 10.5 } },
    { text: "L10 reads it out via redundant attention", options: { fontSize: 10.5, bold: true } },
    { text: "\n    (per-head 196 cells = 0)\n\n", options: { fontSize: 9.5, italic: true, color: C.inkSoft } },
    { text: "5. → letter B/D 결정.\n\n", options: { fontSize: 10.5 } },
    { text: "M5a/M5b off-by-one 조화: 같은 결정 boundary 의 두 측면 (construction L9 / read-out L10).", options: { fontSize: 10, italic: true, color: C.warm } },
  ], { x: 8.5, y: 3.05, w: 4.3, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 23);
}

// ========================================================================
// SLIDE 24 — M5b per-head attention knockout
// ========================================================================
{
  const s = content();
  stripe(s, "M E C H A N I S M   # 10 — p e r - h e a d   k n o c k o u t");
  title(s, "M5b per-head — attention 의 모든 (layer, head) 가 dispensable",
    "20 stim × 7 layer (L8-L14) × 28 head = 196 ablation cells — 모두 IE = 0. \"narrow IE band\" 가 attention 에는 없다.");

  s.addImage({ path: fig("m5b_per_head_attention_ie.png"), x: 0.5, y: 2.6, w: 7.5, h: 4.5 });

  box(s, 8.3, 2.6, 4.7, 4.5, { round: 0.08, fill: C.paper, line: C.fail, title: "결론 — \"construction-and-broadcast\"", titleColor: C.fail });
  s.addText([
    { text: "기존 가설 (H10): ", options: { fontSize: 10.5, bold: true } },
    { text: "\"2-3 narrow attention IE bands\"\n", options: { fontSize: 10, italic: true } },
    { text: "  → Wang+ 2023 같은 attention head 분석 따라.\n\n", options: { fontSize: 10 } },
    { text: "관측: ", options: { fontSize: 10.5, bold: true } },
    { text: "196 cells 모두 IE = 0\n", options: { fontSize: 10, bold: true, color: C.fail } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "어떤 single head 도 load-bearing 아님", options: { fontSize: 10, italic: true } },
    { text: ".\n  → attention 은 ", options: { fontSize: 10 } },
    { text: "fully redundant", options: { fontSize: 10, bold: true, color: C.fail } },
    { text: " at both layer and head 수준.\n\n", options: { fontSize: 10 } },
    { text: "재해석: ", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: "기계 작동은 \"pull through specific head\" 가 아니라\n", options: { fontSize: 10 } },
    { text: "  ", options: { fontSize: 10 } },
    { text: "\"L9 MLP 가 commitment 를 만들고 L10 attention 이 broadcast\"", options: { fontSize: 10, italic: true, bold: true, color: C.warm } },
    { text: "\n  → 이 메커니즘은 단일 head 에 의존하지 않음.\n", options: { fontSize: 10 } },
    { text: "→ H10 은 ", options: { fontSize: 10 } },
    { text: "refuted", options: { fontSize: 10, bold: true, color: C.fail } },
    { text: ", 새 framing 채택.", options: { fontSize: 10 } },
  ], { x: 8.5, y: 3.05, w: 4.3, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 24);
}

// ========================================================================
// SLIDE 25 — §4.6 Qwen pixel counterfactual
// ========================================================================
{
  const s = content();
  stripe(s, "P I X E L   #1 — § 4 . 6   Q w e n   c o u n t e r f a c t u a l");
  title(s, "§4.6 — 픽셀에 \"눈에 거의 안 보이는 노이즈\"만 더해도 응답이 뒤집힌다",
    "v_L10 방향으로 픽셀 공간 gradient ascent (200 step, ε=0.1). 5/5 v_L10 flip vs 매칭 magnitude random 0/15.");

  box(s, 0.5, 2.6, 6.0, 4.5, { round: 0.08, fill: C.paper, line: C.primary, title: "방법 + 응답 비교" });
  s.addText([
    { text: "Setup:\n", options: { fontSize: 11, bold: true } },
    { text: "  자극 baseline = line/blank/none circle (가장 추상)\n", options: { fontSize: 10 } },
    { text: "  Adam 200 step, ε = 0.1, v_L10 방향\n\n", options: { fontSize: 10 } },
    { text: "Baseline (ε = 0):\n", options: { fontSize: 11, bold: true } },
    { text: "  \"The circle will remain stationary as there is\n", options: { fontSize: 9.5, italic: true } },
    { text: "   no indication of movement.\"\n", options: { fontSize: 9.5, italic: true } },
    { text: "  → 추상 (D)\n\n", options: { fontSize: 9.5 } },
    { text: "v_L10 ascent (ε = 0.05):\n", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "  \"The circle will continue to fall downward\n", options: { fontSize: 9.5, italic: true, color: C.good } },
    { text: "   due to gravity.\"\n", options: { fontSize: 9.5, italic: true, color: C.good } },
    { text: "  → ", options: { fontSize: 9.5 } },
    { text: "물리 응답으로 collapse", options: { fontSize: 9.5, bold: true, color: C.good } },
    { text: "\n\n", options: { fontSize: 9.5 } },
    { text: "통계: 5/5 v_L10 flip vs 0/15 random.\n", options: { fontSize: 10, bold: true } },
    { text: "→ 단순한 \"아무 노이즈\" 는 안 되고 ", options: { fontSize: 10 } },
    { text: "방향 특이성", options: { fontSize: 10, italic: true, bold: true } },
    { text: " 보장.", options: { fontSize: 10 } },
  ], { x: 0.7, y: 3.05, w: 5.6, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.5, { round: 0.08, fill: C.cardCircle, line: C.warm, title: "shortcut 의 픽셀 인코드성 (paper 기여 3)", titleColor: C.warm });
  s.addText([
    { text: "발견: ", options: { fontSize: 11, bold: true } },
    { text: "shortcut 은 \"runtime hidden injection\" 만이 아니라 ", options: { fontSize: 11 } },
    { text: "픽셀 자체에 인코드 가능", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: ".\n\n", options: { fontSize: 11 } },
    { text: "왜 중요한가:\n", options: { fontSize: 11, bold: true } },
    { text: "  • 적대적 공격 (adversarial) 의 한 형태\n", options: { fontSize: 10 } },
    { text: "  • 신뢰성 / 안전성: 작은 픽셀 변화가 응답 뒤집음\n", options: { fontSize: 10 } },
    { text: "  • Falsification: 매칭 magnitude random 0/15 이\n", options: { fontSize: 10 } },
    { text: "    \"any perturbation\" 가설을 falsify\n\n", options: { fontSize: 10 } },
    { text: "다음 단계 (슬라이드 26):\n", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "  같은 ascent 가 모든 모델에서 동일하게 작동하는가?\n", options: { fontSize: 10 } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "5-model layer sweep", options: { fontSize: 10, bold: true } },
    { text: " — 매우 다른 결과.", options: { fontSize: 10 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 25);
}

// ========================================================================
// SLIDE 26 — §4.6 cross-model layer sweep
// ========================================================================
{
  const s = content();
  stripe(s, "P I X E L   #2 — § 4 . 6   c r o s s - m o d e l");
  title(s, "§4.6 5-model layer sweep — 픽셀 routability 가 architecture-conditional",
    "Qwen broad shortcut (5 layers ≥ 80%). LLaVA-Next L20+L25. Idefics2 0/9 (anomaly!). Random aggregate 1/250 = 0.4%.");

  table(s, 0.5, 2.6, 12.3,
    ["모델", "Shortcut layers (10/10 flip)", "Random control", "특징"],
    [
      ["Qwen2.5-VL", { text: "L5/10/15/20/25 모두 ≥ 80%", bold: true, color: C.good }, "1/10 (L10), rest 0", "broad shortcut"],
      ["LLaVA-Next", { text: "L20 (10/10), L25 (10/10)", bold: true, color: C.good }, "0/10 모두", "Mid — clean 2 layer"],
      ["LLaVA-1.5", { text: "L25 only (4/10 at n=10)", color: C.warm }, "0/10 모두", "weak shortcut"],
      ["Idefics2", { text: "0/9 layers (L5-L31)", bold: true, color: C.fail }, "0/10 모두", "anomaly — 슬라이드 27"],
      ["InternVL3", { text: "untestable (baseline=1.0)", color: C.inkSoft }, "—", "protocol-saturated"],
    ],
    [1.7, 4.0, 2.5, 4.1], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 5.6, 12.3, 1.5, { round: 0.08, fill: C.accentSoft, line: C.warm, title: "랜덤 컨트롤 — 방향 특이성 검증", titleColor: C.warm });
  s.addText([
    { text: "Aggregate: ", options: { fontSize: 11, bold: true } },
    { text: "5 models × 5 layers × 10 trials = 250 trials, ", options: { fontSize: 11 } },
    { text: "1/250 hit (Qwen L10 only)", options: { fontSize: 11, bold: true, color: C.good } },
    { text: ".\n", options: { fontSize: 11 } },
    { text: "→ 24/25 random-control cells = 0/10. ", options: { fontSize: 11 } },
    { text: "방향 특이성 보장", options: { fontSize: 11, bold: true } },
    { text: " — 단순한 magnitude 는 응답을 뒤집지 못함.\n", options: { fontSize: 11 } },
    { text: "→ ", options: { fontSize: 11 } },
    { text: "v_L direction-specific shortcut", options: { fontSize: 11, italic: true, bold: true, color: C.warm } },
    { text: " 가 architecture-conditional 한 형태로 5 모델에 분포.", options: { fontSize: 11 } },
  ], { x: 0.7, y: 6.05, w: 11.9, h: 1.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 26);
}

// ========================================================================
// SLIDE 27 — Idefics2 9-layer + perceiver hypothesis
// ========================================================================
{
  const s = content();
  stripe(s, "P I X E L   #3 — I d e f i c s 2   p e r c e i v e r");
  title(s, "Idefics2 9-layer disambiguation — perceiver-resampler 가 leading candidate",
    "L5 ~ L31 (16-97% depth) 모두 0/10 flip. v_L projection 은 정상 ascending. 무엇이 routing 을 차단하는가?");

  table(s, 0.5, 2.6, 12.3,
    ["측정", "값", "해석"],
    [
      ["§4.6 픽셀 ascent (9 layers L5-L31)", { text: "0/10 모두", bold: true, color: C.fail }, "픽셀-공간 routability 차단"],
      ["v_L projection (L26-L30 ascent)", { text: "−11 → +28 정상 상승", color: C.good }, "방향성 자체는 잡힘"],
      ["v_L projection (L31)", { text: "−72 → +163 정상", color: C.good }, "절대값 더 큰 ascent 도 작동"],
      ["M4 LM probe AUC", { text: "0.995", bold: true, color: C.good }, "정보가 LM 까지 도달"],
      ["M5a runtime steering (L25 α=20)", { text: "10/10 flip ✅", bold: true, color: C.good }, "Forward-side hidden injection 작동"],
    ],
    [4.5, 3.5, 4.3], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 5.4, 12.3, 1.7, { round: 0.08, fill: C.cardCircle, line: C.warm, title: "Perceiver-resampler hypothesis — Idefics2 가 unique non-MLP projector", titleColor: C.warm });
  s.addText([
    { text: "Forward (LM 이 hidden 받기): ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "통과 OK\n", options: { fontSize: 11 } },
    { text: "Inverse (픽셀 → v_L gradient): ", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "차단\n", options: { fontSize: 11 } },
    { text: "→ Perceiver-resampler 가 ", options: { fontSize: 11 } },
    { text: "정보를 forward 로는 잘 보내지만, inverse pixel-space gradient routability 만 끊는다", options: { fontSize: 11, italic: true, bold: true, color: C.warm } },
    { text: " 는 가설.\n", options: { fontSize: 11 } },
    { text: "주의: ", options: { fontSize: 10 } },
    { text: "Idefics2 는 encoder + projector + AnyRes 가 동시에 다름 — perceiver 단독 isolation 미검증 (controlled projector swap = M-PSwap, NaN 미해결).", options: { fontSize: 10, italic: true } },
  ], { x: 0.7, y: 5.85, w: 11.9, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 27);
}

// ========================================================================
// SLIDE 28 — M5b round 2 methodology
// ========================================================================
{
  const s = content();
  stripe(s, "M 5 b   r o u n d   2 — m e t h o d");
  title(s, "M5b round 2 (post-projection) — \"projector 출력에서 같은 ablation\"",
    "Round 1 은 vision-encoder hidden 위에서. Round 2 는 projector output (model.multi_modal_projector 또는 model.connector) 위.");

  box(s, 0.5, 2.6, 12.3, 2.4, { round: 0.08, fill: C.paper, line: C.divider, title: "왜 round 2 가 필요한가" });
  s.addText([
    { text: "Round 1 결과: ", options: { fontSize: 11, bold: true } },
    { text: "LLaVA family 가 encoder-side NULL → \"LM-side direction 으로만 라우팅\" 으로 해석.\n", options: { fontSize: 11 } },
    { text: "그러나 ", options: { fontSize: 11, bold: true } },
    { text: "round 1 은 ball cell 만 테스트", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: " (ball/filled/blank+both, ball/shaded/blank+none).\n", options: { fontSize: 11 } },
    { text: "Round 2 의 가설:  projector output 에 commitment 가 ", options: { fontSize: 11 } },
    { text: "다른 형태로 인코드되어 있을 수 있다", options: { fontSize: 11, italic: true } },
    { text: ".\n", options: { fontSize: 11 } },
    { text: "→ ", options: { fontSize: 11 } },
    { text: "circle cell 도 테스트", options: { fontSize: 11, bold: true } },
    { text: " — 라벨이 abstract 인 경우 다른 양상이 나올 수 있다.", options: { fontSize: 11 } },
  ], { x: 0.7, y: 3.05, w: 11.9, h: 1.85, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  table(s, 0.5, 5.2, 12.3,
    ["모델", "Round 1 hook (encoder)", "Round 2 hook (post-proj)", "shape"],
    [
      ["Qwen2.5-VL", "vision_hidden_31", "model.model.visual.merger", "(n_groups, 3584)"],
      ["LLaVA-1.5/Next/LMSwap", "vision L22 (-2)", "model.multi_modal_projector", "(576, 4096)"],
      ["Idefics2", "vision L26", "model.connector (perceiver+MLP)", "(64, 4096)"],
      ["InternVL3", "vision L23 (-1)", "model.multi_modal_projector", "(256, 3072)"],
    ],
    [2.0, 2.5, 4.5, 3.3], { fontSize: 10, rowH: 0.4 });

  footer(s); pageNum(s, 28);
}

// ========================================================================
// SLIDE 29 — Per-cell results table (5 × 4)
// ========================================================================
{
  const s = content();
  stripe(s, "M 5 b   r o u n d   2 — l a d d e r");
  title(s, "Round 2 결과 — regime-cross capacity ladder (5 모델 × 4 cell)",
    "discriminating cell 은 circle / filled / blank+both. ball cell 은 모두 PMR=1 유지 (label prior 강함).");

  // Custom layout for 5x4 with verdict column
  const rows = [
    { name: "Qwen2.5-VL", c1: "k=20 → \"remain stationary\"  PMR 1→0 ✅", c2: "stays kinetic", c3: "ball stays PMR=1", verdict: "★★★ 깨끗", color: C.good },
    { name: "Idefics2", c1: "k=20 → \"disappear\" PMR 0\nk=40+ → \"continue to expand\" PMR=1*", c2: "stays \"spin\" (motion verb)", c3: "ball stays PMR=1", verdict: "★★ 부분", color: C.warm },
    { name: "LLaVA-Next", c1: "k=20 stays / k=40+ → \"expand\"\nPMR 1→0", c2: "no break tested", c3: "ball stays PMR=1", verdict: "★★ 부분", color: C.warm },
    { name: "LLaVA-1.5", c1: "baseline PMR=0\n(\"drawn towards red arrow\")", c2: "no break tested", c3: "ball stays PMR=1", verdict: "✦ baseline-abstract", color: C.teal },
    { name: "InternVL3", c1: "stays \"fall downwards\"\nNULL at all k ≤ 160", c2: "drift only", c3: "ball stays PMR=1", verdict: "✗ true NULL", color: C.fail },
    { name: "Qwen 32B (B5 today)", c1: "k=40 → \"remain stationary\" PMR=0\n(7B 는 k=20)", c2: "(no test)", c3: "(no test)", verdict: "★★★ 깨끗", color: C.good },
  ];

  // header
  const hy = 2.6;
  const headers = ["모델", "circle/filled/blank+both", "circle/shaded/blank+none", "ball/filled/blank+both", "verdict"];
  const colX = [0.5, 2.7, 5.6, 8.4, 10.5];
  const colW = [2.2, 2.9, 2.8, 2.1, 2.3];
  headers.forEach((h, i) => s.addText(h, {
    x: colX[i], y: hy, w: colW[i], h: 0.35,
    fontSize: 10, fontFace: FONT, bold: true, color: C.primary }));
  s.addShape("rect", { x: 0.5, y: hy + 0.32, w: 12.3, h: 0.02,
    fill: { color: C.divider }, line: { color: C.divider } });

  rows.forEach((r, i) => {
    const ry = hy + 0.4 + i * 0.6;
    if (i === 5) {  // separator before B5 today row
      s.addShape("rect", { x: 0.5, y: ry - 0.05, w: 12.3, h: 0.02,
        fill: { color: C.accent }, line: { color: C.accent } });
    }
    s.addText(r.name, { x: colX[0], y: ry, w: colW[0], h: 0.6,
      fontSize: 10, fontFace: FONT, bold: true, color: i === 5 ? C.accent : C.ink });
    s.addText(r.c1, { x: colX[1], y: ry, w: colW[1], h: 0.6, fontSize: 9, fontFace: FONT, color: C.ink });
    s.addText(r.c2, { x: colX[2], y: ry, w: colW[2], h: 0.6, fontSize: 9, fontFace: FONT, color: C.inkSoft });
    s.addText(r.c3, { x: colX[3], y: ry, w: colW[3], h: 0.6, fontSize: 9, fontFace: FONT, color: C.inkSoft });
    s.addText(r.verdict, { x: colX[4], y: ry, w: colW[4], h: 0.6,
      fontSize: 10, fontFace: FONT, bold: true, color: r.color });
  });

  s.addText("* k=40+ Idefics2 PMR=1 은 \"continu\" stem matching 의 scoring artifact (슬라이드 30 + m5b_idefics2_non_monotonic.md 참고)", {
    x: 0.5, y: 6.5, w: 12.5, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.inkSoft, italic: true });
  s.addText("핵심: ladder 는 (1) circle vs ball 라벨 의존, (2) Qwen > Idefics2 ~ Next > LLaVA-1.5 > InternVL3 — 5-fold 시그니처와 일치.", {
    x: 0.5, y: 6.85, w: 12.5, h: 0.3,
    fontSize: 11, fontFace: FONT, bold: true, color: C.warm, italic: true });

  footer(s); pageNum(s, 29);
}

// ========================================================================
// SLIDE 30 — 3 NULL phenomena decomposition
// ========================================================================
{
  const s = content();
  stripe(s, "M 5 b   r o u n d   2 — N U L L   d e c o m p o s i t i o n");
  title(s, "Round 1 \"NULL\" 헤드라인이 3 가지 phenomena 로 분해",
    "Round 2 + binary PMR 한계 분석으로 \"NULL\" 의 의미가 명확해진다.");

  const items = [
    {
      tag: "(a)", title: "Genuine NULL",
      example: "InternVL3 (모든 cell + 모든 k ≤ 160)",
      detail: "Text 자체가 안 움직이고 binary PMR 도 PMR=1 유지. encoder + projector + LM 어디에도 ablate 가능한 commitment 없음. Super-saturated 의 진짜 NULL.",
      color: C.fail,
    },
    {
      tag: "(b)", title: "Baseline-already-abstract",
      example: "LLaVA-1.5 + circle cell",
      detail: "Baseline 응답이 \"drawn towards red arrow\" 로 PMR=0. 즉 모델이 이미 abstract regime 에 있어서 ablate 할 physics commitment 가 존재하지 않음. \"NULL\" 가 아니라 \"이미 깨진 상태\". CLIP+Vicuna 의 default mode for circle.",
      color: C.teal,
    },
    {
      tag: "(c)", title: "Binary-PMR conceals real shifts",
      example: "LLaVA-1.5 + ball cell",
      detail: "Text 는 fall → hit by arrow → roll → redrawn 으로 명확히 이동. 모든 응답이 motion verb 라 binary PMR 은 1 로 고정. 실제 regime 은 흔들리지만 측정 도구가 catch 못함.",
      color: C.warm,
    },
  ];

  let y = 2.6;
  items.forEach((it) => {
    box(s, 0.5, y, 12.3, 1.4, { round: 0.08, fill: C.paper, line: it.color, lw: 2 });
    s.addShape("rect", { x: 0.5, y: y, w: 1.0, h: 1.4,
      fill: { color: it.color }, line: { color: it.color } });
    s.addText(it.tag, { x: 0.5, y: y, w: 1.0, h: 1.4,
      fontSize: 24, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(it.title, { x: 1.6, y: y + 0.1, w: 11.0, h: 0.35,
      fontSize: 14, fontFace: FONT, bold: true, color: it.color });
    s.addText("예시: " + it.example, { x: 1.6, y: y + 0.45, w: 11.0, h: 0.3,
      fontSize: 11, fontFace: FONT, italic: true, color: C.ink });
    s.addText(it.detail, { x: 1.6, y: y + 0.75, w: 11.0, h: 0.65,
      fontSize: 10, fontFace: FONT, color: C.ink });
    y += 1.5;
  });

  s.addText("→ paper draft: \"NULL → regime-cross capacity ladder\" reframe + regime-shift score (text-distance) 보조 지표 권장.", {
    x: 0.5, y: 7.05, w: 12.5, h: 0.3,
    fontSize: 11, fontFace: FONT, bold: true, italic: true, color: C.warm });

  footer(s); pageNum(s, 30);
}

// ========================================================================
// SLIDE 31 — B5 today: Qwen 32B post-proj
// ========================================================================
{
  const s = content();
  stripe(s, "B 5   ( 2 0 2 6 - 0 5 - 0 1 )   —   Q w e n   3 2 B");
  title(s, "B5 today — Qwen 32B post-proj SAE replicates ★★★ tier",
    "32B 도 같은 regime-cross capacity. k 임계값만 약간 상승 (7B k=20 → 32B k=40). \"scaling 이 mechanism 을 변경하지 않는다\" 의 두 번째 증거.");

  table(s, 0.5, 2.6, 12.3,
    ["k_zeroed", "intervention text", "PMR", "해석"],
    [
      ["baseline", "\"The circle will move downward and land on the smaller gray shape below it.\"", { text: "1", bold: true }, "kinetic"],
      ["top_k = 20", "\"The circle will continue moving downward along the path indicated by the arrow.\"", { text: "1", bold: true, color: C.warm }, "still kinetic — 7B 와 다름"],
      ["top_k = 40", "\"The circle will remain stationary as there is no indication of movement or change.\"", { text: "0", bold: true, color: C.good }, "abstract regime ✅"],
      ["top_k = 80", "\"The circle will remain stationary as there is no indication of movement or change.\"", { text: "0", bold: true, color: C.good }, "stable abstract"],
      ["top_k = 160", "\"The circle will remain stationary as there is no indication of movement or change.\"", { text: "0", bold: true, color: C.good }, "stable abstract"],
      ["random_0/1/2", "\"... move downward and collide with surface below.\" / 동일", { text: "1", bold: true }, "specificity ✓"],
    ],
    [1.5, 7.0, 0.8, 3.0], { fontSize: 9.5, rowH: 0.55 });

  box(s, 0.5, 6.05, 12.3, 1.0, { round: 0.08, fill: C.cardCircle, line: C.good, title: "5-fold 시그니처에 32B 추가", titleColor: C.good });
  s.addText([
    { text: "• 7B 와 32B 가 ", options: { fontSize: 11 } },
    { text: "같은 ★★★ tier", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " — \"scale 이 mechanism 을 바꾸지 않는다\" 의 두 번째 증거 (§4.8 행동 지표에 더해 mechanism 도 보존).\n", options: { fontSize: 11 } },
    { text: "• 단, 32B 가 ", options: { fontSize: 11 } },
    { text: "k=40 까지 buffer 가 더 두꺼움", options: { fontSize: 11, bold: true } },
    { text: " — feature 가 더 많이 필요. 5120-feature SAE 위에서 0.4% (k=20) 가 아니라 0.8% (k=40).", options: { fontSize: 11 } },
  ], { x: 0.7, y: 6.4, w: 11.9, h: 0.6, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 31);
}

// ========================================================================
// SLIDE 32 — Pillar B motivation
// ========================================================================
{
  const s = content();
  stripe(s, "P I L L A R   B   #1 — M o t i v a t i o n");
  title(s, "Pillar B 동기 — 왜 controlled LM swap 이 필요한가",
    "LLaVA-1.5 (0.18) → LLaVA-Next (0.79) PMR jump 는 LM identity, AnyRes, SFT data, alignment 4 축이 동시에 다름.");

  box(s, 0.5, 2.7, 6.0, 4.4, { round: 0.08, fill: C.paper, line: C.bad, lw: 2, title: "LLaVA-1.5 ↔ LLaVA-Next 의 4 축 confound", titleColor: C.bad });
  s.addText([
    { text: "1. ", options: { fontSize: 12, bold: true } },
    { text: "LM 백본\n", options: { fontSize: 12, bold: true } },
    { text: "   Vicuna-7B → Mistral-7B-Instruct\n\n", options: { fontSize: 11 } },
    { text: "2. ", options: { fontSize: 12, bold: true } },
    { text: "Vision token 처리\n", options: { fontSize: 12, bold: true } },
    { text: "   단일-tile 576 → AnyRes (~4×4 grid)\n\n", options: { fontSize: 11 } },
    { text: "3. ", options: { fontSize: 12, bold: true } },
    { text: "SFT 데이터셋\n", options: { fontSize: 12, bold: true } },
    { text: "   LLaVA-Instruct-150K → LLaVA-Next 760K\n\n", options: { fontSize: 11 } },
    { text: "4. ", options: { fontSize: 12, bold: true } },
    { text: "Vision-language 정렬\n", options: { fontSize: 12, bold: true } },
    { text: "   단일 stage → 2-stage refresh\n\n", options: { fontSize: 11 } },
    { text: "→ \"인코더만 같다\" 한 가지로는 ", options: { fontSize: 11 } },
    { text: "LM 이 결정자라 결론 불가", options: { fontSize: 11, italic: true, color: C.bad } },
    { text: ".", options: { fontSize: 11 } },
  ], { x: 0.7, y: 3.15, w: 5.6, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.7, 6.1, 4.4, { round: 0.08, fill: C.cardCircle, line: C.primary, lw: 2, title: "M-LMSwap — 단일축 controlled swap" });
  s.addText([
    { text: "공통 component:\n", options: { fontSize: 12, bold: true } },
    { text: "  • CLIP-ViT-L-336 (LLaVA-1.5 와 동일)\n", options: { fontSize: 11 } },
    { text: "  • 2-layer MLP projector (랜덤 초기화)\n", options: { fontSize: 11 } },
    { text: "  • LoRA on q/k/v/o_proj (r=32, α=64)\n", options: { fontSize: 11 } },
    { text: "  • LCS-558K + LLaVA-Instruct-665K\n\n", options: { fontSize: 11 } },
    { text: "Variant A: ", options: { fontSize: 12, bold: true } },
    { text: "+ Vicuna-7B-v1.5\n", options: { fontSize: 11 } },
    { text: "Variant B: ", options: { fontSize: 12, bold: true } },
    { text: "+ Mistral-7B-Instruct-v0.2\n\n", options: { fontSize: 11 } },
    { text: "→ ", options: { fontSize: 11 } },
    { text: "LM identity 만 swap", options: { fontSize: 11, bold: true, color: C.primary } },
    { text: ", 나머지 모두 통제.\n", options: { fontSize: 11 } },
    { text: "→ A↔B Δ-PMR 이 ", options: { fontSize: 11 } },
    { text: "LM family 효과의 직접 측정", options: { fontSize: 11, italic: true, bold: true, color: C.primary } },
    { text: ".", options: { fontSize: 11 } },
  ], { x: 6.9, y: 3.15, w: 5.7, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 32);
}

// ========================================================================
// SLIDE 33 — M-LMSwap training pipeline
// ========================================================================
{
  const s = content();
  stripe(s, "P I L L A R   B   #2 — t r a i n i n g   p i p e l i n e");
  title(s, "M-LMSwap 학습 — 2-stage canonical recipe (Option 1)",
    "Stage 1 (~17K step / ~12h): projector pretrain on LCS-558K. Stage 2 (~21K step / ~12h): LoRA tune on LLaVA-Instruct-665K.");

  // pipeline diagram
  const stages = [
    { x: 0.5, label: "Stage 1\nprojector pretrain", sub: "LCS-558K · 17K\n~12h H200\nMLP only", color: C.primaryLight },
    { x: 3.6, label: "Stage 2\ninstruction tune", sub: "LLaVA-Instruct-665K\n21K step · ~12h\nLoRA(r=32, α=64) +\nMLP unfrozen", color: C.primary },
    { x: 6.7, label: "Final ckpt\nstep21000", sub: "MLP weight +\nLoRA adapter\n(merged on load)", color: C.teal },
    { x: 9.8, label: "Regression eval\n(3-gate)", sub: "PMR_nolabel +\nbaseline cell +\ngeneration sanity", color: C.warm },
  ];
  stages.forEach((st, i) => {
    box(s, st.x, 2.7, 3.0, 1.5, { round: 0.08, fill: st.color, line: st.color });
    s.addText(st.label, { x: st.x + 0.05, y: 2.75, w: 2.9, h: 0.5,
      fontSize: 12, fontFace: FONT, bold: true, color: "FFFFFF", align: "center" });
    s.addText(st.sub, { x: st.x + 0.05, y: 3.25, w: 2.9, h: 0.9,
      fontSize: 9.5, fontFace: FONT, color: "FFFFFF", align: "center", italic: true });
    if (i < stages.length - 1) {
      s.addText("→", { x: st.x + 3.0, y: 2.7, w: 0.15, h: 1.5,
        fontSize: 18, fontFace: FONT, bold: true, color: C.ink, align: "center", valign: "middle" });
    }
  });

  // Recipe details
  box(s, 0.5, 4.5, 6.0, 2.6, { round: 0.08, fill: C.paper, line: C.divider, title: "Recipe details (LLaVA-1.5 모방)" });
  s.addText([
    { text: "Stage 1 (projector pretrain):\n", options: { fontSize: 11, bold: true } },
    { text: "  • LR 1e-3, batch 32 (effective)\n", options: { fontSize: 10 } },
    { text: "  • Train MLP only (LM frozen)\n", options: { fontSize: 10 } },
    { text: "  • LCS-558K caption 데이터\n", options: { fontSize: 10 } },
    { text: "  • 17K step ≈ 1 epoch\n\n", options: { fontSize: 10 } },
    { text: "Stage 2 (instruction tune):\n", options: { fontSize: 11, bold: true } },
    { text: "  • LR 2e-4 (LoRA), batch 32\n", options: { fontSize: 10 } },
    { text: "  • LoRA on q/k/v/o_proj (r=32, α=64)\n", options: { fontSize: 10 } },
    { text: "  • MLP unfrozen after PEFT wrap\n", options: { fontSize: 10 } },
    { text: "  • LLaVA-Instruct-665K (Mix665k)\n", options: { fontSize: 10 } },
    { text: "  • 21K step ≈ 1 epoch", options: { fontSize: 10 } },
  ], { x: 0.7, y: 4.95, w: 5.6, h: 2.1, fontFace: FONT, color: C.ink, paraSpaceAfter: 1 });

  // Training timeline / artifacts
  box(s, 6.7, 4.5, 6.1, 2.6, { round: 0.08, fill: C.cardCircle, line: C.good, title: "학습 결과 (Variant A)", titleColor: C.good });
  s.addText([
    { text: "• Stage 1 50-step smoke: ", options: { fontSize: 10.5 } },
    { text: "loss 8.46 → 3.29 ✅", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "\n• Stage 1 17K step: ", options: { fontSize: 10.5 } },
    { text: "MLP grad-norm 안정 (NaN 없음)", options: { fontSize: 10.5, color: C.good } },
    { text: "\n• Stage 2 dry-run: ", options: { fontSize: 10.5 } },
    { text: "256 LoRA + 4 MLP grads, no leakage ✅", options: { fontSize: 10.5, color: C.good } },
    { text: "\n• Stage 2 21K step: ", options: { fontSize: 10.5 } },
    { text: "최종 ckpt step21000 저장 ✅", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "\n\n", options: { fontSize: 10.5 } },
    { text: "Bug fix mid-run:\n", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: "  get_mlp_ref 가 hasattr(\"base_model\") 사용 — HF\n", options: { fontSize: 10 } },
    { text: "  PreTrainedModel 에서는 항상 True 라 잘못된 path.\n", options: { fontSize: 10 } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "isinstance(model, PeftModel) 로 수정", options: { fontSize: 10, bold: true } },
    { text: ".\n\n다음 단계: regression eval (슬라이드 34).", options: { fontSize: 10 } },
  ], { x: 6.9, y: 4.95, w: 5.7, h: 2.1, fontFace: FONT, color: C.ink, paraSpaceAfter: 1 });

  footer(s); pageNum(s, 33);
}

// ========================================================================
// SLIDE 34 — Variant A regression: gate split
// ========================================================================
{
  const s = content();
  stripe(s, "P I L L A R   B   #3 — r e g r e s s i o n   e v a l");
  title(s, "Variant A regression — gate split (step9000 vs step21000)",
    "Aggregate PMR FAIL (recipe drift), 그러나 step21000 line/blank/none baseline = 0.000 PASS — cell-discrimination 학습됨.");

  table(s, 0.5, 2.6, 12.3,
    ["Gate", "step9000", "step21000", "변화"],
    [
      ["Gate 1 — generation sanity (>5 단어, no degeneracy)", { text: "PASS", bold: true, color: C.good }, { text: "PASS", bold: true, color: C.good }, "—"],
      ["Gate 2 — PMR_nolabel ∈ [0.03, 0.50]", { text: "FAIL (0.825)", bold: true, color: C.fail }, { text: "FAIL (0.869)", bold: true, color: C.fail }, "약간 더 높아짐"],
      ["Gate 3 — line/blank/none baseline ≤ 0.6", { text: "FAIL (1.000)", bold: true, color: C.fail }, { text: "PASS (0.000) ✅", bold: true, color: C.good }, { text: "−1.000 pp! discrimination 학습", bold: true, color: C.good }],
    ],
    [4.5, 2.5, 2.5, 2.8], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 4.4, 12.3, 1.5, { round: 0.08, fill: C.cardCircle, line: C.good, title: "step21000 line/blank/none 응답 예시 — 명백한 abstract regime", titleColor: C.good });
  s.addText("\"A circle is drawn on a white background. It is not clear what will happen next. " +
    "It could be a new circle drawn, or it could be a different shape.\"", {
    x: 0.7, y: 4.85, w: 11.9, h: 1.0,
    fontSize: 12, fontFace: FONT, italic: true, color: C.ink });

  box(s, 0.5, 6.0, 12.3, 1.1, { round: 0.08, fill: C.accentSoft, line: C.warm, title: "해석 — A↔B 비교 의미가 회복", titleColor: C.warm });
  s.addText([
    { text: "step21000 은 ", options: { fontSize: 11 } },
    { text: "cell-discrimination 을 학습", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " — 가장 추상적 cell 에서는 abstract response, cue 강한 cell 에서는 kinetic.\n", options: { fontSize: 11 } },
    { text: "Per-cell ordering 이 LLaVA-1.5 와 ", options: { fontSize: 11 } },
    { text: "구조적으로 일치", options: { fontSize: 11, bold: true, color: C.good } },
    { text: " (line ≈ 0, filled high, shaded high). Aggregate 만 +0.4 shifted.\n", options: { fontSize: 11 } },
    { text: "→ A↔B Δ-PMR 비교가 ", options: { fontSize: 11 } },
    { text: "per-cell 수준에서 의미 있다", options: { fontSize: 11, italic: true, bold: true, color: C.warm } },
    { text: ". A1 결정 (Variant B 진행) 권장.", options: { fontSize: 11 } },
  ], { x: 0.7, y: 6.4, w: 11.9, h: 0.65, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 34);
}

// ========================================================================
// SLIDE 35 — Recipe drift hypothesis
// ========================================================================
{
  const s = content();
  stripe(s, "P I L L A R   B   #4 — r e c i p e   d r i f t");
  title(s, "왜 recipe 가 drift 했는가 — 3 가지 후보",
    "LLaVA-1.5 의 published recipe 를 모방했지만 PMR 천장이 0.4 더 높이 위치. 어떤 design choice 가 책임지는가?");

  const candidates = [
    {
      tag: "1", title: "Fresh MLP 랜덤 init",
      detail: "LLaVA-1.5 는 특정 init seed + 정밀 LR 사용. 우리 implementation 은 fresh random init + 자체 LR (1e-3 stage 1). 같은 LCS-558K 위에서도 다른 minima 로 수렴 가능.",
      severity: "high",
    },
    {
      tag: "2", title: "LoRA-only Stage 2 (vs full LM tune)",
      detail: "LLaVA-1.5 Stage 2 는 전체 LM 을 tune. 우리는 LoRA on q/k/v/o_proj 만. LoRA 는 보수적 — Vicuna 의 강한 physics-language prior 를 충분히 억제 못함 → PMR 천장 high.",
      severity: "high",
    },
    {
      tag: "3", title: "Chat template / image token 처리 차이",
      detail: "LLaVA-1.5 는 Vicuna chat template 그대로 + literal `<image>`. 우리는 `processor.apply_chat_template` (inference) + manual Vicuna template (training). 이 미묘한 차이가 vision-language 정렬에 작은 shift 를 만들 수 있음.",
      severity: "medium",
    },
  ];

  let y = 2.6;
  candidates.forEach((c) => {
    const sev = c.severity === "high" ? C.bad : C.warm;
    box(s, 0.5, y, 12.3, 1.3, { round: 0.08, fill: C.paper, line: sev, lw: 1.5 });
    s.addShape("rect", { x: 0.5, y: y, w: 0.7, h: 1.3, fill: { color: sev }, line: { color: sev } });
    s.addText(c.tag, { x: 0.5, y: y, w: 0.7, h: 1.3,
      fontSize: 28, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(c.title, { x: 1.3, y: y + 0.1, w: 11.4, h: 0.35,
      fontSize: 13, fontFace: FONT, bold: true, color: sev });
    s.addText(c.detail, { x: 1.3, y: y + 0.45, w: 11.4, h: 0.85,
      fontSize: 10, fontFace: FONT, color: C.ink });
    s.addText(c.severity, { x: 11.5, y: y + 0.1, w: 1.2, h: 0.3,
      fontSize: 10, fontFace: FONT, bold: true, color: sev, italic: true, align: "right" });
    y += 1.4;
  });

  s.addText("결정 — 두 옵션 모두 한 번에 테스트 못함. step21000 의 cell-discrimination 학습 결과로 A1 (recipe 그대로 + Variant B) 진행이 더 합리적. " +
    "A2 (recipe 재조정 후 재학습) 는 gate fail 이 paper-blocker 가 될 때만 trigger.", {
    x: 0.5, y: 7.0, w: 12.5, h: 0.3,
    fontSize: 10, fontFace: FONT, italic: true, color: C.ink });

  footer(s); pageNum(s, 35);
}

// ========================================================================
// SLIDE 36 — A1/A2/A3 decision matrix
// ========================================================================
{
  const s = content();
  stripe(s, "P I L L A R   B   #5 — d e c i s i o n");
  title(s, "결정 — A1 / A2 / A3 비교 (현재 권장: A1)",
    "step21000 baseline=0 PASS 가 A1 의 원래 risk 를 줄임. A↔B Δ-PMR 가 의미 있는 측정 가능.");

  const decisions = [
    {
      tag: "A1", title: "Variant B 진행 (gate override)",
      cost: "GPU 24h",
      pros: "+ A↔B per-cell Δ-PMR 비교 가능 (axis ordering 일치)\n+ step21000 baseline=0 으로 A1 risk 감소",
      cons: "- A baseline aggregate +0.4 shifted (Δ 비교만 valid)",
      color: C.good, recommended: true,
    },
    {
      tag: "A2", title: "Variant A recipe 재학습",
      cost: "GPU 24h, 확률적",
      pros: "+ LLaVA-1.5 floor 와 정렬 가능 → aggregate 비교도",
      cons: "- 또 실패할 수 있음 (slide 35 1번/2번 hypothesis 가 맞으면 LR/data 조정만으로 부족)\n- 시간 손실",
      color: C.warm, recommended: false,
    },
    {
      tag: "A3", title: "M-PSwap (perceiver swap) 부활",
      cost: "GPU 24h+, NaN 미해결",
      pros: "+ Idefics2 perceiver-resampler 가설 직접 검증 (G3 fix)",
      cons: "- 학습 안정화가 우선 미해결 (NaN at step 1000, 다중 diagnostic 세션 fail)\n- backlogged 상태",
      color: C.teal, recommended: false,
    },
  ];

  let y = 2.6;
  decisions.forEach((d) => {
    box(s, 0.5, y, 12.3, 1.4, { round: 0.08, fill: C.paper, line: d.color, lw: d.recommended ? 2.5 : 1 });
    s.addShape("rect", { x: 0.5, y: y, w: 0.9, h: 1.4, fill: { color: d.color }, line: { color: d.color } });
    s.addText(d.tag, { x: 0.5, y: y, w: 0.9, h: 1.4,
      fontSize: 28, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(d.title + (d.recommended ? "  ★ 추천" : ""), { x: 1.5, y: y + 0.1, w: 11.2, h: 0.3,
      fontSize: 13, fontFace: FONT, bold: true, color: d.color });
    s.addText(d.cost, { x: 1.5, y: y + 0.4, w: 11.2, h: 0.25,
      fontSize: 10, fontFace: FONT, italic: true, color: C.inkSoft });
    s.addText(d.pros, { x: 1.5, y: y + 0.65, w: 11.2, h: 0.4,
      fontSize: 10, fontFace: FONT, color: C.good });
    s.addText(d.cons, { x: 1.5, y: y + 1.0, w: 11.2, h: 0.4,
      fontSize: 10, fontFace: FONT, color: C.bad });
    y += 1.5;
  });

  footer(s); pageNum(s, 36);
}

// ========================================================================
// SLIDE 37 — FAILURE: M-PSwap NaN
// ========================================================================
{
  const s = content();
  stripe(s, "F A I L U R E   #1 — M - P S w a p", C.fail);
  title(s, "M-PSwap (perceiver swap) — NaN at step 1000, backlogged",
    "Pillar B 의 원래 우선순위. Idefics2 perceiver-resampler ↔ MLP swap 으로 §4.6 Idefics2 anomaly 직접 검증 목적이었으나 학습 NaN 미해결.");

  table(s, 0.5, 2.6, 12.3,
    ["단계", "결과", "특이사항"],
    [
      ["Feasibility spike (bypass-only)", { text: "FAIL", bold: true, color: C.fail }, "perceiver 가 forward pass 에 integral — bypass 로는 안 됨"],
      ["Infra 구축 (LoRA + MLPPoolResampler fp32)", { text: "✅", bold: true, color: C.good }, "src/physical_mode/lora/{idefics2_mlp_resampler.py, load_swapped.py}"],
      ["50-step smoke", { text: "PASS", bold: true, color: C.good }, "loss 정상 하강, NaN 없음"],
      ["Full training step 0 → 1000", { text: "PASS", bold: true, color: C.good }, "안정적"],
      ["Full training step 1000 NaN", { text: "FAIL", bold: true, color: C.fail }, "NaN-abort logic 작동, run aborted at outputs/mpswap_run_20260429-033238/step1000"],
      ["Diagnostic suite (NaN reproducer)", { text: "WIP", bold: true, color: C.warm }, "long-text / streaming / bf16 / mask 패턴 stress test — 미재현"],
      ["D0a bf16 forward-only repro (M-LMSwap day-0)", { text: "FAIL to reproduce", color: C.warm }, "step 1465 까지 clean — single-bad-batch 가설 약화"],
    ],
    [4.0, 2.0, 6.3], { fontSize: 9.5, rowH: 0.45 });

  box(s, 0.5, 5.95, 12.3, 1.1, { round: 0.08, fill: C.failSoft, line: C.fail, title: "현재 status — backlogged", titleColor: C.fail });
  s.addText([
    { text: "결정 (2026-04-29): ", options: { fontSize: 10.5, bold: true } },
    { text: "submission_plan.md §6 Pillar-B drop rule 을 ", options: { fontSize: 10.5 } },
    { text: "조기 적용", options: { fontSize: 10.5, italic: true, bold: true, color: C.warm } },
    { text: " (week 8 → week 4). 인프라 보존, NaN 진단 일시 중단, M-LMSwap 으로 우선순위 이전.\n", options: { fontSize: 10.5 } },
    { text: "교훈: ", options: { fontSize: 10.5, bold: true } },
    { text: "perceiver-resampler 가 forward pass 에 integral 한 architecture 는 swap-style controlled experiment 가 더 어렵다. 다음 시도 시 ", options: { fontSize: 10.5 } },
    { text: "discriminator 기반 NaN repro batch", options: { fontSize: 10.5, italic: true } },
    { text: " 가 필요.", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 6.4, w: 11.9, h: 0.65, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 37);
}

// ========================================================================
// SLIDE 38 — FAILURE: M-MP Qwen × MCQ split
// ========================================================================
{
  const s = content();
  stripe(s, "F A I L U R E   /   F I N D I N G   #2 — M - M P   c r o s s - m e t h o d   s p l i t", C.warm);
  title(s, "M-MP Phase 3 — Qwen × MCQ 에서 M5a / M5b 가 갈린다",
    "M5a (steering): MCQ NULL (yesno 와 같음). M5b (SAE): MCQ break (describe 와 같음). \"Generative-vs-Categorical\" framing 분해.");

  table(s, 0.5, 2.6, 12.3,
    ["프롬프트", "M5a (steering α=40)", "M5b (SAE k=20)", "패턴"],
    [
      ["open (existing baseline)", { text: "10/10 flip ✅", color: C.good }, { text: "20/20 break ✅", color: C.good }, "positive/positive"],
      ["describe_scene (generative)", { text: "10/10 flip ✅", color: C.good }, { text: "20/20 break ✅", color: C.good }, "positive/positive"],
      ["meta_phys_yesno (categorical)", { text: "0/10 ✗", color: C.fail }, { text: "0/20 ✗", color: C.fail }, "null/null"],
      ["meta_phys_mcq (audit follow-up)", { text: "0/10 ✗", color: C.fail }, { text: "10/10 break ✅", color: C.good }, { text: "split! null/positive", bold: true, color: C.warm }],
    ],
    [3.5, 3.0, 3.0, 2.8], { fontSize: 10, rowH: 0.5 });

  box(s, 0.5, 5.4, 12.3, 1.7, { round: 0.08, fill: C.accentSoft, line: C.warm, title: "Reframe — \"Generative-vs-Categorical\" framing 이 부족", titleColor: C.warm });
  s.addText([
    { text: "초기 framing: ", options: { fontSize: 10.5 } },
    { text: "\"M5a/M5b 는 generative prompt 에서만 작동, categorical 에서는 NULL\".  ", options: { fontSize: 10.5, italic: true } },
    { text: "MCQ 가 framing 분해.\n", options: { fontSize: 10.5 } },
    { text: "관측: ", options: { fontSize: 10.5, bold: true } },
    { text: "MCQ 는 categorical task 인데 M5b 가 break.  M5a 는 NULL.  → \"task type\" 이 아니라 ", options: { fontSize: 10.5 } },
    { text: "intervention method × prompt format", options: { fontSize: 10.5, italic: true, bold: true, color: C.warm } },
    { text: " 의 interaction.\n", options: { fontSize: 10.5 } },
    { text: "현재 입장 (audit-tightened): ", options: { fontSize: 10.5, bold: true } },
    { text: "\"yes/no 의 M5b immunity 는 yes/no-prompt-specific (n=1 categorical-binary)\". 더 많은 categorical-binary prompt 로 axis 확정 필요.", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 5.85, w: 11.9, h: 1.2, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 38);
}

// ========================================================================
// SLIDE 39 — FAILURE: M-MP Idefics2 single-cell + audit caveat
// ========================================================================
{
  const s = content();
  stripe(s, "F A I L U R E   /   F I N D I N G   #3 — M - M P   I d e f i c s 2", C.warm);
  title(s, "M-MP Idefics2 — single-cell finding + audit caveat",
    "단일 cell 에서 30/30 framing-shift (kinetic→suspended). 그러나 audit 결과 \"1 cell × 10 stim × 3 k\" 로 reframe — architecture-level 아님.");

  box(s, 0.5, 2.6, 6.0, 4.5, { round: 0.08, fill: C.cardCircle, line: C.good, title: "원래 발견 (2026-04-28 morning)", titleColor: C.good });
  s.addText([
    { text: "Cell: ", options: { fontSize: 11, bold: true } },
    { text: "shaded/ground/both (ball)\n\n", options: { fontSize: 11 } },
    { text: "Top-k SAE ablation:\n", options: { fontSize: 11, bold: true } },
    { text: "  baseline: \"The ball is falling.\"\n", options: { fontSize: 10, italic: true } },
    { text: "  k=160: \"The ball is in the air.\"\n", options: { fontSize: 10, italic: true } },
    { text: "  k=320: \"The ball is in the air.\"\n", options: { fontSize: 10, italic: true } },
    { text: "  k=500: \"The ball is in the air.\"\n\n", options: { fontSize: 10, italic: true } },
    { text: "관측: ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "framing shift kinetic → suspended\n", options: { fontSize: 11, color: C.good } },
    { text: "  • 10 stim 모두 동일 intervention text\n", options: { fontSize: 10 } },
    { text: "  • Random 10/10 retains kinetic\n", options: { fontSize: 10 } },
    { text: "  • → SAE features encode kinetic-verb\n", options: { fontSize: 10 } },
    { text: "      production specifically (specificity ✓)\n\n", options: { fontSize: 10 } },
    { text: "초기 framing: ", options: { fontSize: 10.5 } },
    { text: "\"30/30 stim across 3 k values\"\n", options: { fontSize: 10.5, italic: true } },
    { text: "(advisor 지적 후 reframe).", options: { fontSize: 10 } },
  ], { x: 0.7, y: 3.05, w: 5.6, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.5, { round: 0.08, fill: C.failSoft, line: C.fail, title: "Audit caveat (2026-04-28 review)", titleColor: C.fail });
  s.addText([
    { text: "Reframe:\n", options: { fontSize: 11, bold: true } },
    { text: "  \"30 independent stim\" → \"1 cell × 10 stim × 3 k replicates\"\n\n", options: { fontSize: 10 } },
    { text: "Cell 1 audit caveats:\n", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "  • 1 unique intervention text 만 (\"in the air\")\n", options: { fontSize: 10 } },
    { text: "    → diversity 부족\n", options: { fontSize: 10 } },
    { text: "  • verbosity-preference confound 가능\n", options: { fontSize: 10 } },
    { text: "  • 2x2 (task × format) 에서 1 cell 만 채워짐\n\n", options: { fontSize: 10 } },
    { text: "Cell 2 test (audit follow-up):\n", options: { fontSize: 11, bold: true } },
    { text: "  textured/ground/cast_shadow ball cell\n", options: { fontSize: 10 } },
    { text: "  → ", options: { fontSize: 10 } },
    { text: "baseline 자체가 \"ball is in the air\" (suspended)", options: { fontSize: 10, italic: true } },
    { text: "\n  → top-k ablation no-op\n  → ", options: { fontSize: 10 } },
    { text: "specificity 재확인", options: { fontSize: 10, italic: true, color: C.good } },
    { text: " 하지만 cell 1 의 architecture-level 격상은 ", options: { fontSize: 10 } },
    { text: "보류", options: { fontSize: 10, bold: true, color: C.fail } },
    { text: ".\n\n→ 3rd cell with kinetic baseline 필요.", options: { fontSize: 10 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 39);
}

// ========================================================================
// SLIDE 40 — FAILURE: Phase 2 zombie polling + recipe drift
// ========================================================================
{
  const s = content();
  stripe(s, "F A I L U R E   #4 — i n f r a   l e s s o n s", C.fail);
  title(s, "Infra 학습 — zombie polling 5h hang + recipe drift 발견",
    "두 인프라 사건이 paper 수준 deliverable 에 직접 영향. 두 사건 모두 정확한 root cause + fix 적용으로 recover.");

  // Event 1
  box(s, 0.5, 2.6, 12.3, 2.1, { round: 0.08, fill: C.failSoft, line: C.fail, lw: 2, title: "Event 1: Phase 2 zombie polling 5h hang (2026-04-30)", titleColor: C.fail });
  s.addText([
    { text: "Setup: ", options: { fontSize: 10.5, bold: true } },
    { text: "chain_post_proj_phase2.sh 가 ", options: { fontSize: 10.5 } },
    { text: "kill -0 $PID", options: { fontSize: 10.5, fontFace: "Courier New" } },
    { text: " 로 prior chain 의 종료를 대기.\n", options: { fontSize: 10.5 } },
    { text: "버그: ", options: { fontSize: 10.5, bold: true, color: C.fail } },
    { text: "kill -0", options: { fontSize: 10.5, fontFace: "Courier New" } },
    { text: " 가 ", options: { fontSize: 10.5 } },
    { text: "Z-state defunct (zombie) 프로세스에서도 success", options: { fontSize: 10.5, bold: true } },
    { text: ". prior chain 이 zombie 로 남으면 polling 무한 대기.\n", options: { fontSize: 10.5 } },
    { text: "결과: ", options: { fontSize: 10.5 } },
    { text: "5 시간 GPU 0 idle", options: { fontSize: 10.5, bold: true, color: C.fail } },
    { text: " — Phase 2 작업 시작 안 됨.\n", options: { fontSize: 10.5 } },
    { text: "Fix: ", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: "[ -e /proc/$PID ] && ! grep -q \"^State.*Z\" /proc/$PID/status", options: { fontSize: 10.5, fontFace: "Courier New" } },
    { text: " 로 실제 alive 체크.\n", options: { fontSize: 10.5 } },
    { text: "교훈: ", options: { fontSize: 10.5 } },
    { text: "long polling chain 에서 zombie 가 silent failure mode", options: { fontSize: 10.5, italic: true, bold: true, color: C.warm } },
    { text: " — race condition 처럼 \"평소엔 안 보이는 버그\".", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 3.05, w: 11.9, h: 1.65, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  // Event 2
  box(s, 0.5, 4.85, 12.3, 2.2, { round: 0.08, fill: C.accentSoft, line: C.warm, lw: 2, title: "Event 2: Variant A regression gate FAIL — recipe drift 발견 (2026-05-01)", titleColor: C.warm });
  s.addText([
    { text: "Setup: ", options: { fontSize: 10.5, bold: true } },
    { text: "Variant A 학습 21K step 완료 후 m_lmswap_regression_eval.py 의 [0.03, 0.50] gate.\n", options: { fontSize: 10.5 } },
    { text: "관측: ", options: { fontSize: 10.5, bold: true, color: C.fail } },
    { text: "PMR_nolabel = 0.825 (step9000) / 0.869 (step21000) — gate 2 FAIL.", options: { fontSize: 10.5, bold: true } },
    { text: "  Aggregate 0.4 too high.\n", options: { fontSize: 10.5 } },
    { text: "분석: ", options: { fontSize: 10.5, bold: true } },
    { text: "그러나 (1) 480 stim 중 201 unique (image-blind 아님), (2) line→filled→shaded ramp 가 LLaVA-1.5 와 동일, (3) ", options: { fontSize: 10.5 } },
    { text: "step21000 line/blank/none 에서 PMR=0.000 (cell-discrimination 학습)", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: ".\n", options: { fontSize: 10.5 } },
    { text: "재해석: ", options: { fontSize: 10.5, bold: true, color: C.warm } },
    { text: "Recipe drift", options: { fontSize: 10.5, italic: true, bold: true } },
    { text: " — 우리 implementation 이 LLaVA-1.5 보다 더 physics-leaning 으로 수렴. ", options: { fontSize: 10.5 } },
    { text: "A↔B 통제는 양쪽에 동일 drift 적용 → 비교 의미 보존", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: ".\n", options: { fontSize: 10.5 } },
    { text: "교훈: ", options: { fontSize: 10.5 } },
    { text: "single gate (aggregate PMR) 만 보면 \"실패\" 처럼 보이지만, ", options: { fontSize: 10.5, italic: true } },
    { text: "per-cell pattern 을 봐야 진짜 status 가 보임", options: { fontSize: 10.5, italic: true, bold: true, color: C.warm } },
    { text: ".", options: { fontSize: 10.5 } },
  ], { x: 0.7, y: 5.3, w: 11.9, h: 1.7, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 40);
}

// ========================================================================
// SLIDE 41 — 5-fold downstream signature
// ========================================================================
{
  const s = content();
  stripe(s, "S Y N T H E S I S   #1 — 5 - f o l d   s i g n a t u r e");
  title(s, "5겹 시그니처 — 같은 architecture clustering 이 5 가지 다른 측정으로 보존",
    "Single architectural property 의 redundant manifestation. paper 의 strongest claim.");

  table(s, 0.5, 2.6, 12.3,
    ["#", "지표", "non-CLIP cluster", "CLIP cluster", "스토리"],
    [
      ["1", "PMR ceiling (M2 _nolabel)", { text: "[0.84, 0.92]", color: C.good }, { text: "[0.14, 0.37]", color: C.fail }, "행동 1 — 5.5× separation"],
      ["2", "Decision-stability (RC)", { text: "≥ 0.95", color: C.good }, "≤ 0.7", "행동 2 — n=5 seed 안정성"],
      ["3", "픽셀 encodability (§4.6)", { text: "Qwen broad", color: C.good }, "weak (LLaVA-1.5 4/10)", "메커니즘 1 — pixel routability"],
      ["4", "LM logit-lens AUC (M4)", { text: "Qwen 0.96 / Idefics2 0.995", color: C.good }, "0.76-0.79", "메커니즘 2 — LM probe"],
      ["5", "Encoder SAE break (M5b r1)", { text: "3 of 3 break", color: C.good }, { text: "2 of 2 NULL on ball", color: C.fail }, "메커니즘 3 — feature ablation"],
      [{ text: "5+", color: C.accent, bold: true }, { text: "Post-proj regime-cross (M5b r2)", color: C.accent }, { text: "Qwen ★★★ + 32B + Idefics2 ★★", color: C.good }, { text: "LLaVA-1.5 ✦ baseline-abstract\nLLaVA-Next ★★ partial", color: C.warm }, { text: "메커니즘 4 — 오늘 추가 (B5)", color: C.accent }],
    ],
    [0.4, 3.0, 3.0, 2.7, 3.2], { fontSize: 9, rowH: 0.5 });

  box(s, 0.5, 6.0, 12.3, 1.0, { round: 0.08, fill: C.cardCircle, line: C.warm, title: "Strongest claim", titleColor: C.warm });
  s.addText("5(+1) 개의 별도 측정 방식이 모두 같은 architectural clustering 을 가리킨다 → " +
    "단일 architectural property 의 redundant manifestation.  단순 \"인코더 capacity\" 가설로는 설명 불가.", {
    x: 0.7, y: 6.4, w: 11.9, h: 0.55,
    fontSize: 11, fontFace: FONT, italic: true, bold: true, color: C.ink });

  footer(s); pageNum(s, 41);
}

// ========================================================================
// SLIDE 42 — Encoder vs LM dissociation map
// ========================================================================
{
  const s = content();
  stripe(s, "S Y N T H E S I S   #2 — d i s s o c i a t i o n   m a p");
  title(s, "Encoder ↔ LM dissociation map (5 모델 × 측정)",
    "LM-side flip 가능 여부 + Encoder-side commitment 존재 여부의 2 × 2 분류.  CLIP family 와 non-CLIP family 로 갈린다.");

  table(s, 0.5, 2.6, 12.3,
    ["모델", "M3 vision AUC (encoder)", "M4 LM AUC", "M5a steering (LM-side flip)", "M5b encoder SAE", "분류"],
    [
      ["Qwen2.5-VL", "0.99", "0.96", { text: "✅ L10 α=40", color: C.good }, { text: "✅ k=20 break", color: C.good }, "encoder + LM 둘 다"],
      ["Idefics2", "0.93", "0.995", { text: "✅ L25 α=20", color: C.good }, { text: "✅ k=160 break", color: C.good }, "encoder + LM 둘 다"],
      ["InternVL3", "0.89", "untestable", { text: "untestable (sat)", color: C.inkSoft }, { text: "✅ k=160 break", color: C.good }, "encoder + saturated"],
      ["LLaVA-Next", "0.81", "0.79", { text: "✅ L20+L25", color: C.good }, { text: "✗ encoder NULL", color: C.fail }, "LM only"],
      ["LLaVA-1.5", "0.73", "0.76", { text: "✗ L25 α=0~60", color: C.fail }, { text: "✗ encoder NULL", color: C.fail }, "neither (encoder bottleneck)"],
    ],
    [2.0, 1.8, 1.5, 2.5, 2.3, 2.2], { fontSize: 9.5, rowH: 0.5 });

  box(s, 0.5, 5.6, 12.3, 1.5, { round: 0.08, fill: C.paper, line: C.divider, title: "구조" });
  s.addText([
    { text: "Non-CLIP family (Qwen / Idefics2 / InternVL3): ", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "encoder 안에 ~30 SAE feature 로 commitment 국소화. LM-side direction 도 작동.\n", options: { fontSize: 11 } },
    { text: "LLaVA-Next (CLIP + Mistral + AnyRes): ", options: { fontSize: 11, bold: true, color: C.warm } },
    { text: "encoder NULL but LM 작동 — physics commitment 가 LM 안에서만 라우팅.\n", options: { fontSize: 11 } },
    { text: "LLaVA-1.5 (CLIP + Vicuna): ", options: { fontSize: 11, bold: true, color: C.fail } },
    { text: "encoder NULL + LM 도 안 흔들림 — encoder 가 진짜 bottleneck (행동 PMR 0.18 의 직접 원인).", options: { fontSize: 11 } },
  ], { x: 0.7, y: 6.05, w: 11.9, h: 1.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 3 });

  footer(s); pageNum(s, 42);
}

// ========================================================================
// SLIDE 43 — Limitations + paper plan
// ========================================================================
{
  const s = content();
  stripe(s, "L I M I T A T I O N S   +   P A P E R   P L A N");
  title(s, "한계 + paper plan (Track B, ICLR 2027)",
    "4 개 paper gap (G1-G4) 매핑 + 5 개 한계.  현재 Pillar B 진행 중, M-PSwap 은 backlogged.");

  box(s, 0.5, 2.6, 6.0, 4.5, { round: 0.08, fill: C.paper, line: C.bad, title: "5 한계 (limitations)", titleColor: C.bad });
  s.addText([
    { text: "1. Single-task evaluation (G1)\n", options: { fontSize: 10.5, bold: true } },
    { text: "  next-state-prediction 만 검증. Counting / spatial /\n", options: { fontSize: 10 } },
    { text: "  causality 등 미검증. M-MP Pillar A 가 부분 fix.\n\n", options: { fontSize: 10 } },
    { text: "2. Sparse non-Qwen (G2)\n", options: { fontSize: 10.5, bold: true } },
    { text: "  4 비-Qwen 모델만 (LLaVA-1.5/Next/Idefics2/InternVL3).\n", options: { fontSize: 10 } },
    { text: "  Pixtral / Phi-3.5-V / GPT-4V 미테스트. → B4 (오늘).\n\n", options: { fontSize: 10 } },
    { text: "3. n=1 perceiver isolation (G3)\n", options: { fontSize: 10.5, bold: true } },
    { text: "  Idefics2 가 유일 perceiver. controlled swap 미검증\n", options: { fontSize: 10 } },
    { text: "  (M-PSwap NaN). → B2 fallback (literature-grounded).\n\n", options: { fontSize: 10 } },
    { text: "4. 5-fold framing 명확성 (G4)\n", options: { fontSize: 10.5, bold: true } },
    { text: "  paper §6 Marr 3-level 재구성 필요. → M-Marr (Pillar C).\n\n", options: { fontSize: 10 } },
    { text: "5. Human baseline 미수집\n", options: { fontSize: 10.5, bold: true } },
    { text: "  Prolific 20 raters × 50 stim. paper-blocking 단계 (M7).", options: { fontSize: 10 } },
  ], { x: 0.7, y: 3.05, w: 5.6, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  box(s, 6.7, 2.6, 6.1, 4.5, { round: 0.08, fill: C.cardCircle, line: C.primary, title: "Track B (ICLR 2027) timeline" });
  s.addText([
    { text: "선택 venue (2026-04-28 결정):\n", options: { fontSize: 10.5, bold: true } },
    { text: "  • ICLR 2027 primary (deadline ~late Sep 2026)\n", options: { fontSize: 10 } },
    { text: "  • NeurIPS 2027 secondary (~mid May 2027)\n", options: { fontSize: 10 } },
    { text: "  • TMLR rolling fallback\n\n", options: { fontSize: 10 } },
    { text: "Framing:\n", options: { fontSize: 10.5, bold: true } },
    { text: "  Production-VLM \"world-model commitment\"\n", options: { fontSize: 10 } },
    { text: "  3 Marr levels: Computational (PMR) →\n", options: { fontSize: 10 } },
    { text: "  Representational (M3+M4) →\n", options: { fontSize: 10 } },
    { text: "  Mechanistic (M5a+M5b)\n", options: { fontSize: 10 } },
    { text: "  Connect: V-JEPA / RT-2 / OpenVLA in §1+§9.\n\n", options: { fontSize: 10 } },
    { text: "Drop / defer rules:\n", options: { fontSize: 10.5, bold: true } },
    { text: "  • week 8 Pillar B 결과 없으면 → M-PSwap drop, B2 lean.\n", options: { fontSize: 10 } },
    { text: "  • week 14 still missing → ICLR 2027 → NeurIPS 2027 (+7 month).\n", options: { fontSize: 10 } },
    { text: "  • Robust-not-flashy stance (2026-04-28).", options: { fontSize: 10 } },
  ], { x: 6.9, y: 3.05, w: 5.7, h: 4.0, fontFace: FONT, color: C.ink, paraSpaceAfter: 2 });

  footer(s); pageNum(s, 43);
}

// ========================================================================
// SLIDE 44 — Next priorities (B4 / B1 / A1/A2)
// ========================================================================
{
  const s = content();
  stripe(s, "N E X T   —   p r i o r i t i e s");
  title(s, "다음 priorities — B4 (Pixtral) → B1 (multi-cell n) → A1/A2 결정",
    "GPU 작업 + paper 작업 병렬 진행. B4·B1 은 today's PPT 종료 후 자동 시작 예정.");

  const priorities = [
    {
      tag: "B4", title: "Pixtral chat template fix + M-Add6 6th model",
      cost: "~30 min dev + 5 min infer",
      desc: "FB3 시도 시 jinja TypeError 발견. 수정 후 m_add6_pixtral_m8a inference + score. G2 (sparse non-Qwen) 의 6th data point.",
      color: C.good, status: "next",
    },
    {
      tag: "B1", title: "Multi-cell aggregation n=40 intervention (5-model)",
      cost: "~3-4 h GPU",
      desc: "filled+blank+{none, cast_shadow, motion_arrow, both} = 4 cells × 10 stim = 40 stim per model. Regime-cross capacity ladder 의 statistical strength 강화.",
      color: C.good, status: "after B4",
    },
    {
      tag: "A1 vs A2", title: "Pillar B 결정 (Variant B 진행 vs A 재학습)",
      cost: "GPU 24h",
      desc: "step21000 baseline=0 PASS 가 A1 risk 줄임 (slide 36). Per-cell Δ-PMR 비교가 의미 있는 측정.  A1 추천.",
      color: C.warm, status: "user decision",
    },
    {
      tag: "C2-C7", title: "Doc consolidation (이미 완료)",
      cost: "—",
      desc: "✅ insight m5b_post_projection_cross_model.md / m5b_idefics2_non_monotonic.md / lmswap_a_recipe_drift.md\n✅ hypotheses.md H-regime-cross 추가 ✅ paper_gaps.md / roadmap.md / CHANGELOG.md 업데이트",
      color: C.teal, status: "done",
    },
  ];

  let y = 2.6;
  priorities.forEach((p) => {
    box(s, 0.5, y, 12.3, 1.05, { round: 0.08, fill: C.paper, line: p.color, lw: 1.5 });
    s.addShape("rect", { x: 0.5, y: y, w: 1.4, h: 1.05, fill: { color: p.color }, line: { color: p.color } });
    s.addText(p.tag, { x: 0.5, y: y, w: 1.4, h: 1.05,
      fontSize: 18, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(p.title, { x: 2.0, y: y + 0.05, w: 8.5, h: 0.3,
      fontSize: 12, fontFace: FONT, bold: true, color: p.color });
    s.addText(p.cost, { x: 10.5, y: y + 0.05, w: 2.2, h: 0.3,
      fontSize: 9.5, fontFace: FONT, italic: true, color: C.inkSoft, align: "right" });
    s.addText(p.desc, { x: 2.0, y: y + 0.36, w: 10.7, h: 0.65,
      fontSize: 9.5, fontFace: FONT, color: C.ink });
    y += 1.15;
  });

  footer(s); pageNum(s, 44);
}

// ========================================================================
// SLIDE 45 — Conclusion + Q&A
// ========================================================================
{
  const s = pres.addSlide();
  bg(s, C.bgDark);
  s.addShape("rect", { x: 0, y: 0, w: W, h: 0.18,
    fill: { color: C.accent }, line: { color: C.accent } });

  s.addText("결론", {
    x: 0.5, y: 0.8, w: W - 1, h: 1.0,
    fontSize: 44, fontFace: FONT, bold: true, color: "FFFFFF", align: "center" });

  s.addShape("rect", { x: W/2 - 0.6, y: 1.95, w: 1.2, h: 0.04,
    fill: { color: C.accent }, line: { color: C.accent } });

  // 4 conclusions in cards
  const concs = [
    { tag: "1", t: "Architecture-level reframe",
      d: "행동 PMR ceiling 은 인코더 표현력 단독으로 결정 안 됨.  5-fold 시그니처가 동일 architectural property 의 redundant manifestation.  CLIP 2-point 비교 (LLaVA-1.5 vs Next) 가 가장 깨끗한 disconfirmer." },
    { tag: "2", t: "Causal localization (LM + Encoder)",
      d: "M5a steering: 3 of 4 모델 10/10 flip (Qwen L10, LLaVA-Next L20-25, Idefics2 L25).  M5b SIP+MLP knockout: L9 MLP 가 sufficient + necessary.  M5b SAE: 비-CLIP 3/3 break, CLIP family encoder NULL on ball.  → encoder ↔ LM dissociation map." },
    { tag: "3", t: "Pixel-encodability + post-proj ladder",
      d: "§4.6 5-model: Qwen broad / LLaVA-Next L20+L25 / LLaVA-1.5 L25 only / Idefics2 0/9 (perceiver candidate) / InternVL3 saturated.  M5b round 2: regime-cross capacity ladder + 3 NULL phenomena 분해." },
    { tag: "4", t: "Pillar B 진행 + 다음 단계",
      d: "M-LMSwap Variant A 학습 완료 (step21000 cell-discrimination 학습), Variant B 대기.  M-PSwap NaN 으로 backlogged.  B4 Pixtral + B1 multi-cell + M-Marr / M7 paper-blocker 단계." },
  ];

  let y = 2.4;
  concs.forEach((c) => {
    s.addShape("roundRect", { x: 0.6, y: y, w: W - 1.2, h: 0.95, rectRadius: 0.08,
      fill: { color: "152848" }, line: { color: C.accent, width: 1 } });
    s.addShape("rect", { x: 0.6, y: y, w: 0.6, h: 0.95,
      fill: { color: C.accent }, line: { color: C.accent } });
    s.addText(c.tag, { x: 0.6, y: y, w: 0.6, h: 0.95,
      fontSize: 22, fontFace: FONT, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(c.t, { x: 1.4, y: y + 0.05, w: W - 2.0, h: 0.3,
      fontSize: 13, fontFace: FONT, bold: true, color: C.accent });
    s.addText(c.d, { x: 1.4, y: y + 0.35, w: W - 2.0, h: 0.6,
      fontSize: 10, fontFace: FONT, color: "DEE6F2" });
    y += 1.05;
  });

  s.addText("Q & A · 토론 · 다음 우선순위", {
    x: 0.5, y: y + 0.1, w: W - 1, h: 0.4,
    fontSize: 16, fontFace: FONT, italic: true, color: "B7C2D6", align: "center" });

  s.addText("동반 자료:  slide_notes_detailed_review_ko.md  ·  references/roadmap.md  ·  docs/insights/*.md  ·  docs/CHANGELOG.md", {
    x: 0.5, y: H - 0.5, w: W - 1, h: 0.3,
    fontSize: 10, fontFace: FONT, italic: true, color: "8FA0BD", align: "center" });

  pageNum(s, 45, true);
}

// ========================================================================
// Save
// ========================================================================
pres.writeFile({ fileName: OUT })
  .then((f) => console.log("Saved:", f))
  .catch((e) => { console.error(e); process.exit(1); });







