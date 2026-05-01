// Storyline-driven Korean presentation for Physical-Mode Activation in VLMs
// Audience: non-domain. Heavy use of figures + tables. Big visuals.
// Run: NODE_PATH=$(npm root -g) node docs/review_ppt/build_storyline.js

const PPTX = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..", ".."); // project root
const FIG = path.join(ROOT, "docs", "figures");
const STIM_M2 = path.join(ROOT, "inputs", "mvp_full_20260424-093926_e9d79da3", "images");
const STIM_M8A = path.join(ROOT, "inputs", "m8a_qwen_20260425-091713_8af4836f", "images");
const STIM_M8C = path.join(ROOT, "inputs", "m8c_photos_20260425-162031", "images");
const OUT = path.join(__dirname, "physical_mode_full_review_ko.pptx");

const fig = (name) => path.join(FIG, name);
const stim_m2 = (name) => path.join(STIM_M2, name);
const stim_m8a = (name) => path.join(STIM_M8A, name);
const stim_m8c = (name) => path.join(STIM_M8C, name);

function assertFiles(paths) {
  for (const p of paths) {
    if (!fs.existsSync(p)) {
      console.error("Missing asset:", p);
      process.exit(1);
    }
  }
}

// Color palette: Ocean Gradient (deep blue dominant)
const C = {
  bgDark: "0E1E3D",       // title-slide dark navy
  bgLight: "F4F6FB",      // light off-blue background
  primary: "0B3D6B",      // deep blue (titles)
  primaryLight: "1F5F95",
  accent: "F4A23C",       // warm orange accent
  accentSoft: "FFE9CC",
  teal: "1C7293",
  ink: "1A2233",          // dark text
  inkSoft: "4A5568",      // muted text
  paper: "FFFFFF",
  cardCircle: "E8EFF7",
  cardBall: "FFF0DE",
  divider: "C9D4E5",
  good: "2C7A4F",
  bad: "B5371C",
  warm: "C4621D",
};

const FONT_HEAD = "Calibri";
const FONT_BODY = "Calibri";

const pres = new PPTX();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "원이 공으로 보이는 순간 — VLM의 추상→물리 shortcut 분석";
pres.author = "thyun.park";

const W = 13.333;
const H = 7.5;

// helpers ----------------------------------------------------------------

function addBackground(slide, color) {
  slide.background = { color: color };
}

function addPageNumber(slide, n, total, dark = false) {
  slide.addText(`${n} / ${total}`, {
    x: W - 1.2, y: H - 0.45, w: 1.0, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY,
    color: dark ? "B7C2D6" : C.inkSoft, align: "right",
  });
}

function addFooter(slide, dark = false) {
  slide.addText("원 → 공: VLM의 추상-물리 shortcut을 행동·메커니즘·픽셀 수준에서 분석", {
    x: 0.5, y: H - 0.45, w: 11, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY,
    color: dark ? "8FA0BD" : C.inkSoft, italic: true,
  });
}

function sectionStripe(slide, label) {
  // small accent bar with section label, top-left
  slide.addShape("rect", {
    x: 0.5, y: 0.4, w: 0.16, h: 0.45,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  slide.addText(label, {
    x: 0.75, y: 0.38, w: 5, h: 0.5,
    fontSize: 11, fontFace: FONT_BODY,
    bold: true, color: C.primary,
    charSpacing: 4,
  });
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.5, y: 0.85, w: 12.3, h: 0.85,
    fontSize: 28, fontFace: FONT_HEAD,
    bold: true, color: C.primary,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 1.65, w: 12.3, h: 0.45,
      fontSize: 14, fontFace: FONT_BODY,
      color: C.teal, italic: true,
    });
  }
  // thin underline accent
  slide.addShape("rect", {
    x: 0.5, y: 2.15, w: 1.0, h: 0.04,
    fill: { color: C.accent }, line: { color: C.accent },
  });
}

function addContentSlide(builder) {
  const slide = pres.addSlide();
  addBackground(slide, C.bgLight);
  return slide;
}

// ========================================================================
// SLIDE 1 — Title
// ========================================================================
{
  const s = pres.addSlide();
  addBackground(s, C.bgDark);

  // accent block top
  s.addShape("rect", {
    x: 0, y: 0, w: W, h: 0.18,
    fill: { color: C.accent }, line: { color: C.accent },
  });

  // big circle (abstract) and ball (physical) — visual hook
  s.addShape("ellipse", {
    x: 1.0, y: 4.2, w: 1.6, h: 1.6,
    fill: { type: "solid", color: "FFFFFF" },
    line: { color: "FFFFFF", width: 4 },
  });
  s.addText("원", {
    x: 1.0, y: 4.55, w: 1.6, h: 0.9,
    fontSize: 32, fontFace: FONT_HEAD, bold: true,
    color: C.bgDark, align: "center",
  });
  s.addText("→", {
    x: 2.7, y: 4.55, w: 1.0, h: 0.9,
    fontSize: 44, fontFace: FONT_HEAD, bold: true,
    color: C.accent, align: "center",
  });
  // ball visual: outer ring + shaded look
  s.addShape("ellipse", {
    x: 3.8, y: 4.2, w: 1.6, h: 1.6,
    fill: { type: "solid", color: C.accent },
    line: { color: "FFFFFF", width: 4 },
  });
  s.addShape("ellipse", {
    x: 3.95, y: 4.32, w: 0.55, h: 0.4,
    fill: { type: "solid", color: "FFE9CC" },
    line: { color: "FFE9CC" },
  });
  s.addText("공", {
    x: 3.8, y: 4.55, w: 1.6, h: 0.9,
    fontSize: 32, fontFace: FONT_HEAD, bold: true,
    color: "FFFFFF", align: "center",
  });

  s.addText("원이 공으로 보이는 순간", {
    x: 6.0, y: 1.4, w: 7.0, h: 1.2,
    fontSize: 44, fontFace: FONT_HEAD, bold: true,
    color: "FFFFFF",
  });
  s.addText("VLM이 추상 도형을 물리 객체로 \"착각\"하는 현상을\n행동·메커니즘·픽셀 수준에서 분석한다", {
    x: 6.0, y: 2.6, w: 7.0, h: 1.4,
    fontSize: 18, fontFace: FONT_BODY,
    color: "DEE6F2", italic: true,
  });

  s.addShape("rect", {
    x: 6.0, y: 4.05, w: 0.6, h: 0.04,
    fill: { color: C.accent }, line: { color: C.accent },
  });

  s.addText("5개 오픈소스 비전-언어 모델 비교 연구", {
    x: 6.0, y: 4.2, w: 7.0, h: 0.4,
    fontSize: 16, fontFace: FONT_BODY, bold: true,
    color: C.accent,
  });
  s.addText("Qwen2.5-VL · LLaVA-1.5 · LLaVA-Next · Idefics2 · InternVL3", {
    x: 6.0, y: 4.6, w: 7.0, h: 0.4,
    fontSize: 14, fontFace: FONT_BODY,
    color: "B7C2D6",
  });

  s.addText("프로젝트 종합 검토 · 2026년 5월 1일 (M0–M9 · §4 series · Pillar A · M-LMSwap · post-proj round 2)", {
    x: 6.0, y: 6.4, w: 7.0, h: 0.4,
    fontSize: 12, fontFace: FONT_BODY,
    color: "8FA0BD", italic: true,
  });
}

// ========================================================================
// SLIDE 2 — The Hook: an everyday surprise
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "S T O R Y   1 — 놀라운 관찰");
  addTitle(s, "사람이 보면 \"그냥 동그라미\"", "그런데 AI는 \"공이 떨어진다\"라고 답한다.");

  // left: image
  s.addImage({ path: fig("01_line_blank_none.png"), x: 0.6, y: 2.6, w: 3.4, h: 3.4 });
  s.addText("Figure 1. 본 연구의 baseline 자극\n(line · blank · 단서 없음)", {
    x: 0.4, y: 6.0, w: 3.8, h: 0.6,
    fontSize: 11, fontFace: FONT_BODY,
    color: C.inkSoft, italic: true, align: "center",
  });

  // right: comparison cards
  // human card
  s.addShape("roundRect", {
    x: 4.5, y: 2.6, w: 4.0, h: 1.7, rectRadius: 0.08,
    fill: { color: C.cardCircle }, line: { color: C.divider, width: 1 },
  });
  s.addText("사람의 응답", {
    x: 4.7, y: 2.7, w: 3.8, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addText("\"흰 배경 위 검은 원이 그려져 있다.\"\n→ 추상적 도형. 운동 단서 없음.", {
    x: 4.7, y: 3.15, w: 3.8, h: 1.1,
    fontSize: 13, fontFace: FONT_BODY, color: C.ink,
  });

  // VLM card
  s.addShape("roundRect", {
    x: 8.7, y: 2.6, w: 4.2, h: 1.7, rectRadius: 0.08,
    fill: { color: C.cardBall }, line: { color: C.warm, width: 1 },
  });
  s.addText("Qwen2.5-VL의 응답", {
    x: 8.9, y: 2.7, w: 4.0, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText("\"The ball will fall down due to gravity.\"\n→ 갑자기 \"공\"·\"중력\"·\"낙하\".", {
    x: 8.9, y: 3.15, w: 4.0, h: 1.1,
    fontSize: 13, fontFace: FONT_BODY, color: C.ink,
  });

  // takeaway box (extended height to fit 3 bullet lines)
  s.addShape("roundRect", {
    x: 4.5, y: 4.45, w: 8.4, h: 2.0, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.accent, width: 2 },
  });
  s.addText("관찰 — 무엇이 이상한가?", {
    x: 4.7, y: 4.5, w: 8.0, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.accent,
  });
  s.addText([
    { text: "• 이미지에 ", options: { fontSize: 12 } },
    { text: "중력 단서·지면·텍스처가 전혀 없다", options: { fontSize: 12, bold: true, color: C.bad } },
    { text: ".\n• 모델은 추상↔물리 두 해석 중 ", options: { fontSize: 12 } },
    { text: "한쪽으로만 무너진다(collapse)", options: { fontSize: 12, bold: true, color: C.bad } },
    { text: ".\n• 이런 ", options: { fontSize: 12 } },
    { text: "shortcut", options: { fontSize: 12, bold: true } },
    { text: "은 알려져 있지만, ", options: { fontSize: 12 } },
    { text: "어디서·왜 일어나는지는 미해결", options: { fontSize: 12, italic: true } },
    { text: ".", options: { fontSize: 12 } },
  ], {
    x: 4.7, y: 4.9, w: 8.0, h: 1.5,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 4,
  });

  s.addText("프롬프트:  \"What do you see in this image, and what might happen next?\"", {
    x: 0.5, y: 6.6, w: 12.5, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, color: C.inkSoft, italic: true,
  });

  addFooter(s);
  addPageNumber(s, 2, 30);
}

// ========================================================================
// SLIDE 3 — The Question
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "S T O R Y   2 — 우리의 질문");
  addTitle(s, "원이 공으로 보이는 순간을 어떻게 잡을까?",
    "한 문장: AI는 어떤 시각 단서가 주어졌을 때 추상 도형을 물리 객체로 재해석할까?");

  // 3-axis question cards
  const axes = [
    {
      tag: "WHEN",
      ko: "언제 (행동)",
      desc: "어떤 자극 / 어떤 모델에서 shortcut 이 얼마나 강한가? 행동 시그니처는?",
      color: C.primary,
    },
    {
      tag: "WHERE",
      ko: "어디서 (메커니즘)",
      desc: "모델 내부의 어떤 레이어 / 어떤 방향이 \"물리 모드\" 결정을 인과적으로 일으키는가?",
      color: C.teal,
    },
    {
      tag: "HOW",
      ko: "어떻게 (픽셀)",
      desc: "shortcut 이 픽셀에 인코드 가능한가? 이미지에 작은 노이즈만으로 응답을 뒤집을 수 있나?",
      color: C.warm,
    },
  ];

  axes.forEach((a, i) => {
    const x0 = 0.5 + i * 4.3;
    s.addShape("roundRect", {
      x: x0, y: 2.7, w: 4.0, h: 3.6, rectRadius: 0.1,
      fill: { color: C.paper }, line: { color: a.color, width: 2 },
    });
    s.addShape("rect", {
      x: x0, y: 2.7, w: 4.0, h: 0.5,
      fill: { color: a.color }, line: { color: a.color },
    });
    s.addText(a.tag, {
      x: x0 + 0.2, y: 2.7, w: 3.6, h: 0.5,
      fontSize: 12, fontFace: FONT_HEAD, bold: true,
      color: "FFFFFF", charSpacing: 6,
    });
    s.addText(a.ko, {
      x: x0 + 0.3, y: 3.4, w: 3.4, h: 0.6,
      fontSize: 22, fontFace: FONT_HEAD, bold: true,
      color: a.color,
    });
    s.addText(a.desc, {
      x: x0 + 0.3, y: 4.1, w: 3.4, h: 1.9,
      fontSize: 14, fontFace: FONT_BODY,
      color: C.ink,
    });
  });

  // bottom synthesis
  s.addShape("roundRect", {
    x: 0.5, y: 6.5, w: 12.3, h: 0.6, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("세 질문에 동시에 답하기 위해 동일한 자극·동일한 5개 모델로 행동·메커니즘·픽셀 실험을 일관되게 수행했다.", {
    x: 0.7, y: 6.5, w: 12.0, h: 0.6,
    fontSize: 13, fontFace: FONT_BODY, bold: true, color: C.warm,
  });

  addFooter(s);
  addPageNumber(s, 3, 30);
}

// ========================================================================
// SLIDE 4 — Why does this matter?
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "S T O R Y   3 — 왜 이게 중요한가");
  addTitle(s, "왜 \"shortcut\"이 중요한가?", "AI가 \"보이지 않는 것\"을 \"보았다\"고 답하면, 우리는 그 답을 신뢰할 수 없다.");

  const items = [
    {
      icon: "①",
      title: "신뢰성 / AI 안전성",
      body: "VLM이 보고 있다고 주장하는 시각 정보가 실제로는 \"이미지 내용\"이 아니라 \"학습 데이터의 priors\" 라면, 의료·자율주행·로봇 같은 안전 영역에서 위험한 환각.",
    },
    {
      icon: "②",
      title: "World-model 가설",
      body: "최근 V-JEPA·RT-2·OpenVLA 등은 VLM이 내부에 \"세상의 물리 모델\"을 갖는다고 본다. 그렇다면 그 \"물리 모드\"가 언제 켜지는가? — 본 연구의 직접적인 답.",
    },
    {
      icon: "③",
      title: "해석 가능성 (interpretability)",
      body: "행동만 측정하면 \"어디가 문제인지\" 모른다. 우리는 인코더·LM 레이어·픽셀 공간까지 들어가서 \"shortcut의 물리적 위치\"를 좁혀 들어간다.",
    },
  ];

  items.forEach((it, i) => {
    const y0 = 2.7 + i * 1.4;
    s.addShape("roundRect", {
      x: 0.5, y: y0, w: 1.0, h: 1.1, rectRadius: 0.1,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText(it.icon, {
      x: 0.5, y: y0 + 0.1, w: 1.0, h: 0.9,
      fontSize: 38, fontFace: FONT_HEAD, bold: true,
      color: "FFFFFF", align: "center",
    });
    s.addText(it.title, {
      x: 1.7, y: y0, w: 11.0, h: 0.4,
      fontSize: 18, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    s.addText(it.body, {
      x: 1.7, y: y0 + 0.4, w: 11.0, h: 0.8,
      fontSize: 13, fontFace: FONT_BODY, color: C.ink,
    });
  });

  addFooter(s);
  addPageNumber(s, 4, 30);
}

// ========================================================================
// SLIDE 5 — Background: how a VLM works (simple block)
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "B A C K G R O U N D — VLM 한 장 도식");
  addTitle(s, "VLM은 어떻게 작동하나? (3분 요약)", "그림→토큰 → 언어모델 → 텍스트. 본 연구는 이 파이프라인의 각 단계를 분리해서 본다.");

  // pipeline boxes: image -> vision encoder -> projector -> LM -> text
  const boxes = [
    { label: "이미지", sub: "(픽셀)", color: C.teal },
    { label: "Vision\nEncoder", sub: "CLIP / SigLIP / InternViT", color: C.primary },
    { label: "Projector", sub: "MLP / perceiver-resampler", color: C.primaryLight },
    { label: "Language\nModel", sub: "Qwen / Vicuna / Mistral / InternLM", color: C.primary },
    { label: "응답", sub: "(텍스트)", color: C.warm },
  ];

  const bxW = 2.1, bxH = 1.5;
  const totalW = bxW * boxes.length + 0.45 * (boxes.length - 1);
  const startX = (W - totalW) / 2;
  const yMid = 3.3;
  boxes.forEach((b, i) => {
    const x = startX + i * (bxW + 0.45);
    s.addShape("roundRect", {
      x, y: yMid, w: bxW, h: bxH, rectRadius: 0.1,
      fill: { color: b.color }, line: { color: b.color },
    });
    s.addText(b.label, {
      x: x, y: yMid + 0.18, w: bxW, h: 0.7,
      fontSize: 16, fontFace: FONT_HEAD, bold: true,
      color: "FFFFFF", align: "center",
    });
    s.addText(b.sub, {
      x: x, y: yMid + 0.85, w: bxW, h: 0.55,
      fontSize: 10, fontFace: FONT_BODY,
      color: "DEE6F2", align: "center", italic: true,
    });
    if (i < boxes.length - 1) {
      // arrow — use a thick text glyph, robust across renderers
      const ax = x + bxW;
      s.addText("▶", {
        x: ax + 0.0, y: yMid + 0.5, w: 0.45, h: 0.5,
        fontSize: 26, fontFace: FONT_HEAD, bold: true,
        color: C.accent, align: "center",
      });
    }
  });

  // annotation under each
  const notes = [
    "원본 RGB",
    "패치 단위 표현",
    "LM 어휘 공간으로 변환",
    "주의(attention) + 추론",
    "물리 동사? 추상?",
  ];
  notes.forEach((n, i) => {
    const x = startX + i * (bxW + 0.45);
    s.addText(n, {
      x: x, y: yMid + bxH + 0.15, w: bxW, h: 0.3,
      fontSize: 10, fontFace: FONT_BODY, italic: true,
      color: C.inkSoft, align: "center",
    });
  });

  // bottom caption — what we measure where
  s.addShape("roundRect", {
    x: 0.5, y: 5.7, w: 12.3, h: 1.3, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.divider, width: 1 },
  });
  s.addText("본 연구의 측정 위치", {
    x: 0.7, y: 5.75, w: 4, h: 0.35,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addText([
    { text: "Vision Encoder ", options: { bold: true, color: C.primary } },
    { text: "→ probe(AUC), SAE feature 학습  |  " },
    { text: "LM 레이어 ", options: { bold: true, color: C.primary } },
    { text: "→ logit lens, VTI steering(±α·v_L), SIP patching  |  " },
    { text: "픽셀 ", options: { bold: true, color: C.warm } },
    { text: "→ gradient ascent counterfactual  |  " },
    { text: "응답 ", options: { bold: true, color: C.warm } },
    { text: "→ PMR / GAR / RC" },
  ], {
    x: 0.7, y: 6.1, w: 12.0, h: 0.85,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 5, 30);
}

// ========================================================================
// SLIDE 6 — Background: what existed before
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "B A C K G R O U N D — 선행 연구 한눈에");
  addTitle(s, "이미 알려진 것 vs 우리가 채우는 것",
    "VLM에 대한 \"부분적인 단서\"는 있다. 우리는 그 단서들을 한 자극·한 메트릭 위에 올린다.");

  // 2x3 grid: prior work cards
  const prior = [
    { t: "Eyes Wide Shut\n(Tong et al., 2024)", b: "VLM이 놓치는 시각적 primitive를 모음. 행동만 측정." },
    { t: "Pixels-to-Principles\n(Ballout et al., 2025)", b: "vision encoder는 물리 단서를 갖지만, LM이 사용 안 함이라 보고. 사진만 사용." },
    { t: "VLMs are Blind\n(Rahmanzadehgervi et al., 2024)", b: "추상 도형 7개 task에서 VLM 58% — \"인코더는 알고 디코더는 모른다\"." },
    { t: "Shape vs Texture bias\n(Gavrikov et al., 2024)", b: "프롬프트로 모양/질감 편향 조정. 하지만 사람 96% 도달 못함." },
    { t: "MechBench\n(Zhang et al., 2024)", b: "기계적 추론 벤치마크. 모델 크기를 키워도 안 풀림 (architectural limit)." },
    { t: "VTI / Activation Patching\n(Liu+ '25 / Wang+ '23)", b: "LM 안에서 인과 개입 도구. VLM에는 거의 적용 안 됨." },
  ];
  prior.forEach((p, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 4.3, y = 2.7 + row * 1.65;
    s.addShape("roundRect", {
      x, y, w: 4.05, h: 1.45, rectRadius: 0.08,
      fill: { color: C.paper }, line: { color: C.divider, width: 1 },
    });
    s.addShape("rect", {
      x, y, w: 0.1, h: 1.45,
      fill: { color: C.teal }, line: { color: C.teal },
    });
    s.addText(p.t, {
      x: x + 0.2, y: y + 0.1, w: 3.8, h: 0.55,
      fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    s.addText(p.b, {
      x: x + 0.2, y: y + 0.65, w: 3.8, h: 0.75,
      fontSize: 11, fontFace: FONT_BODY, color: C.ink,
    });
  });

  s.addShape("roundRect", {
    x: 0.5, y: 6.05, w: 12.3, h: 0.95, rectRadius: 0.08,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1.5 },
  });
  s.addText("우리가 채우는 빈 공간", {
    x: 0.7, y: 6.1, w: 4, h: 0.35,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText("동일 자극·동일 5개 모델 위에서 (행동 PMR) + (LM·encoder 메커니즘 개입) + (픽셀 인코드 가능성) 을 한꺼번에 측정한 사례는 없음. → \"무엇이 어디서 일어나는가\"에 일관된 답.", {
    x: 0.7, y: 6.45, w: 12.0, h: 0.55,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 6, 30);
}

// ========================================================================
// SLIDE 7 — Stimuli: the abstraction ladder
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "M E T H O D — 자극 설계 (1)");
  addTitle(s, "추상화 사다리 — \"같은 동그라미, 다른 옷\"",
    "왼쪽일수록 추상(기하), 오른쪽일수록 물리(공). 본 연구의 핵심 축.");

  const ladderImgs = [
    { f: stim_m2("line_blank_none_fall_000.png"), label: "line\n(선만)" },
    { f: stim_m2("filled_blank_none_fall_000.png"), label: "filled\n(채움)" },
    { f: stim_m2("shaded_blank_none_fall_000.png"), label: "shaded\n(3D 셰이딩)" },
    { f: stim_m2("textured_blank_none_fall_000.png"), label: "textured\n(질감)" },
  ];
  const cellW = 2.7, cellH = 2.7;
  const totalW = cellW * 4 + 0.4 * 3;
  const startX = (W - totalW) / 2;
  const y0 = 2.7;
  ladderImgs.forEach((it, i) => {
    const x = startX + i * (cellW + 0.4);
    s.addShape("roundRect", {
      x, y: y0, w: cellW, h: cellH + 0.7, rectRadius: 0.06,
      fill: { color: C.paper }, line: { color: C.divider, width: 1 },
    });
    if (fs.existsSync(it.f)) {
      s.addImage({ path: it.f, x: x + 0.15, y: y0 + 0.15, w: cellW - 0.3, h: cellH - 0.3 });
    }
    s.addText(it.label, {
      x: x, y: y0 + cellH + 0.0, w: cellW, h: 0.55,
      fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.primary, align: "center",
    });
  });

  // arrow underneath
  s.addShape("rect", {
    x: startX, y: 6.05, w: totalW, h: 0.05,
    fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("← 추상 (geometry)", {
    x: startX, y: 6.15, w: 4.5, h: 0.35,
    fontSize: 12, fontFace: FONT_BODY, italic: true, color: C.primary,
  });
  s.addText("물리 (object) →", {
    x: startX + totalW - 4.5, y: 6.15, w: 4.5, h: 0.35,
    fontSize: 12, fontFace: FONT_BODY, italic: true, color: C.warm, align: "right",
  });

  // bottom caption
  s.addText("\"같은 원\"이 인지에 어떤 옷을 입었느냐에 따라 모델이 다르게 반응하는지 — 그 \"전환점\"을 찾는 게 H1 가설.", {
    x: 0.5, y: 6.65, w: 12.3, h: 0.4,
    fontSize: 12, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  addFooter(s);
  addPageNumber(s, 7, 30);
}

// ========================================================================
// SLIDE 8 — Stimuli: factorial axes
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "M E T H O D — 자극 설계 (2)");
  addTitle(s, "물리 단서를 한 축씩 더해 본다",
    "추상화 × 배경 × 부가 단서 — 5축 factorial 디자인 (총 2,880 자극)");

  // left: factor table
  const tbl = [
    [{ text: "축", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
     { text: "값", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
     { text: "의미 (사람이 보는 단서)", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } }],
    [{ text: "object_level" }, { text: "line / filled / shaded / textured" }, { text: "도형 → 3D 공으로 가는 사다리" }],
    [{ text: "bg_level" }, { text: "blank / ground / scene" }, { text: "지면이 있나? 풍경이 있나?" }],
    [{ text: "cue_level" }, { text: "none / cast_shadow / motion_arrow / both" }, { text: "그림자·화살표 (motion 단서)" }],
    [{ text: "event" }, { text: "fall / horizontal / rise" }, { text: "프레임 위치 (낙하/수평/상승)" }],
    [{ text: "label (prompt)" }, { text: "circle / ball / planet / _nolabel" }, { text: "언어 라벨이 응답을 흔드는지" }],
  ];
  s.addTable(tbl, {
    x: 0.5, y: 2.7, w: 7.6, h: 2.8,
    colW: [1.5, 2.5, 3.6],
    fontSize: 10.5, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink,
  });

  // right: example images grid 2x3
  const ex = [
    { f: fig("01_line_blank_none.png"), c: "line · blank · none" },
    { f: fig("02_line_ground_none.png"), c: "line · ground · none" },
    { f: fig("03_shaded_ground_none.png"), c: "shaded · ground · none" },
    { f: fig("04_textured_ground_arrow_shadow.png"), c: "textured · ground · 화살표+그림자" },
  ];
  const cw = 2.1, ch = 2.1;
  ex.forEach((e, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 8.4 + col * (cw + 0.2);
    const y = 2.7 + row * (ch + 0.55);
    s.addImage({ path: e.f, x, y, w: cw, h: ch });
    s.addText(e.c, {
      x, y: y + ch, w: cw, h: 0.45,
      fontSize: 9, fontFace: FONT_BODY, color: C.inkSoft, align: "center", italic: true,
    });
  });

  // bottom: stim totals
  s.addShape("roundRect", {
    x: 0.5, y: 5.7, w: 7.6, h: 1.3, rectRadius: 0.06,
    fill: { color: C.bgLight }, line: { color: C.divider, width: 1 },
  });
  s.addText("M2 (메인 자극): 4 × 3 × 4 × 3 × 10 seed = 1,440  ×  3 라벨 + label-free  =  2,880 추론 / 모델", {
    x: 0.7, y: 5.78, w: 7.2, h: 0.5,
    fontSize: 12, fontFace: FONT_BODY, bold: true, color: C.primary,
  });
  s.addText("외부 타당성 확장: M8a (5 도형) · M8d (3 카테고리: 차/사람/새) · M8c (60 실사진)", {
    x: 0.7, y: 6.25, w: 7.2, h: 0.45,
    fontSize: 11, fontFace: FONT_BODY, color: C.ink,
  });
  s.addText("핵심: \"같은 사건(낙하)을 다양한 추상화/단서 옷으로 입혀서, 모델이 어떤 옷일 때 무너지는지를 본다.\"", {
    x: 0.7, y: 6.65, w: 7.2, h: 0.35,
    fontSize: 10.5, fontFace: FONT_BODY, italic: true, color: C.inkSoft,
  });

  addFooter(s);
  addPageNumber(s, 8, 30);
}

// ========================================================================
// SLIDE 9 — Models (the 5 VLMs)
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "M E T H O D — 테스트 모델");
  addTitle(s, "5개 오픈소스 VLM",
    "인코더 계열 × LM 계열을 골고루 섞어, 어떤 부품이 결정적인지 비교 가능하게 디자인.");

  const tbl = [
    [
      { text: "모델", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "Vision Encoder", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "Projector", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "Language Model", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "이미지 처리", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "역할", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [
      { text: "Qwen2.5-VL-7B" }, { text: "SigLIP" }, { text: "MLP" },
      { text: "Qwen2-7B" }, { text: "동적 504×504" },
      { text: "메인 — saturated 모델, 인과 실험의 anchor", options: { color: C.warm, bold: true } },
    ],
    [
      { text: "LLaVA-1.5-7B" }, { text: "CLIP-ViT-L/14" }, { text: "MLP" },
      { text: "Vicuna-7B (LLaMA-2)" }, { text: "고정 336×336" },
      { text: "Floor — unsaturated 모델, S-curve 가장 깨끗" },
    ],
    [
      { text: "LLaVA-Next-7B" }, { text: "CLIP-ViT-L/14" }, { text: "MLP" },
      { text: "Mistral-7B" }, { text: "AnyRes 5-tile" },
      { text: "Mid — 같은 인코더 다른 LM/처리, 결정자 분리에 핵심", options: { color: C.warm, bold: true } },
    ],
    [
      { text: "Idefics2-8B" }, { text: "SigLIP-SO400M" },
      { text: "perceiver-resampler", options: { color: C.bad, bold: true } },
      { text: "Mistral-7B" }, { text: "384×384" },
      { text: "유일한 perceiver — 픽셀 shortcut 차단의 단서", options: { color: C.warm, bold: true } },
    ],
    [
      { text: "InternVL3-8B" }, { text: "InternViT-300M" }, { text: "MLP (pixel-shuffle)" },
      { text: "InternLM3-8B" }, { text: "동적 448×448" },
      { text: "Saturated — 비-CLIP 인코더 비교점" },
    ],
  ];
  s.addTable(tbl, {
    x: 0.5, y: 2.7, w: 12.3, h: 3.8,
    colW: [1.8, 2.0, 2.0, 1.8, 1.7, 3.0],
    fontSize: 11, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink,
    valign: "middle",
  });

  s.addShape("roundRect", {
    x: 0.5, y: 6.55, w: 12.3, h: 0.55, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("핵심 비교쌍 — LLaVA-1.5 vs LLaVA-Next: 동일 CLIP-ViT-L 인코더, 다른 LM·처리 → \"인코더만이 결정자가 아님\"의 가장 깨끗한 disconfirmer.", {
    x: 0.7, y: 6.55, w: 12.0, h: 0.55,
    fontSize: 11, fontFace: FONT_BODY, bold: true, color: C.warm,
  });

  addFooter(s);
  addPageNumber(s, 9, 30);
}

// ========================================================================
// SLIDE 10 — Metrics
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "M E T H O D — 측정 지표");
  addTitle(s, "무엇을 \"shortcut의 강도\"로 셀까?",
    "응답 텍스트 위에서 자동 채점 + bootstrap CI. 사람 검수 ~5% 불일치.");

  const metricTbl = [
    [
      { text: "지표", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "정의 (한 줄로)", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "예시", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [
      { text: "PMR", options: { bold: true, color: C.primary } },
      { text: "응답에 \"falls / rolls / bounces\" 같은 물리 동사가 들어간 비율" },
      { text: "Qwen2.5-VL: 0.94 / LLaVA-1.5: 0.18" },
    ],
    [
      { text: "PMR(_nolabel)", options: { bold: true, color: C.primary } },
      { text: "라벨이 없는 \"open-ended\" 프롬프트일 때의 PMR — 이미지가 진짜 얼마나 물리로 보였는지" },
      { text: "Qwen 0.94 → 모델 자체 편향 측정" },
    ],
    [
      { text: "GAR", options: { bold: true, color: C.primary } },
      { text: "물리 응답 중에서 \"하방으로 떨어진다\"고 답한 비율 (중력 정합도)" },
      { text: "ball 라벨: 0.79" },
    ],
    [
      { text: "RC (consistency)", options: { bold: true, color: C.primary } },
      { text: "T=0.7 샘플링에서 N seed의 PMR call 일관도 (결정 안정성)" },
      { text: "M2 평균 0.92" },
    ],
    [
      { text: "H2 paired-Δ", options: { bold: true, color: C.warm } },
      { text: "PMR(label) − PMR(_nolabel) — \"라벨이 응답을 얼마나 끌고 가는지\"" },
      { text: "LLaVA ball: +0.475" },
    ],
    [
      { text: "v_L 방향", options: { bold: true, color: C.warm } },
      { text: "L 레이어 hidden state에서 mean(physics=1) − mean(physics=0). \"물리 모드 방향\" 추정" },
      { text: "Qwen v_L10 (3584 dim)" },
    ],
  ];
  s.addTable(metricTbl, {
    x: 0.5, y: 2.7, w: 12.3, h: 4.0,
    colW: [2.3, 6.7, 3.3],
    fontSize: 11, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink, valign: "middle",
  });

  s.addText("자동 채점 (rule-based scorer): physical verb stem (다국어) + abstract marker gating. Wilson 95% CI / 5000-iter bootstrap CI.", {
    x: 0.5, y: 6.85, w: 12.3, h: 0.35,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft,
  });

  addFooter(s);
  addPageNumber(s, 10, 30);
}

// ========================================================================
// SLIDE 11 — Findings overview map
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G S — 6개 핵심 결과 한눈에");
  addTitle(s, "발견의 흐름 — 6단계로 좁혀 들어간다",
    "행동 → 메커니즘 → 픽셀 → 외부 타당성. 각 단계가 다음 단계의 질문을 낳는다.");

  const stages = [
    { n: "1", t: "행동 사다리", b: "5 모델 PMR이 0.18 ↔ 0.99 — 같은 자극, 모델별로 천차만별." },
    { n: "2", t: "Encoder boomerang", b: "인코더는 모두 \"보고\" 있다(AUC≈1.0) — 하지만 행동은 갈림." },
    { n: "3", t: "원인은 architecture", b: "같은 CLIP인 LLaVA-1.5(0.18) vs LLaVA-Next(0.70). 인코더 단독 결정자 X." },
    { n: "4", t: "LM의 한 레이어가 결정", b: "Qwen L10에 ±α·v_L 더하면 응답이 뒤집힌다(causal)." },
    { n: "5", t: "Encoder의 ~30개 feature", b: "SAE 학습 후 top-k feature만 ablate → 3 of 5 모델 PMR 무너짐." },
    { n: "6", t: "픽셀에 인코드 가능", b: "픽셀 공간 gradient ascent로 추상 원 → 물리 응답 유도. Idefics2만 차단." },
  ];

  const cw = 2.0, ch = 2.0;
  const startX = 0.5;
  const startY = 2.7;
  stages.forEach((st, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = startX + col * (cw + 2.05);
    const y = startY + row * (ch + 0.3);
    s.addShape("roundRect", {
      x, y, w: cw + 1.95, h: ch, rectRadius: 0.08,
      fill: { color: C.paper }, line: { color: C.divider, width: 1 },
    });
    // numeric badge
    s.addShape("ellipse", {
      x: x + 0.15, y: y + 0.15, w: 0.7, h: 0.7,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText(st.n, {
      x: x + 0.15, y: y + 0.2, w: 0.7, h: 0.6,
      fontSize: 22, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(st.t, {
      x: x + 0.95, y: y + 0.15, w: cw + 0.95, h: 0.5,
      fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    s.addText(st.b, {
      x: x + 0.15, y: y + 0.95, w: cw + 1.65, h: 1.0,
      fontSize: 11, fontFace: FONT_BODY, color: C.ink,
    });
  });

  addFooter(s);
  addPageNumber(s, 11, 30);
}

// ========================================================================
// SLIDE 12 — Result 1: PMR ladder
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   1 — 행동 사다리");
  addTitle(s, "같은 자극, 다른 모델 — PMR 0.18에서 0.99까지",
    "5개 모델이 같은 480개 자극 위에서 보이는 \"물리 모드 비율\"의 격차");

  s.addImage({ path: fig("m2_cross_model_pmr_ladder.png"), x: 0.4, y: 2.4, w: 7.5, h: 4.5 });

  // right side: takeaway table
  const tbl = [
    [
      { text: "모델", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "PMR(_nolabel)", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "위치", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [{ text: "LLaVA-1.5" }, { text: "0.38 [0.34, 0.43]" }, { text: "Floor", options: { color: C.good, bold: true } }],
    [{ text: "LLaVA-Next" }, { text: "0.79 [0.75, 0.83]" }, { text: "Mid" }],
    [{ text: "Qwen2.5-VL" }, { text: "0.94 [0.92, 0.96]" }, { text: "Saturated", options: { color: C.bad, bold: true } }],
    [{ text: "Idefics2" }, { text: "0.97 [0.95, 0.98]" }, { text: "Saturated", options: { color: C.bad, bold: true } }],
    [{ text: "InternVL3" }, { text: "0.99 [0.98, 1.00]" }, { text: "Saturated", options: { color: C.bad, bold: true } }],
  ];
  s.addTable(tbl, {
    x: 8.1, y: 2.7, w: 4.8, h: 2.7,
    colW: [1.6, 2.0, 1.2],
    fontSize: 11, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink, valign: "middle",
  });

  s.addShape("roundRect", {
    x: 8.1, y: 5.6, w: 4.8, h: 1.3, rectRadius: 0.06,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("핵심", {
    x: 8.3, y: 5.65, w: 2, h: 0.3,
    fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText("같은 인코더 계열(CLIP) 안에서도 PMR이 0.38 ↔ 0.79로 갈린다.\n→ 인코더 단독 결정자 가설 disconfirm.", {
    x: 8.3, y: 6.0, w: 4.5, h: 0.85,
    fontSize: 11, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 12, 30);
}

// ========================================================================
// SLIDE 13 — Result 1b: H1 ramp & H2 paired delta
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   1b — 추상 → 물리 전환");
  addTitle(s, "추상화가 올라가면 PMR도 올라간다 — 단, 천장이 있는 모델은 빼고",
    "H1: object_level ramp / H2: 라벨 효과 — 모델별로 \"활용 가능한 헤드룸\"이 다르다.");

  s.addImage({ path: fig("m2_cross_model_h1_ramp.png"), x: 0.4, y: 2.4, w: 6.4, h: 4.2 });
  s.addImage({ path: fig("m2_cross_model_h2_paired_delta.png"), x: 6.9, y: 2.4, w: 6.0, h: 4.2 });

  s.addText("Figure 2a. H1 — abstraction → PMR (LLaVA-1.5만 깨끗한 +0.30 ramp)", {
    x: 0.4, y: 6.65, w: 6.4, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });
  s.addText("Figure 2b. H2 paired-Δ — \"라벨이 끌고 가는 양\"의 부호 패턴 3가지", {
    x: 6.9, y: 6.65, w: 6.0, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  addFooter(s);
  addPageNumber(s, 13, 30);
}

// ========================================================================
// SLIDE 14 — Result 2: Encoder boomerang
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   2 — 인코더 boomerang");
  addTitle(s, "인코더는 \"이미 봤다\" — 하지만 LM이 \"무시\" 또는 \"과대 사용\"",
    "5개 인코더가 모두 추상↔물리를 0.99 AUC로 선형 분리. 행동은 0.18 ↔ 0.99.");

  s.addImage({ path: fig("encoder_chain_5model.png"), x: 0.5, y: 2.4, w: 12.3, h: 4.0 });

  s.addShape("roundRect", {
    x: 0.5, y: 6.5, w: 12.3, h: 0.6, rectRadius: 0.05,
    fill: { color: C.bgLight }, line: { color: C.divider, width: 1 },
  });
  s.addText("해석 — \"Encoder knows, decoder gates\".  같은 자극 입력에 대해 인코더 표현은 모두 충분하다(stim-y AUC=1.0). 행동 격차는 인코더 표현력으로 설명되지 않는다.", {
    x: 0.7, y: 6.55, w: 12.0, h: 0.5,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 14, 30);
}

// ========================================================================
// SLIDE 15 — Result 3: M9 generalization audit
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   3 — 모델 × 자극 일반화");
  addTitle(s, "어디서 측정해도 같은 클러스터링 (M9 generalization audit)",
    "3 모델 × 3 자극 source × bootstrap CI. PMR 천장 / 사진 압축 / H7 효과의 일관성.");

  s.addImage({ path: fig("m9_summary.png"), x: 0.5, y: 2.4, w: 12.3, h: 4.4 });

  s.addText("(좌) 합성 자극: non-CLIP cluster [0.84, 0.92] vs CLIP-1.5 [0.14, 0.21] 완전 분리.   (우) 사진(M8c): 모든 모델 [0.28, 0.55]로 수렴 — 사진은 인코더 격차를 압축한다.", {
    x: 0.5, y: 6.85, w: 12.3, h: 0.4,
    fontSize: 11, fontFace: FONT_BODY, italic: true, color: C.inkSoft,
  });

  addFooter(s);
  addPageNumber(s, 15, 30);
}

// ========================================================================
// SLIDE 16 — Mechanism: VTI Steering at LM L10
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   4 — LM 한 레이어의 인과적 결정");
  addTitle(s, "LM L10에 +α·v_L10을 더하면, \"원\"이 \"공\"이 된다",
    "v_L10 = mean(hidden | physics) − mean(hidden | abstract).  α=40에서 10/10 응답이 뒤집힘.");

  // left: image
  s.addImage({ path: fig("01_line_blank_none.png"), x: 0.5, y: 2.6, w: 3.5, h: 3.5 });
  s.addText("입력: line · blank · none\n(원, 단서 없음)", {
    x: 0.4, y: 6.1, w: 3.7, h: 0.6,
    fontSize: 11, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  // arrow
  s.addText("→", {
    x: 4.2, y: 4.0, w: 0.8, h: 0.8,
    fontSize: 60, fontFace: FONT_HEAD, bold: true, color: C.accent, align: "center",
  });

  // right: response examples table
  const tbl = [
    [
      { text: "α", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "L10 응답 (요약)", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "PMR call", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [{ text: "0  (baseline)" }, { text: "\"This is just a circle on a white background.\"" }, { text: "추상 (D)", options: { color: C.good } }],
    [{ text: "+10" }, { text: "\"Just a black circle... no movement.\"" }, { text: "추상 (D)", options: { color: C.good } }],
    [{ text: "+20" }, { text: "\"It is a circle, drawn statically.\"" }, { text: "추상 (D)", options: { color: C.good } }],
    [{ text: "+40", options: { bold: true, color: C.warm } },
     { text: "\"It stays still — the circle appears to be floating in space without external force.\"", options: { color: C.warm, bold: true } },
     { text: "물리·정지 (B)", options: { color: C.bad, bold: true } }],
    [{ text: "−40", options: { bold: true, color: C.warm } },
     { text: "\"The circle remains stationary, suspended.\" (정적 물리)", options: { color: C.warm, bold: true } },
     { text: "물리·정지 (B)", options: { color: C.bad, bold: true } }],
  ];
  s.addTable(tbl, {
    x: 5.1, y: 2.6, w: 7.7, h: 3.0,
    colW: [1.2, 4.7, 1.8],
    fontSize: 10.5, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink, valign: "middle",
  });

  s.addShape("roundRect", {
    x: 5.1, y: 5.85, w: 7.7, h: 1.2, rectRadius: 0.06,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1.5 },
  });
  s.addText("핵심: L10에서만 일어나고(L15·L20·L25는 안 됨), |α|>임계값에서 \"regime axis\"로 작동.\n+α=동적, −α=정적 — 두 방향 모두 추상 → 물리 모드로 끌고 간다.", {
    x: 5.3, y: 5.92, w: 7.4, h: 1.05,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 16, 30);
}

// ========================================================================
// SLIDE 17 — Mechanism: M5b SAE intervention cross-model
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   5 — Encoder의 ~30개 feature");
  addTitle(s, "Encoder의 SAE feature만 꺼도 PMR이 무너진다",
    "5 모델 × actually-consumed layer. \"인코더에 국소화된 물리 단서 표현\"이 비-CLIP 클러스터에만 존재.");

  s.addImage({ path: fig("m5b_sae_intervention_cross_model.png"), x: 0.5, y: 2.4, w: 6.8, h: 4.0 });

  // right: table
  const tbl = [
    [
      { text: "Model", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "Layer", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "k=20", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "k=160", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "결과", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [{ text: "Qwen2.5-VL" }, { text: "L31" }, { text: "1.00" }, { text: "0.00" }, { text: "BREAK", options: { color: C.bad, bold: true } }],
    [{ text: "LLaVA-1.5" }, { text: "L22" }, { text: "1.00" }, { text: "1.00" }, { text: "NULL", options: { color: C.good, bold: true } }],
    [{ text: "LLaVA-Next" }, { text: "L22" }, { text: "1.00" }, { text: "1.00" }, { text: "NULL", options: { color: C.good, bold: true } }],
    [{ text: "Idefics2" }, { text: "L26" }, { text: "1.00" }, { text: "0.00" }, { text: "BREAK", options: { color: C.bad, bold: true } }],
    [{ text: "InternVL3" }, { text: "L23" }, { text: "1.00" }, { text: "0.00" }, { text: "BREAK", options: { color: C.bad, bold: true } }],
  ];
  s.addTable(tbl, {
    x: 7.5, y: 2.6, w: 5.4, h: 2.6,
    colW: [1.5, 0.8, 0.8, 0.9, 1.4],
    fontSize: 10.5, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink, valign: "middle",
  });

  s.addShape("roundRect", {
    x: 7.5, y: 5.4, w: 5.4, h: 1.6, rectRadius: 0.06,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("3 of 5 break, 2 LLaVA NULL", {
    x: 7.65, y: 5.45, w: 5.2, h: 0.35,
    fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText("Random k 컨트롤은 모두 1.00 — 방향 특이성.\n• 비-CLIP: 인코더 안에 \"물리 모드\" 표현이 국소화\n• CLIP family: 인코더에 없음 → LM-side로만 라우팅", {
    x: 7.65, y: 5.8, w: 5.2, h: 1.2,
    fontSize: 11, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 17, 30);
}

// ========================================================================
// SLIDE 18 — SIP / patching localization (sanity check)
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   5b — 부품별 필요성");
  addTitle(s, "Qwen 안에서: L9 MLP 하나가 \"물리 모드\"를 만든다",
    "n=20 SIP 페어. Attention knockout는 모두 IE=0 (redundant); MLP knockout만 L9에서 IE=+1.0.");

  s.addImage({ path: fig("m5b_knockout_per_layer_ie.png"), x: 0.4, y: 2.5, w: 7.5, h: 3.6 });

  // right: SIP per-layer
  s.addImage({ path: fig("m5b_sip_per_layer_ie.png"), x: 8.0, y: 2.5, w: 4.9, h: 2.6 });

  s.addShape("roundRect", {
    x: 8.0, y: 5.2, w: 4.9, h: 1.85, rectRadius: 0.06,
    fill: { color: C.bgLight }, line: { color: C.divider, width: 1 },
  });
  s.addText("Triangulation (Qwen)", {
    x: 8.15, y: 5.25, w: 4.6, h: 0.3,
    fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addText([
    { text: "• Encoder ", options: { bold: true } },
    { text: "약 30개 SAE feature →\n" },
    { text: "• L0–L9 ", options: { bold: true } },
    { text: "visual token이 정보 운반 →\n" },
    { text: "• L9 MLP ", options: { bold: true, color: C.warm } },
    { text: "에서 commitment 생성 →\n" },
    { text: "• L10 ", options: { bold: true, color: C.warm } },
    { text: "에서 attention이 redundantly read-out →\n" },
    { text: "• 출력 letter B/D 결정." },
  ], {
    x: 8.15, y: 5.55, w: 4.65, h: 1.5,
    fontSize: 11, fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 2,
  });

  s.addText("(좌) MLP knockout — L9에서만 IE=+1.0, 다른 곳은 0.   (우) SIP patching — L9까지 IE=1.0, L14+ 0 (수직 절벽).", {
    x: 0.4, y: 6.2, w: 7.5, h: 0.4,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft,
  });

  addFooter(s);
  addPageNumber(s, 18, 30);
}

// ========================================================================
// SLIDE 19 — Pixel encodability §4.6
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   6 — 픽셀 자체에 인코드 가능");
  addTitle(s, "이미지에 \"눈에 안 보이는 노이즈\"만 더해도 응답이 뒤집힌다",
    "v_L10 방향으로 픽셀 공간 gradient ascent (ε=0.05). 5/5 flip vs 매칭 random 0/15.");

  // left: trajectory plot
  s.addImage({ path: fig("sec4_6_counterfactual_stim_trajectory.png"), x: 0.4, y: 2.5, w: 7.0, h: 4.0 });
  s.addText("Figure 6a. v_L10 projection trajectory (Qwen). 200 step Adam, ε=0.1.", {
    x: 0.4, y: 6.55, w: 7.0, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  // right: response comparison
  s.addShape("roundRect", {
    x: 7.6, y: 2.5, w: 5.3, h: 1.7, rectRadius: 0.06,
    fill: { color: C.cardCircle }, line: { color: C.primary, width: 1 },
  });
  s.addText("Baseline (ε=0)", {
    x: 7.75, y: 2.55, w: 5.0, h: 0.35,
    fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addText("\"The circle will remain stationary as there is no indication of movement.\"", {
    x: 7.75, y: 2.95, w: 5.0, h: 1.2,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink, italic: true,
  });

  s.addShape("roundRect", {
    x: 7.6, y: 4.3, w: 5.3, h: 1.7, rectRadius: 0.06,
    fill: { color: C.cardBall }, line: { color: C.warm, width: 1.5 },
  });
  s.addText("v_L10 ascent (ε=0.05)", {
    x: 7.75, y: 4.35, w: 5.0, h: 0.35,
    fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText("\"The circle will continue to fall downward due to gravity.\" — 픽셀 차이는 거의 안 보이는데 응답이 \"공의 낙하\"로 collapse.", {
    x: 7.75, y: 4.7, w: 5.0, h: 1.3,
    fontSize: 12, fontFace: FONT_BODY, color: C.ink, italic: true,
  });

  s.addShape("roundRect", {
    x: 7.6, y: 6.1, w: 5.3, h: 0.85, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("→ shortcut은 \"runtime hidden injection\"만이 아니라 픽셀 자체에 인코드 가능. 매칭 magnitude random 0/15가 \"any perturbation\" 가설을 falsify.", {
    x: 7.75, y: 6.15, w: 5.05, h: 0.75,
    fontSize: 11, fontFace: FONT_BODY, color: C.warm, bold: true,
  });

  addFooter(s);
  addPageNumber(s, 19, 30);
}

// ========================================================================
// SLIDE 20 — Pixel cross-model + Idefics2 anomaly
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   6b — 모델별 \"픽셀 routability\"");
  addTitle(s, "픽셀 인코드 가능성은 architecture-conditional",
    "5 모델 × LM layer × n=10. Qwen 5층 모두 ≥80%. Idefics2는 9 층 모두 0/10.");

  s.addImage({ path: fig("sec4_6_cross_model_layer_sweep.png"), x: 0.5, y: 2.4, w: 9.0, h: 4.5 });

  // right callout
  s.addShape("roundRect", {
    x: 9.7, y: 2.5, w: 3.3, h: 4.4, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.warm, width: 2 },
  });
  s.addText("Idefics2 단독 패턴", {
    x: 9.85, y: 2.55, w: 3.0, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText([
    { text: "• 9개 layer (L5–L31, 16–97% 깊이) 모두 0/10 flip\n", options: { fontSize: 11 } },
    { text: "• 그러나 v_L projection은 정상 ascending\n", options: { fontSize: 11 } },
    { text: "• M4 LM probe AUC 0.995 ", options: { fontSize: 11 } },
    { text: "(정보는 LM 도달!)\n", options: { fontSize: 11, bold: true, color: C.bad } },
    { text: "• M5a forward steering 10/10 ", options: { fontSize: 11 } },
    { text: "(forward는 작동)\n\n", options: { fontSize: 11, bold: true, color: C.bad } },
    { text: "→ perceiver-resampler 가설:", options: { fontSize: 11, bold: true } },
    { text: "\nperceiver는 \"픽셀 → v_L\" 역방향 routability 만 차단.", options: { fontSize: 11, italic: true } },
  ], {
    x: 9.85, y: 2.95, w: 3.05, h: 3.9,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 2,
  });

  addFooter(s);
  addPageNumber(s, 20, 30);
}

// ========================================================================
// SLIDE 21 — External validity: M8a, M8d, M8c
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   7 — 외부 타당성");
  addTitle(s, "원 → 다른 도형 / 다른 카테고리 / 실사진까지 확인",
    "M8a (5 도형) / M8d (차/사람/새) / M8c (60 실사진). 핵심 클러스터링은 모두 보존.");

  // 3 columns
  s.addImage({ path: fig("m8a_shape_grid.png"), x: 0.5, y: 2.5, w: 4.0, h: 3.8 });
  s.addImage({ path: fig("m8d_full_scene_samples.png"), x: 4.7, y: 2.5, w: 4.0, h: 3.8 });
  s.addImage({ path: fig("m8c_photo_grid.png"), x: 8.9, y: 2.5, w: 4.0, h: 3.8 });

  s.addText("M8a — 5 도형 × 4 추상화", {
    x: 0.5, y: 6.35, w: 4.0, h: 0.3,
    fontSize: 11, fontFace: FONT_HEAD, bold: true, color: C.primary, align: "center",
  });
  s.addText("Qwen 1/4 PASS, LLaVA 4/4 — 비대칭 자체가 가설 검증", {
    x: 0.4, y: 6.65, w: 4.2, h: 0.4,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  s.addText("M8d — 차 / 사람 / 새", {
    x: 4.7, y: 6.35, w: 4.0, h: 0.3,
    fontSize: 11, fontFace: FONT_HEAD, bold: true, color: C.primary, align: "center",
  });
  s.addText("LLaVA 3/3 H7 PASS (라벨이 regime을 선택)", {
    x: 4.6, y: 6.65, w: 4.2, h: 0.4,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  s.addText("M8c — 60 실사진", {
    x: 8.9, y: 6.35, w: 4.0, h: 0.3,
    fontSize: 11, fontFace: FONT_HEAD, bold: true, color: C.primary, align: "center",
  });
  s.addText("사진은 인코더 격차를 압축, label 효과 절반으로", {
    x: 8.8, y: 6.65, w: 4.2, h: 0.4,
    fontSize: 10, fontFace: FONT_BODY, italic: true, color: C.inkSoft, align: "center",
  });

  addFooter(s);
  addPageNumber(s, 21, 30);
}

// ========================================================================
// SLIDE 22 — Bonus: scaling & multilingual
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "F I N D I N G   7b — 추가 강건성 검증");
  addTitle(s, "모델을 키워도, 한국어를 써도 — 클러스터링은 그대로",
    "§4.8 Qwen 7B vs 32B / §4.3 한국어·일본어 라벨. \"scale doesn't fix grounding.\"");

  // left: scaling table
  s.addText("§4.8 Qwen 7B vs 32B (M2 동일 자극)", {
    x: 0.5, y: 2.5, w: 6.2, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  const sc = [
    [
      { text: "지표", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "7B", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "32B", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
      { text: "Δ", options: { bold: true, fill: { color: C.primary }, color: "FFFFFF" } },
    ],
    [{ text: "Aggregate PMR" }, { text: "0.931" }, { text: "0.926" }, { text: "−0.005", options: { color: C.good } }],
    [{ text: "abstract_reject" }, { text: "0.002" }, { text: "0.065" }, { text: "35×", options: { color: C.bad, bold: true } }],
    [{ text: "H2 ball−circle" }, { text: "+0.071" }, { text: "+0.010" }, { text: "−0.061" }],
    [{ text: "cue=none PMR" }, { text: "0.797" }, { text: "0.711" }, { text: "−0.086", options: { color: C.bad } }],
  ];
  s.addTable(sc, {
    x: 0.5, y: 2.95, w: 6.2, h: 2.7,
    colW: [2.0, 1.4, 1.4, 1.4],
    fontSize: 11, fontFace: FONT_BODY,
    border: { type: "solid", pt: 1, color: C.divider },
    color: C.ink, valign: "middle",
  });
  s.addText("→ 5× 파라미터 스케일이 PMR 천장을 못 깨뜨린다 (MechBench-style). 약-cue에서만 32B가 미세하게 abstract-mode를 더 본다.", {
    x: 0.5, y: 5.75, w: 6.3, h: 0.7,
    fontSize: 11, fontFace: FONT_BODY, italic: true, color: C.warm,
  });

  // right: multilingual figure
  s.addText("§4.3 한국어 vs 영어 라벨 (5 모델)", {
    x: 7.0, y: 2.5, w: 6.0, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addImage({ path: fig("sec4_3_korean_vs_english_cross_model.png"), x: 7.0, y: 2.95, w: 6.0, h: 3.0 });
  s.addText("• 라벨 ordering 4/5 모델 보존 (planet > ball > circle).\n• LLaVA-1.5 가장 큰 swing — Vicuna 한국어 SFT 약함의 시그널.\n• Idefics2 일본어 \"惑星\"에서 24% Chinese fallback.", {
    x: 7.0, y: 6.0, w: 6.0, h: 1.0,
    fontSize: 11, fontFace: FONT_BODY, color: C.ink,
  });

  addFooter(s);
  addPageNumber(s, 22, 30);
}

// ========================================================================
// SLIDE 23 — Synthesis: 5-fold redundant signature
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "S Y N T H E S I S — 5겹 다운스트림 시그니처");
  addTitle(s, "5개 \"증거\"가 같은 architecture clustering을 가리킨다",
    "단일 architectural property가 5겹 redundant하게 표현. 단순 \"인코더 capacity\" 가설로는 설명 불가.");

  const sigs = [
    { n: "①", t: "PMR ceiling", b: "비-CLIP [0.84, 0.92] vs CLIP [0.14, 0.37] 분리" },
    { n: "②", t: "Decision-stability ceiling (RC)", b: "비-CLIP은 cue 발화 시 5 seed 모두 동일 call" },
    { n: "③", t: "픽셀 encodability", b: "Qwen broad shortcut, Idefics2 0/9 (perceiver bottleneck)" },
    { n: "④", t: "LM logit-lens probe AUC", b: "encoder probe ladder와 동일 클러스터링" },
    { n: "⑤", t: "Encoder SAE feature ablation", b: "3 of 5 break, 2 LLaVA NULL (encoder vs LM 분기)" },
  ];

  sigs.forEach((sg, i) => {
    const y0 = 2.45 + i * 0.78;
    s.addShape("roundRect", {
      x: 0.5, y: y0, w: 12.3, h: 0.7, rectRadius: 0.06,
      fill: { color: C.paper }, line: { color: C.divider, width: 1 },
    });
    s.addShape("ellipse", {
      x: 0.7, y: y0 + 0.1, w: 0.5, h: 0.5,
      fill: { color: C.accent }, line: { color: C.accent },
    });
    s.addText(sg.n, {
      x: 0.7, y: y0 + 0.13, w: 0.5, h: 0.5,
      fontSize: 16, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(sg.t, {
      x: 1.35, y: y0 + 0.06, w: 5.0, h: 0.32,
      fontSize: 12, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    s.addText(sg.b, {
      x: 1.35, y: y0 + 0.36, w: 11.0, h: 0.32,
      fontSize: 11, fontFace: FONT_BODY, color: C.ink,
    });
  });

  // synthesis takeaway under list
  s.addShape("roundRect", {
    x: 0.5, y: 6.45, w: 12.3, h: 0.55, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("→ 5개 시그니처가 같은 3-cluster decomposition을 만든다 (High / Mid / Low saturation). 단일 architectural property의 redundant manifestation.", {
    x: 0.7, y: 6.45, w: 12.0, h: 0.55,
    fontSize: 11.5, fontFace: FONT_BODY, bold: true, italic: true, color: C.warm,
  });

  addFooter(s);
  addPageNumber(s, 23, 30);
}

// ========================================================================
// SLIDE 24 — Pillar B 동기 (왜 controlled LM swap이 필요한가)
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "P I L L A R   B — c o n t r o l l e d   L M   s w a p");
  addTitle(s, "왜 LM-only 통제 실험이 필요한가",
    "LLaVA-1.5(0.18) → LLaVA-Next(0.79) PMR jump 는 인코더 외 4-axis confound 가 섞여 있다.");

  // 4-axis confound box
  s.addShape("roundRect", {
    x: 0.5, y: 2.7, w: 6.0, h: 4.2, rectRadius: 0.1,
    fill: { color: C.paper }, line: { color: C.bad, width: 2 },
  });
  s.addText("LLaVA-1.5 vs LLaVA-Next의 4축 confound", {
    x: 0.7, y: 2.8, w: 5.6, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.bad,
  });
  s.addText([
    { text: "1. LM 백본:  Vicuna-7B → Mistral-7B-Instruct\n", options: { fontSize: 12, bold: true } },
    { text: "2. 시각 토큰 처리:  단일-tile 576 → AnyRes 다중 grid\n", options: { fontSize: 12 } },
    { text: "3. SFT 데이터:  LLaVA-Instruct-150K → LLaVA-Next 760K\n", options: { fontSize: 12 } },
    { text: "4. Vision-language 정렬:  단일-stage → 2-stage refresh\n", options: { fontSize: 12 } },
    { text: "\n→ 인코더 동일(CLIP-ViT-L-336) 한 가지로는 \"LM이 결정자\" 라고 결론 낼 수 없음.", options: { fontSize: 12, italic: true, color: C.warm } },
  ], {
    x: 0.7, y: 3.25, w: 5.6, h: 3.5,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 4,
  });

  // M-LMSwap design box
  s.addShape("roundRect", {
    x: 6.8, y: 2.7, w: 6.0, h: 4.2, rectRadius: 0.1,
    fill: { color: C.cardCircle }, line: { color: C.primary, width: 2 },
  });
  s.addText("M-LMSwap — controlled 단일축 LM swap", {
    x: 7.0, y: 2.8, w: 5.6, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  s.addText([
    { text: "공통:  CLIP-ViT-L-336 + 2-layer MLP projector\n", options: { fontSize: 12, bold: true } },
    { text: "Variant A:  + Vicuna-7B-v1.5\n", options: { fontSize: 12 } },
    { text: "Variant B:  + Mistral-7B-Instruct-v0.2\n", options: { fontSize: 12 } },
    { text: "\nLoRA(r=32, α=64): q/k/v/o_proj on LM\n", options: { fontSize: 11.5 } },
    { text: "Stage 1 — projector pretrain:  LCS-558K, 17K steps\n", options: { fontSize: 11.5 } },
    { text: "Stage 2 — instruction tune:  LLaVA-Instruct-665K, 21K steps\n", options: { fontSize: 11.5 } },
    { text: "\n→ 단일축 (LM 백본만) 통제 비교로 LM identity 효과 isolate.", options: { fontSize: 12, italic: true, color: C.primary, bold: true } },
  ], {
    x: 7.0, y: 3.25, w: 5.6, h: 3.6,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 4,
  });

  s.addText("주의:  LLaVA family는 M5b post-proj 인코더-side에서 NULL — Pillar B는 \"LM bottleneck인가\"의 직접 답.", {
    x: 0.5, y: 7.0, w: 12.5, h: 0.3,
    fontSize: 11, fontFace: FONT_BODY, color: C.inkSoft, italic: true,
  });

  addFooter(s);
  addPageNumber(s, 24, 30);
}

// ========================================================================
// SLIDE 25 — M-LMSwap training: Variant A 21K 완료 + Variant B 대기
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "P I L L A R   B — t r a i n i n g   s t a t u s");
  addTitle(s, "M-LMSwap Variant A 학습 완료 (21K) — Variant B 게이트 대기",
    "Vicuna 백본 학습이 완료되었으나 regression eval 게이트 통과는 실패. 진단 진행 중.");

  // Stage timeline (top)
  s.addShape("roundRect", {
    x: 0.5, y: 2.6, w: 12.3, h: 1.5, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.divider, width: 1 },
  });
  s.addText("학습 타임라인 — Variant A (CLIP+Vicuna)", {
    x: 0.7, y: 2.7, w: 11.9, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  // 4 stage boxes inline
  const stages = [
    { x: 0.7, label: "Stage 1\nprojector pretrain", sub: "LCS-558K · 17K steps\n~12h H200", color: C.primaryLight },
    { x: 3.7, label: "Stage 2\ninstruction tune", sub: "LLaVA-Instruct-665K\n21K steps · ~12h", color: C.primary },
    { x: 6.7, label: "Final ckpt\nstep21000", sub: "MLP + LoRA(r=32)\nadapter merged", color: C.teal },
    { x: 9.7, label: "Regression eval\n(gate)", sub: "PMR_nolabel + baseline cell\n+ generation sanity", color: C.warm },
  ];
  stages.forEach((st) => {
    s.addShape("roundRect", {
      x: st.x, y: 3.15, w: 2.8, h: 0.85, rectRadius: 0.05,
      fill: { color: st.color }, line: { color: st.color },
    });
    s.addText(st.label, {
      x: st.x + 0.05, y: 3.2, w: 2.7, h: 0.45,
      fontSize: 11, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(st.sub, {
      x: st.x + 0.05, y: 3.6, w: 2.7, h: 0.4,
      fontSize: 9, fontFace: FONT_BODY, color: "FFFFFF", align: "center", italic: true,
    });
  });

  // Gate result box (left)
  s.addShape("roundRect", {
    x: 0.5, y: 4.4, w: 6.0, h: 2.6, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.bad, width: 2 },
  });
  s.addText("Regression eval 결과 (step9000) — gate FAIL", {
    x: 0.7, y: 4.5, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.bad,
  });
  s.addText([
    { text: "Gate 1 — Generation sanity (>5 단어, no degeneracy):", options: { fontSize: 11 } },
    { text: "  PASS\n", options: { fontSize: 11, bold: true, color: C.good } },
    { text: "Gate 2 — PMR_nolabel ∈ [0.03, 0.50]:", options: { fontSize: 11 } },
    { text: "  FAIL  (0.825)\n", options: { fontSize: 11, bold: true, color: C.bad } },
    { text: "Gate 3 — line/blank/none baseline ≤ 0.6:", options: { fontSize: 11 } },
    { text: "  FAIL  (1.000)\n", options: { fontSize: 11, bold: true, color: C.bad } },
    { text: "\nA의 PMR 천장이 너무 높아 LLaVA-1.5(0.18) bracket 이 아님 — Vicuna+CLIP recipe 가 LLaVA-1.5 와 다른 학습 분포로 수렴.", options: { fontSize: 10, italic: true, color: C.inkSoft } },
  ], {
    x: 0.7, y: 4.95, w: 5.6, h: 2.0,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 3,
  });

  // Diagnostic insight (right)
  s.addShape("roundRect", {
    x: 6.8, y: 4.4, w: 6.0, h: 2.6, rectRadius: 0.08,
    fill: { color: C.cardCircle }, line: { color: C.teal, width: 2 },
  });
  s.addText("진단 — A↔B 비교 가능성은 살아있다", {
    x: 7.0, y: 4.5, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.teal,
  });
  s.addText([
    { text: "응답 다양성:  201 unique / 480 stim (image-blind 아님)\n", options: { fontSize: 11 } },
    { text: "축 ordering 보존:", options: { fontSize: 11, bold: true } },
    { text: "  line 0.483 < filled 0.908 ≤ shaded 1.000\n", options: { fontSize: 11 } },
    { text: "→ LLaVA-1.5 와 동일 ordering, 단지 +0.4 shifted up\n", options: { fontSize: 11, italic: true, color: C.teal } },
    { text: "\n결론:  A는 LLaVA-1.5 \"floor\" replicate 는 실패했지만,\n", options: { fontSize: 10.5 } },
    { text: "내부 PMR ordering 은 살아있어 ", options: { fontSize: 10.5 } },
    { text: "A↔B Δ-PMR 비교 자체는 유의미 가능", options: { fontSize: 10.5, bold: true, color: C.good } },
    { text: ".", options: { fontSize: 10.5 } },
  ], {
    x: 7.0, y: 4.95, w: 5.6, h: 2.0,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 3,
  });

  s.addText("결정 대기:  (1) gate override 후 B 진행 / (2) A recipe 재학습 — 슬라이드 27.", {
    x: 0.5, y: 7.0, w: 12.5, h: 0.3,
    fontSize: 11, fontFace: FONT_BODY, color: C.inkSoft, italic: true,
  });

  addFooter(s);
  addPageNumber(s, 25, 30);
}

// ========================================================================
// SLIDE 26 — Post-projection SAE round 2: regime-cross capacity ladder
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "M 5 b   r o u n d   2 — p o s t - p r o j e c t i o n   S A E");
  addTitle(s, "Post-projection SAE — regime-cross capacity ladder",
    "Projector 출력 위에서 SAE를 학습 → top-k feature 약 20-160개 ablate. circle cell 에서만 5-model 차이가 드러난다.");

  // 5-model x 4-cell table
  s.addShape("roundRect", {
    x: 0.5, y: 2.55, w: 12.3, h: 3.4, rectRadius: 0.05,
    fill: { color: C.paper }, line: { color: C.divider, width: 1 },
  });

  const headerY = 2.65;
  const rowH = 0.5;
  const cols = [
    { x: 0.7, w: 1.9, label: "모델" },
    { x: 2.6, w: 2.4, label: "circle / filled / blank+both" },
    { x: 5.0, w: 2.4, label: "circle / shaded / blank+none" },
    { x: 7.4, w: 2.4, label: "ball / filled / blank+both" },
    { x: 9.8, w: 2.9, label: "regime-cross 능력" },
  ];
  cols.forEach((c) => {
    s.addText(c.label, {
      x: c.x, y: headerY, w: c.w, h: rowH,
      fontSize: 11, fontFace: FONT_HEAD, bold: true, color: C.primary, align: "left",
    });
  });
  s.addShape("rect", {
    x: 0.7, y: headerY + 0.42, w: 11.9, h: 0.02,
    fill: { color: C.divider }, line: { color: C.divider },
  });

  const rows = [
    { name: "Qwen2.5-VL", c1: "k=20 → \"remain stationary\"\nclean break (PMR 1→0)", c2: "stays kinetic", c3: "stays kinetic", verdict: "★★★ — 가장 깨끗", color: C.good },
    { name: "Idefics2", c1: "k=20 → \"disappear\" (PMR 0)\nk=40+ → \"expand outward\"\n(non-monotonic)", c2: "stays \"spin\"\n(motion verb 유지)", c3: "stays \"fall/bounce\"", verdict: "★★ — 부분 break", color: C.warm },
    { name: "LLaVA-Next", c1: "k=40+ → \"expand\"\nk=20 → \"move downward\"", c2: "no break tested", c3: "stays \"fall down\"", verdict: "★★ — 부분 break", color: C.warm },
    { name: "LLaVA-1.5", c1: "baseline PMR=0\n(이미 abstract regime!)", c2: "no break tested", c3: "stays \"fall/roll\"", verdict: "✦ — 베이스라인 abstract", color: C.teal },
    { name: "InternVL3", c1: "stays \"fall downwards\"\nNULL at all k ≤ 160", c2: "drift only", c3: "stays \"fall\"", verdict: "✗ — 불가능", color: C.bad },
  ];
  rows.forEach((r, i) => {
    const y = headerY + 0.55 + i * 0.55;
    s.addText(r.name, {
      x: 0.7, y: y, w: 1.9, h: 0.55,
      fontSize: 10.5, fontFace: FONT_HEAD, bold: true, color: C.ink,
    });
    s.addText(r.c1, {
      x: 2.6, y: y, w: 2.4, h: 0.55,
      fontSize: 9.5, fontFace: FONT_BODY, color: C.ink,
    });
    s.addText(r.c2, {
      x: 5.0, y: y, w: 2.4, h: 0.55,
      fontSize: 9.5, fontFace: FONT_BODY, color: C.inkSoft,
    });
    s.addText(r.c3, {
      x: 7.4, y: y, w: 2.4, h: 0.55,
      fontSize: 9.5, fontFace: FONT_BODY, color: C.inkSoft,
    });
    s.addText(r.verdict, {
      x: 9.8, y: y, w: 2.9, h: 0.55,
      fontSize: 10.5, fontFace: FONT_HEAD, bold: true, color: r.color,
    });
  });

  // Insight callout
  s.addShape("roundRect", {
    x: 0.5, y: 6.05, w: 12.3, h: 1.0, rectRadius: 0.08,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "핵심 발견 — ", options: { fontSize: 12, bold: true, color: C.warm } },
    { text: "regime-cross capacity 는 ", options: { fontSize: 12 } },
    { text: "(1) circle vs ball 라벨 의존", options: { fontSize: 12, bold: true } },
    { text: ", (2) ", options: { fontSize: 12 } },
    { text: "5-model 사다리 (Qwen > Idefics2/Next > LLaVA-1.5 > InternVL3)", options: { fontSize: 12, bold: true } },
    { text: ".\nLLaVA-1.5 의 \"NULL\" 은 진짜 NULL 이 아니라 ", options: { fontSize: 12 } },
    { text: "circle cell 의 baseline 자체가 이미 abstract regime", options: { fontSize: 12, bold: true, color: C.teal } },
    { text: " — physics commitment 가 ablate 할 대상으로 존재하지 않음.", options: { fontSize: 12 } },
  ], {
    x: 0.7, y: 6.1, w: 11.9, h: 0.9,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 4,
  });

  addFooter(s);
  addPageNumber(s, 26, 30);
}

// ========================================================================
// SLIDE 27 — Regime-shift vs binary PMR + 결정 시점 (A1/A2/A3)
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "S C O R I N G   A R T I F A C T   &   D E C I S I O N");
  addTitle(s, "Regime-shift 는 보이지만 binary PMR 이 가린다 — 결정 시점",
    "텍스트는 \"fall\" → \"hit by arrow\" → \"roll\" 로 명확히 이동하지만, PMR=1 이 모두 동일 점수로 처리.");

  // Left: scoring artifact box
  s.addShape("roundRect", {
    x: 0.5, y: 2.65, w: 6.0, h: 4.0, rectRadius: 0.08,
    fill: { color: C.paper }, line: { color: C.warm, width: 2 },
  });
  s.addText("Scoring artifact — binary PMR 의 한계", {
    x: 0.7, y: 2.75, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.warm,
  });
  s.addText([
    { text: "예시 — LLaVA-1.5 ball+filled+blank+both:\n", options: { fontSize: 10.5, bold: true } },
    { text: "  baseline:        \"The ball will fall.\"  (PMR=1)\n", options: { fontSize: 10 } },
    { text: "  k=20 ablate:    \"hit by the red arrow.\" (PMR=1)\n", options: { fontSize: 10 } },
    { text: "  k=80 ablate:   \"will roll down the hill.\" (PMR=1)\n", options: { fontSize: 10 } },
    { text: "  k=160 ablate:  \"redrawn.\" (PMR=0, but rare)\n", options: { fontSize: 10 } },
    { text: "\nText 는 fall→hit→roll 로 분명히 이동하지만,\n모든 \"motion verb\" 가 PMR=1 으로 동등 점수.\n", options: { fontSize: 11, italic: true, color: C.bad } },
    { text: "\n결과:  M5b paper 헤드라인 \"NULL\" 은\n", options: { fontSize: 11, bold: true } },
    { text: "  (1) 일부는 진짜 NULL (InternVL3),\n", options: { fontSize: 10.5 } },
    { text: "  (2) 일부는 baseline-already-abstract\n        (LLaVA-1.5 + circle),\n", options: { fontSize: 10.5 } },
    { text: "  (3) 일부는 binary PMR 의 가림\n        (LLaVA-1.5 + ball: text 는 이동).\n", options: { fontSize: 10.5 } },
    { text: "→ \"NULL\" → \"regime-cross capacity ladder\" 로 reframe.", options: { fontSize: 11, bold: true, italic: true, color: C.teal } },
  ], {
    x: 0.7, y: 3.2, w: 5.6, h: 3.4,
    fontFace: FONT_BODY, color: C.ink, paraSpaceAfter: 2,
  });

  // Right: decision matrix
  s.addShape("roundRect", {
    x: 6.8, y: 2.65, w: 6.0, h: 4.0, rectRadius: 0.08,
    fill: { color: C.cardCircle }, line: { color: C.primary, width: 2 },
  });
  s.addText("결정 시점 — Pillar B 다음 한 발", {
    x: 7.0, y: 2.75, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.primary,
  });
  const decisions = [
    {
      tag: "A1",
      title: "Variant B 진행 (gate override)",
      cost: "GPU 24h · A의 +0.4 shift 같이 적용된다고 가정",
      pros: "+ A↔B Δ-PMR 비교 가능 (axis ordering 보존)",
      cons: "- A baseline 이 LLaVA-1.5 와 다른 분포",
      color: C.good,
    },
    {
      tag: "A2",
      title: "Variant A recipe 재학습",
      cost: "GPU 24h · LR / data-mix 조정 (회수 미보장)",
      pros: "+ LLaVA-1.5 floor 와 정렬 가능",
      cons: "- 확률적 — 또 실패하면 시간 손실",
      color: C.warm,
    },
    {
      tag: "A3",
      title: "M-PSwap (perceiver swap) 부활",
      cost: "현재 backlog · NaN 미해결 · 24h+",
      pros: "+ Idefics2 perceiver-resampler 가설 직접 검증",
      cons: "- 아직 학습 안정화 미해결",
      color: C.teal,
    },
  ];
  decisions.forEach((d, i) => {
    const y = 3.2 + i * 1.1;
    s.addShape("rect", {
      x: 7.0, y: y, w: 0.5, h: 1.0,
      fill: { color: d.color }, line: { color: d.color },
    });
    s.addText(d.tag, {
      x: 7.0, y: y, w: 0.5, h: 1.0,
      fontSize: 16, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", align: "center", valign: "middle",
    });
    s.addText(d.title, {
      x: 7.6, y: y, w: 5.2, h: 0.3,
      fontSize: 11.5, fontFace: FONT_HEAD, bold: true, color: C.ink,
    });
    s.addText([
      { text: d.cost + "\n", options: { fontSize: 9.5, italic: true, color: C.inkSoft } },
      { text: d.pros + "\n", options: { fontSize: 9.5, color: C.good } },
      { text: d.cons, options: { fontSize: 9.5, color: C.bad } },
    ], {
      x: 7.6, y: y + 0.3, w: 5.2, h: 0.7,
      fontFace: FONT_BODY, paraSpaceAfter: 1,
    });
  });

  // Bottom recommendation
  s.addShape("roundRect", {
    x: 0.5, y: 6.75, w: 12.3, h: 0.45, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "현재 권장 — ", options: { fontSize: 11.5, bold: true, color: C.warm } },
    { text: "step21000 final ckpt regression 재실행 (B6) 결과 본 후 A1 vs A2 결정.  병렬로 binary PMR → regime-shift score 정의 작업 (C3).", options: { fontSize: 11.5, color: C.ink } },
  ], {
    x: 0.7, y: 6.81, w: 11.9, h: 0.35,
    fontFace: FONT_BODY,
  });

  addFooter(s);
  addPageNumber(s, 27, 30);
}

// ========================================================================
// SLIDE 28 — Limitations & Future
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "L I M I T A T I O N S & F U T U R E");
  addTitle(s, "한계 — 그리고 다음 한 발",
    "Architecture-level finding 은 lock 되었지만, isolation 은 아직 미해결.");

  const items = [
    {
      t: "Projector isolation 미검증",
      b: "Idefics2의 perceiver-resampler가 leading candidate. Encoder/LM 동일 + perceiver↔MLP swap 으로 검증 필요 (M-PSwap LoRA 진행 중).",
      tag: "Pillar B"
    },
    {
      t: "LM-only counterfactual 부재",
      b: "LLaVA-1.5(0.18) → LLaVA-Next(0.70) PMR jump 는 4축 confound. CLIP+Vicuna vs CLIP+Mistral controlled swap (M-LMSwap, 진행 중).",
      tag: "Pillar B"
    },
    {
      t: "Single-task evaluation",
      b: "next-state-prediction 만 검증. Counting / spatial / causality 등 다른 shortcut 미검증. M-MP Pillar A 가 4-prompt 로 확장 (cross-method split @ Qwen×MCQ 발견)."
    },
    {
      t: "Human baseline 미수집",
      b: "M7 Prolific 20 raters × 50 stim 계획 — paper-blocking 단계. 인간이 \"같은 자극\"에서 같은 PMR 패턴을 보이는지 직접 검증."
    },
    {
      t: "External model 한정",
      b: "Pixtral / Phi-3.5-V / GPT-4V / Gemini-VL 등 닫힌 모델은 미테스트. 5 model 의 클러스터링이 더 큰 모델에서도 보존되는지 확인 필요."
    },
  ];

  items.forEach((it, i) => {
    const y0 = 2.6 + i * 0.85;
    s.addShape("roundRect", {
      x: 0.5, y: y0, w: 12.3, h: 0.78, rectRadius: 0.05,
      fill: { color: C.paper }, line: { color: C.divider, width: 1 },
    });
    s.addText(`${i + 1}. ${it.t}`, {
      x: 0.7, y: y0 + 0.07, w: 6.5, h: 0.35,
      fontSize: 13, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    if (it.tag) {
      s.addShape("roundRect", {
        x: 11.2, y: y0 + 0.1, w: 1.5, h: 0.3, rectRadius: 0.05,
        fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
      });
      s.addText(it.tag, {
        x: 11.2, y: y0 + 0.1, w: 1.5, h: 0.3,
        fontSize: 9, fontFace: FONT_BODY, bold: true, color: C.warm, align: "center",
      });
    }
    s.addText(it.b, {
      x: 0.7, y: y0 + 0.4, w: 11.5, h: 0.4,
      fontSize: 11, fontFace: FONT_BODY, color: C.ink,
    });
  });

  addFooter(s);
  addPageNumber(s, 28, 30);
}

// ========================================================================
// SLIDE 29 — Conclusion
// ========================================================================
{
  const s = addContentSlide();
  sectionStripe(s, "C O N C L U S I O N");
  addTitle(s, "원이 공으로 보이는 순간 — 정리",
    "행동 → 메커니즘 → 픽셀의 3차원으로 \"shortcut의 위치\"를 좁혀 들어갔다.");

  const conclusions = [
    {
      n: "1",
      t: "Architecture-level reframe",
      b: "행동 PMR ceiling 은 인코더 표현력 단독으로 결정 안 됨. 5-fold downstream signature 가 동일 architectural property 의 redundant manifestation. CLIP 2-point 비교(LLaVA-1.5 vs Next)가 가장 깨끗한 disconfirmer.",
    },
    {
      n: "2",
      t: "Causal localization (LM + Encoder)",
      b: "M5a runtime steering: 3 of 4 모델 10/10 PMR flip. M5b SAE encoder ablation: 3 of 5 break, 2 LLaVA NULL. \"CLIP cluster commitment 는 LM-side direction 으로만 라우팅, 비-CLIP 은 encoder + LM 둘 다.\"",
    },
    {
      n: "3",
      t: "Pixel encodability — architecture-conditional",
      b: "픽셀 공간 gradient ascent로 3 of 5 모델 flip. Idefics2 9-layer 0/10 → perceiver-resampler 가 forward 정보 통과는 시키되 inverse 픽셀 routability 만 차단. M4 + M5a + §4.6 dissociation 으로 가설 정밀화.",
    },
  ];

  conclusions.forEach((c, i) => {
    const y0 = 2.45 + i * 1.25;
    s.addShape("roundRect", {
      x: 0.5, y: y0, w: 12.3, h: 1.15, rectRadius: 0.08,
      fill: { color: C.paper }, line: { color: C.primary, width: 1.5 },
    });
    s.addShape("rect", {
      x: 0.5, y: y0, w: 0.55, h: 1.15,
      fill: { color: C.primary }, line: { color: C.primary },
    });
    s.addText(c.n, {
      x: 0.5, y: y0 + 0.25, w: 0.55, h: 0.65,
      fontSize: 26, fontFace: FONT_HEAD, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(c.t, {
      x: 1.25, y: y0 + 0.08, w: 11.4, h: 0.35,
      fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.primary,
    });
    s.addText(c.b, {
      x: 1.25, y: y0 + 0.42, w: 11.4, h: 0.7,
      fontSize: 11, fontFace: FONT_BODY, color: C.ink,
    });
  });

  s.addShape("roundRect", {
    x: 0.5, y: 6.4, w: 12.3, h: 0.6, rectRadius: 0.05,
    fill: { color: C.accentSoft }, line: { color: C.accent, width: 1 },
  });
  s.addText("Big picture — VLM의 \"원→공\" shortcut은 단순 model quirk가 아니라 architecture-level saturation의 다차원적 표현이다.", {
    x: 0.7, y: 6.4, w: 12.0, h: 0.6,
    fontSize: 12.5, fontFace: FONT_BODY, bold: true, italic: true, color: C.warm,
  });

  addFooter(s);
  addPageNumber(s, 29, 30);
}

// ========================================================================
// SLIDE 30 — Q&A / closing
// ========================================================================
{
  const s = pres.addSlide();
  addBackground(s, C.bgDark);
  s.addShape("rect", {
    x: 0, y: 0, w: W, h: 0.18,
    fill: { color: C.accent }, line: { color: C.accent },
  });

  s.addText("감사합니다", {
    x: 0.5, y: 1.6, w: W - 1, h: 1.4,
    fontSize: 60, fontFace: FONT_HEAD, bold: true,
    color: "FFFFFF", align: "center",
  });

  s.addShape("rect", {
    x: W / 2 - 0.6, y: 3.1, w: 1.2, h: 0.06,
    fill: { color: C.accent }, line: { color: C.accent },
  });

  s.addText("Q & A · 토론 · 다음 우선순위 논의", {
    x: 0.5, y: 3.3, w: W - 1, h: 0.55,
    fontSize: 22, fontFace: FONT_BODY, italic: true,
    color: "DEE6F2", align: "center",
  });

  // bottom resources
  s.addShape("roundRect", {
    x: 2.5, y: 4.6, w: W - 5, h: 1.7, rectRadius: 0.1,
    fill: { color: "1A2A50" }, line: { color: C.accent, width: 1 },
  });
  s.addText("동반 자료", {
    x: 2.7, y: 4.7, w: 8, h: 0.4,
    fontSize: 14, fontFace: FONT_HEAD, bold: true, color: C.accent,
  });
  s.addText("• 슬라이드별 상세 한국어 해설:  docs/review_ppt/physical_mode_storyline_ko.md\n• 본 논문 자료 인덱스:  references/roadmap.md (single source of truth)\n• 가설 evidence chain:  docs/hypotheses.md / per-milestone insights:  docs/insights/*.md", {
    x: 2.7, y: 5.05, w: W - 5.4, h: 1.2,
    fontSize: 12, fontFace: FONT_BODY,
    color: "DEE6F2", paraSpaceAfter: 4,
  });

  s.addText("Repo · 구성:  src/physical_mode/ (package) · scripts/0{1..6}_*.py · scripts/sec4_6_*.py · scripts/sae_*.py", {
    x: 0.5, y: 6.8, w: W - 1, h: 0.4,
    fontSize: 11, fontFace: FONT_BODY,
    color: "8FA0BD", align: "center", italic: true,
  });

  addPageNumber(s, 30, 30, true);
}

// ========================================================================
// Save
// ========================================================================
pres.writeFile({ fileName: OUT })
  .then((f) => console.log("Saved:", f))
  .catch((e) => { console.error(e); process.exit(1); });
