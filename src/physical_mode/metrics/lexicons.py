"""Word lists for PMR / GAR scoring.

The lexicons are deliberately conservative and hand-curated. Expand them via
docs/scoring_rubric.md whenever a false negative is observed in the run logs.
"""

from __future__ import annotations

# Verbs / verb-phrase cores that indicate the model predicted a physical event.
# Each entry is a *stem* that we match against word prefixes (cheap lemmatization).
PHYSICS_VERB_STEMS: frozenset[str] = frozenset({
    # Stems are matched via word.startswith(stem). Use the shortest form that
    # covers all English inflections of interest (e.g., "mov" catches
    # move/moves/moved/moving; "move" would miss "moving").
    "fall",      # fall, falls, falling, fallen
    "fell",      # fell (past of fall)
    "drop",      # drop, drops, dropped, dropping
    "roll",      # roll, rolls, rolled, rolling
    "bounc",     # bounce, bounces, bounced, bouncing
    "slid",      # slide, slides, slid, sliding (also "slide")
    "slip",      # slip, slips, slipped, slipping
    "land",      # land, lands, landed, landing
    "tumbl",     # tumble, tumbles, tumbled, tumbling
    "tip",       # tip, tips, tipped, tipping
    "toppl",     # topple(s/d/ing)
    "sway",      # sway(s/ed/ing)
    "swing",     # swing(s/ing)
    "swung",     # past of swing
    "hit",       # hit, hits, hitting
    "collid",    # collide(s/d/ing)
    "impact",    # impact(s/ed)
    "plumm",     # plummet(s/ed)
    "descend",   # descend(s/ing/ed)
    "sink",      # sink, sinks, sinking
    "sank",      # past of sink
    "spin",      # spin, spins, spinning
    "spun",      # past of spin
    "launch",    # launch(es/ed/ing)
    "fly",       # fly
    "fli",       # flies, flied
    "flew",
    "flown",
    "travel",    # travel(s/ing/led)
    "mov",       # move, moves, moved, moving
    "accelerat", # accelerate(s/d/ing)
    "shift",     # shift(s/ed/ing)
    "crash",     # crash(es/ed/ing)
    "continu",   # continue(s/d/ing)
    "rotat",     # rotate(s/d/ing)
    "push",      # push(es/ed/ing)
    "pull",      # pull(s/ed/ing)
    "drift",     # drift(s/ed/ing)
    "glid",      # glide, glides, glided, gliding
    "plung",     # plunge(s/d/ing)
})

# Verbs that indicate the model explicitly said nothing physical will happen.
HOLD_STILL_STEMS: frozenset[str] = frozenset({
    "stay",      # stay(s/ed/ing)
    "remain",
    "rest",
    "sit",       # sit(s/ting)
    "hold",
    "hover",     # hover(s/ed/ing) — ambiguous but we count as "not fall"
    "float",
    "still",
    "stationary",
    "suspended",
    "unchanged",
    "same",
})

# Markers that the model predicted downward motion (for GAR).
DOWN_DIRECTION_PHRASES: frozenset[str] = frozenset({
    "down",
    "downward",
    "downwards",
    "below",
    "to the ground",
    "onto the ground",
    "onto the floor",
    "to the floor",
    "hits the ground",
    "hit the ground",
    "lands on the ground",
    "lands on the floor",
})

# Explicit "abstract / geometric" cues — the model rejected physical interpretation.
#
# IMPORTANT: do NOT add tokens that overlap with M8d *abstract-role labels*
# (silhouette / stick figure / figurine / duck / statue — see
# inference.prompts.LABELS_BY_SHAPE). The OPEN_TEMPLATE prompt echoes the
# label, so any model response on a label-stim contains the label text.
# Including a label here would force PMR=0 for *every* response on that
# stim — manufacturing the H7 abstract pole at the classifier level
# instead of measuring it from model behavior.
ABSTRACT_MARKERS: frozenset[str] = frozenset({
    "abstract",
    "geometric",
    "not a physical",
    "just a shape",
    "just a circle",
    "just an image",
    "a drawing of",
    "a diagram",
    "it is a circle",
    "this is a circle",
    "two-dimensional",
    "2d shape",
    "no motion",
    "nothing moves",
    "nothing physical",
    "nothing will happen",
    "nothing happens",
    "won't move",
    "will not move",
    "cannot move",
    "does not move",
    "doesn't move",
})


# Korean physics-verb stems for §4.3-style runs where the model occasionally
# emits Hangul-only responses. These are matched as substrings against the
# raw text (Korean is agglutinative and not space-segmented like English, so
# substring matching on a stem like "떨어" reliably catches all inflections
# of 떨어지다 — 떨어진다 / 떨어지는 / 떨어졌다 / 떨어지고 / 떨어지기).
#
# IMPORTANT: same overlap-with-label rule as ABSTRACT_MARKERS — do NOT add
# Korean tokens that overlap with §4.3 labels (공 / 원 / 행성). The OPEN_TEMPLATE
# echoes the label, so any added overlapping token would force PMR=1 on every
# response.
KOREAN_PHYSICS_VERB_STEMS: frozenset[str] = frozenset({
    "떨어",     # 떨어지다 fall/drop (떨어진다, 떨어졌다, 떨어지는, 떨어지고)
    "낙하",     # 낙하 falling (Sino-Korean noun)
    "추락",     # 추락 crash/plummet
    "굴러",     # 굴러가다 roll
    "구르",     # 구르다 roll (alt form)
    "움직",     # 움직이다 move (움직이고, 움직이기, 움직였다)
    "이동",     # 이동 movement (Sino-Korean)
    "튀어",     # 튀어오르다 bounce up
    "튀기",     # 튀기다 bounce
    "미끄러",   # 미끄러지다 slide/slip
    "흔들",     # 흔들리다 sway/shake
    "쓰러",     # 쓰러지다 fall over/topple
    "기울",     # 기울다 tilt/lean
    "회전",     # 회전 rotation
    "돌아",     # 돌아가다 turn/spin
    "날아",     # 날아가다 fly
    "비행",     # 비행 flight
    "부딪",     # 부딪치다 collide
    "충돌",     # 충돌 collision
    "내려",     # 내려가다 go down, 내려오다 come down
    "떠올",     # 떠오르다 rise
    "오르",     # 오르다 climb/rise
    "가속",     # 가속 acceleration
    "튕기",     # 튕기다 bounce off
})

# Korean abstract / hold-still markers — substring matched against text.
# Conservative set: only phrases that strongly imply abstract framing or
# explicit no-motion. Avoid bare "정지" (could appear inside "정지하지 않고"
# = "without stopping" inside a kinetic response) and "이미지" / "도형"
# (overlap with prompt echoes).
KOREAN_ABSTRACT_MARKERS: frozenset[str] = frozenset({
    "그대로",         # as is, unchanged
    "변하지 않",      # doesn't change
    "변화 없",        # no change
    "변동 없",        # no movement/change
    "움직이지 않",    # doesn't move
    "변동이 없",      # no change
    "추상",           # abstract
})


# Japanese physics-verb stems for §4.3-style runs where the model
# emits Japanese-only or Japanese-mixed responses. Stems are matched
# as substrings against the raw text. Japanese is also non-space-
# segmented like Korean, so substring matching catches verb conjugations
# (落ちる / 落ちて / 落ちた / 落ちている all contain 落ち).
#
# Same overlap-with-label rule as KOREAN_PHYSICS_VERB_STEMS — do NOT
# add Japanese tokens that overlap with §4.3 labels (ボール / 円 / 惑星).
# In particular: 動 alone overlaps with 運動 / 動物 / 動作 (compound
# nouns); we use 動く + 動い to bind the verb-form context.
JAPANESE_PHYSICS_VERB_STEMS: frozenset[str] = frozenset({
    "落ち",         # 落ちる fall (落ちる / 落ちて / 落ちた / 落ちます / 落ちている)
    "落下",         # 落下 falling (Sino-Japanese)
    "落とし",       # 落とす drop (transitive)
    "動く",         # 動く move (verb form)
    "動い",         # 動いている / 動いて / 動いた (move conjugations)
    "移動",         # 移動 movement
    "転が",         # 転がる roll
    "跳ね",         # 跳ねる bounce
    "滑",           # 滑る slide (also 滑り)
    "飛ん",         # 飛んで / 飛んだ / 飛んでいる fly
    "飛び",         # 飛び去る / 飛び立つ
    "衝突",         # 衝突 collision
    "ぶつか",       # ぶつかる collide
    "回転",         # 回転 rotation
    "揺れ",         # 揺れる sway
    "倒れ",         # 倒れる fall over
    "転落",         # 転落 falling
    "降り",         # 降りる descend
    "加速",         # 加速 acceleration
    "墜落",         # 墜落 crash
    "ドロップ",     # Katakana loanword for "drop" (physical drop)
})


# Chinese (simplified) physics-verb stems for §4.3-style runs.
# Surfaced when Idefics2 (Mistral-7B LM, limited Japanese SFT) responded
# to Japanese label `惑星` (which is also a valid Chinese word for
# planet) with Chinese-language physics descriptions — 19/80 responses
# on Idefics2's `惑星` arm were simplified Chinese:
# "惑星会向下落下", "惑星掉入黑洞", "惑星会下降", etc.
#
# Same overlap-with-label rule as the Korean / Japanese sets — none of
# `球` / `圆` / `行星` (Chinese names of the 3 §4.3 roles) appear here.
# Note: some stems overlap with JA (落下, 加速); duplication is harmless
# (substring matching short-circuits on first match).
CHINESE_PHYSICS_VERB_STEMS: frozenset[str] = frozenset({
    "下落",         # fall down (Chinese phrasing distinct from JA 落ち)
    "掉入",         # fall into
    "掉落",         # fall down
    "跌落",         # fall
    "坠落",         # plummet (CN simplified 坠, JA uses 墜)
    "下降",         # descend
    "旋转",         # rotate (CN simplified 旋转, JA uses 回転)
    "飞行",         # fly (CN simplified 飞, JA uses 飛)
    "滚动",         # roll
    "弹起",         # bounce
    "撞击",         # impact
    "倒下",         # fall over
    "落地",         # land on the ground
    "下坠",         # plummet down
})

# Japanese abstract / hold-still markers — substring matched against text.
# Conservative set; avoids ambiguous tokens like 静か (quiet, often
# kinetic-context-compatible).
JAPANESE_ABSTRACT_MARKERS: frozenset[str] = frozenset({
    "動かない",     # doesn't move (verb negation)
    "動きません",   # doesn't move (polite)
    "そのまま",     # as is, unchanged
    "変わらない",   # doesn't change
    "変化なし",     # no change
    "変化はない",   # no change is
    "静止",         # stationary
    "抽象",         # abstract
})


# ---------------------------------------------------------------------------
# M8d category-specific regime keywords.
# Used by metrics.pmr.classify_regime to assign one of {kinetic, static,
# abstract, ambiguous} to a model response.
# ---------------------------------------------------------------------------

# Stems are matched via word.startswith(stem) (see metrics.pmr._any_stem_hit).
# Use the shortest form that covers all English inflections of interest:
#   - "mov"  catches move/moves/moved/moving
#   - "stay" catches stay/stays/stayed/staying
#   - "rest" catches rest/rests/rested/resting
# Phrases with spaces (e.g., "stand still") cannot match here; if needed,
# add them to a phrase-based lexicon instead.
# Universal kinetic stems — gravity / collision / locomotion that apply across
# car / person / bird (and any future M8d category). The qwen smoke surfaced
# many "the car will fall into the hole" / "the figurine will drop and land"
# responses being mis-classified as `ambiguous` because the per-category
# kinetic set didn't include gravity-fall stems. Adding them here keeps the
# per-category sets focused on category-distinctive verbs.
UNIVERSAL_KINETIC_STEMS: frozenset[str] = frozenset({
    "fall",     # fall, falls, falling, fallen
    "fell",     # past of fall
    "drop",     # drop, drops, dropped, dropping
    "land",     # land, lands, landing
    "plumm",    # plummet
    "descend",  # descend(s/ing)
    "sink",     # sink, sinks, sinking
    "sank",     # past of sink
    "plung",    # plunge
    "tumbl",    # tumble
    "dive",     # dive, dives, diving, dived
    "crash",    # crash(es/ed)
    "collid",   # collide
    "impact",   # impact
    "hit",      # hit, hits, hitting
    "bounc",    # bounce
    "slid",     # slide, slid, sliding
    "slip",     # slip, slipping
    "spin",     # spin, spinning
    "spun",     # past of spin
    "rotat",    # rotate, rotates, rotating
})

CATEGORY_REGIME_KEYWORDS: dict[str, dict[str, frozenset[str]]] = {
    "car": {
        # "drift" intentionally NOT included — drift is racing-kinetic but
        # also colloquial static ("the car drifts to a stop"); ambiguous.
        "kinetic": frozenset({"driv", "roll", "spee", "mov", "race", "accel", "trav", "head"}),
        # "remain" added (smoke showed "remain stationary" + "remains stopped"
        # are common static phrasings).
        "static":  frozenset({"park", "stop", "stay", "still", "stationary", "display", "remain"}),
    },
    "person": {
        "kinetic": frozenset({"walk", "run", "jog", "step", "stride", "mov", "march", "pace", "jump"}),
        "static":  frozenset({"stand", "stay", "still", "stationary", "motionless", "frozen", "sit", "rest", "remain"}),
    },
    "bird": {
        "kinetic": frozenset({"fly", "fli", "flew", "flown", "swim", "swam", "soar", "waddl", "mov", "glid", "flap", "hop"}),
        "static":  frozenset({"perch", "sit", "stay", "still", "stationary", "rest", "remain"}),
    },
}
