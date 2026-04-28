"""Hand-labeled smoke tests for PMR / GAR / abstract-reject scoring."""

from __future__ import annotations

import pytest

from physical_mode.metrics.pmr import (
    score_abstract_reject,
    score_describe,
    score_for_variant,
    score_gar,
    score_hold_still,
    score_meta_yesno,
    score_pmr,
)


PMR_POSITIVE = [
    "It will fall down due to gravity.",
    "The ball rolls to the right.",
    "The object bounces off the wall.",
    "It slides along the ground.",
    "The ball drops onto the floor.",
    "The ball will land on the ground.",
    "The blocks will topple over.",
    "It tumbles down the slope.",
    "The sphere accelerates toward the ground.",
    "It moves to the right and then falls.",
    # Regression: conjugations that older "move" stem missed.
    "The ball will continue moving forward due to its momentum.",
    "The ball will continue to roll forward on the surface.",
    "The ball is rotating as it descends.",
    "It is gliding across the floor.",
    "The ball plunges into the ground.",
    # Korean fallback (§4.3 cross-model: Idefics2 / LLaVA-Next emit Hangul).
    "행성이 땅으로 떨어지는 중.",
    "행성이 지면으로 떨어지고 있습니다.",
    "지면으로 떨어 지기 시작.",
    "위로 이동.",
    "공이 움직이기 시작될 것입니다.",
    # Japanese fallback (§4.3 ext: anticipated kanji/hiragana responses).
    "ボールは地面に落ちます。",          # ball falls to ground
    "ボールが転がっていきます。",         # ball rolls away
    "円は移動するでしょう。",            # circle will move
    "惑星は落下します。",                # planet falls
    "ボールが動いています。",             # ball is moving
    "ボールがドロップ.",                 # Katakana drop
    # Chinese fallback (§4.3 ext: Idefics2 emits Chinese on Japanese 惑星).
    "惑星会向下落下.",                   # planet falls down
    "惑星会掉入黑洞.",                   # planet falls into black hole
    "惑星会下降.",                       # planet descends
    "惑星向下跌落.",                     # planet falls
    "惑星会向下坠落.",                   # planet plummets
    "惑星将会继续飞行.",                 # planet will continue to fly
]

PMR_NEGATIVE = [
    "This is just a circle on a page.",
    "It is an abstract geometric shape — nothing physical happens.",
    "It stays in place.",
    "The circle will remain stationary.",
    "Nothing happens; it is a 2D shape.",
    "It is a drawing of a circle.",
    "The shape does not move.",
    "It is suspended in the air indefinitely.",
    "It is a diagram; there is no motion.",
    "The image shows a circle. No change occurs.",
    # Korean negatives.
    "원이 더 작아지기 시작될 것",     # abstract change, no physics verb
    "행성이 그대로 있을 것입니다.",      # 그대로 abstract marker
    "공이 움직이지 않을 것입니다.",     # 움직이지 않 abstract marker
    # Japanese negatives.
    "ボールはそのままです。",                # そのまま abstract marker
    "円は動かない状態です。",                # 動かない abstract marker
    "惑星は静止しています。",                # 静止 abstract marker
    # §4.6 random-direction control patterns: scorer was false-positiving
    # via "mov" stem inside "no indication of movement".
    "The circle will remain stationary as there is no indication of movement or change in its position.",
    "The circle will likely remain stationary as there is no indication of motion or force acting upon it in the image.",
    "The shape remains stationary; nothing causes movement.",
]


@pytest.mark.parametrize("text", PMR_POSITIVE)
def test_pmr_positive(text):
    assert score_pmr(text) == 1, f"false negative on: {text!r}"


@pytest.mark.parametrize("text", PMR_NEGATIVE)
def test_pmr_negative(text):
    assert score_pmr(text) == 0, f"false positive on: {text!r}"


def test_gar_only_when_ground_and_fall():
    # No ground → None.
    assert score_gar("It falls down.", event_template="fall", bg_level="blank") is None
    # Ground present and fall event → 1 if "down" present, 0 otherwise.
    assert score_gar("It falls down to the ground.", "fall", "ground") == 1
    assert score_gar("It rolls to the right.", "fall", "ground") == 0
    # Ground present but event is horizontal → None.
    assert score_gar("It moves right.", "horizontal", "ground") is None


def test_hold_still_and_abstract_reject():
    assert score_hold_still("The ball stays still.") == 1
    assert score_hold_still("The ball falls.") == 0
    assert score_abstract_reject("This is an abstract geometric shape.") == 1
    assert score_abstract_reject("The ball rolls away.") == 0


def test_abstract_reject_suppresses_physics_verb_match():
    """'This is just a circle, so it won't move' should NOT count as PMR=1."""
    s = "This is just a circle; it will not move."
    assert score_pmr(s) == 0


# ---------------------------------------------------------------------------
# M-MP Phase 1+2 scorers (Track B Pillar A — multi-prompt cross-task)
# ---------------------------------------------------------------------------


META_YES = [
    "Yes",
    "No",  # only the position matters; "no" should also pass
    "Yes.",
    "No.",
    "Yes, this is a depiction of a real-world physical event.",
    "No, the image is a simple geometric representation.",
    "Answer: Yes",
    "Answer: no",
    "yeah, gravity pulls it down",
    "Nope.",
]
META_YES_EXPECTED = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
META_UNPARSE = ["Maybe?", "", "I'm not sure", "...", "12345"]


@pytest.mark.parametrize("text,expected", list(zip(META_YES, META_YES_EXPECTED)))
def test_meta_yesno_parseable(text, expected):
    assert score_meta_yesno(text) == expected


@pytest.mark.parametrize("text", META_UNPARSE)
def test_meta_yesno_unparseable_returns_minus_one(text):
    assert score_meta_yesno(text) == -1


DESCRIBE_PHYSICS = [
    "A bowling ball suspended above a bowling lane.",
    "Gravity pulls the ball down.",
    "A ball with arrow pointing downward, suggesting it is about to fall.",
    "The ball rolls toward the surface.",
    "A planet falling from space.",
    "The ball is in motion.",
    "The sphere is hovering above the floor.",
    "The object is in mid-air.",
]

DESCRIBE_ABSTRACT = [
    "A simple black outline of a circle.",
    "A circle is depicted in the image.",
    "Simple line drawing of a sphere on white background.",
    "A geometric symbol on a plain background.",
    "The image shows a sketch of a circle.",
]


@pytest.mark.parametrize("text", DESCRIBE_PHYSICS)
def test_describe_physics_mode(text):
    assert score_describe(text) == 1, f"Expected physics-mode for: {text!r}"


@pytest.mark.parametrize("text", DESCRIBE_ABSTRACT)
def test_describe_abstract_mode(text):
    assert score_describe(text) == 0, f"Expected abstract-mode for: {text!r}"


def test_describe_physics_wins_over_abstract_framing():
    """Physics tokens win over abstract-framing markers when both are present.

    Rationale: the M-MP G1 question is whether the model expresses
    physics-mode *commitment*. If the model says "ball that might fall"
    — even hedged inside a 'line drawing' framing — that IS a physics-mode
    commitment about the depicted object. Pure geometric framing without
    any physics token (e.g., "A simple outline of a circle") still scores 0.
    """
    s = "A simple line drawing of a ball that might fall."
    assert score_describe(s) == 1
    # Pure geometric framing without physics → still 0
    assert score_describe("A simple line drawing of a circle.") == 0


# ---------------------------------------------------------------------------
# score_for_variant — Phase 3 cross-prompt dispatch
# ---------------------------------------------------------------------------


def test_score_for_variant_open_uses_score_pmr():
    """open variant routes to score_pmr (kinetic-prediction lexicon)."""
    assert score_for_variant("The ball will fall down.", "open") == 1
    assert score_for_variant("The circle stays the same.", "open") == 0
    assert score_for_variant("This is just an abstract shape.", "open") == 0


def test_score_for_variant_describe_uses_score_describe():
    """describe_scene variant routes to score_describe (description lexicon)."""
    assert score_for_variant("A bowling ball suspended above a lane.", "describe_scene") == 1
    assert score_for_variant("A simple outline of a circle.", "describe_scene") == 0


def test_score_for_variant_yesno_treats_unparseable_as_zero():
    """meta_phys_yesno variant returns 1/0; -1 (unparseable) → 0."""
    assert score_for_variant("Yes, this is real.", "meta_phys_yesno") == 1
    assert score_for_variant("No, it's an abstract shape.", "meta_phys_yesno") == 0
    assert score_for_variant("Maybe?", "meta_phys_yesno") == 0  # unparseable → 0
    assert score_for_variant("", "meta_phys_yesno") == 0


def test_score_for_variant_unknown_falls_back_to_score_pmr():
    """Unknown variant (forced_choice / forced_choice_no_label) falls back to score_pmr."""
    # Forced-choice text "A) It falls down." has the kinetic stem "fall"
    assert score_for_variant("A) It falls down.", "forced_choice") == 1
    assert score_for_variant("D) Abstract shape.", "forced_choice") == 0


def test_score_for_variant_open_no_label():
    """open_no_label variant routes to score_pmr."""
    assert score_for_variant("It falls.", "open_no_label") == 1
    assert score_for_variant("This is a static circle.", "open_no_label") == 0
