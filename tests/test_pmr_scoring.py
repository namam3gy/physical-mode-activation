"""Hand-labeled smoke tests for PMR / GAR / abstract-reject scoring."""

from __future__ import annotations

import pytest

from physical_mode.metrics.pmr import (
    score_abstract_reject,
    score_gar,
    score_hold_still,
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
