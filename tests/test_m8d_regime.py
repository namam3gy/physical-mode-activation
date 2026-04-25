"""Tests for M8d regime classifier (kinetic / static / abstract / ambiguous)."""

from __future__ import annotations

import pytest

from physical_mode.metrics.pmr import classify_regime


# (category, response_text, expected_regime)
KINETIC_CASES = [
    ("car",    "The car drives forward at high speed.",         "kinetic"),
    ("car",    "It rolls along the road.",                       "kinetic"),
    ("car",    "The race car speeds away.",                      "kinetic"),
    ("person", "The person walks forward.",                      "kinetic"),
    ("person", "The athlete runs across the field.",             "kinetic"),
    ("bird",   "The bird flies into the sky.",                   "kinetic"),
    ("bird",   "The duck swims across the pond.",                "kinetic"),
    ("bird",   "The duck waddles along the shore.",              "kinetic"),
]

STATIC_CASES = [
    ("car",    "The car is parked beside the curb.",             "static"),
    ("car",    "The figurine stays on display.",                 "static"),
    ("person", "The statue stands motionless in the square.",    "static"),
    ("person", "The person stays still and waits.",              "static"),
    ("bird",   "The bird perches on the branch.",                "static"),
]

ABSTRACT_CASES = [
    ("car",    "This is just a drawing of a rectangle.",         "abstract"),
    ("person", "This is an abstract stick figure — nothing happens.", "abstract"),
    ("bird",   "It is a silhouette; nothing moves.",             "abstract"),
]

AMBIGUOUS_CASES = [
    ("car",    "Hmm, I am not sure.",                            "ambiguous"),
    ("bird",   "It is unclear what will happen.",                "ambiguous"),
]


@pytest.mark.parametrize("category,text,expected", KINETIC_CASES)
def test_kinetic_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", STATIC_CASES)
def test_static_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", ABSTRACT_CASES)
def test_abstract_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", AMBIGUOUS_CASES)
def test_ambiguous_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


def test_abstract_overrides_kinetic_keyword():
    """An abstract reject phrase must override even an explicit kinetic verb."""
    s = "It is just a silhouette; nothing physical happens. The car drives nowhere."
    assert classify_regime("car", s) == "abstract"


# ---------------------------------------------------------------------------
# Label-echo regression tests.
#
# When the M8d abstract-role label ("silhouette" / "stick figure" / "figurine")
# is used to fill the OPEN_TEMPLATE prompt, the prompt itself contains the
# label text and the model often echoes it. classify_regime must NOT be
# fooled into the abstract bucket just because the response contains
# "silhouette" — the abstract decision has to come from the model, not
# from the label.
# ---------------------------------------------------------------------------

LABEL_ECHO_KINETIC_CASES = [
    ("car",    "The silhouette drives forward at high speed.",  "kinetic"),
    ("car",    "The figurine moves toward the parking lot.",    "kinetic"),
    ("bird",   "The silhouette of the duck flies away.",        "kinetic"),
    ("person", "The stick figure walks forward.",               "kinetic"),
]

LABEL_ECHO_STATIC_CASES = [
    ("car",    "The figurine stays on the shelf.",              "static"),
    ("person", "The stick figure stands still in the field.",   "static"),
]


@pytest.mark.parametrize("category,text,expected", LABEL_ECHO_KINETIC_CASES)
def test_label_echo_kinetic(category, text, expected):
    """When model echoes the M8d label (silhouette / stick figure / figurine) in a kinetic
    response, regime should still be kinetic — silhouette must NOT be in ABSTRACT_MARKERS."""
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", LABEL_ECHO_STATIC_CASES)
def test_label_echo_static(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


def test_unsupported_category_raises():
    """classify_regime called for an M8a category (or any unknown) must raise."""
    with pytest.raises(ValueError, match="circle"):
        classify_regime("circle", "The ball rolls down.")


# Bare-stem coverage tests — pin the "mov" stem regression for the I1 fix
# (replacing "moves"/"moving"/"moved" with stem "mov", which also covers "move").
BARE_MOVE_CASES = [
    ("car",    "The car will move forward.",            "kinetic"),
    ("car",    "The car moves toward the parking lot.", "kinetic"),
    ("person", "The person will move toward the door.", "kinetic"),
    ("bird",   "The bird moves through the air.",       "kinetic"),
]


@pytest.mark.parametrize("category,text,expected", BARE_MOVE_CASES)
def test_bare_move_stem(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"
