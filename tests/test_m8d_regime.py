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
    """'It is a silhouette; the bird flies' → abstract wins because of explicit reject."""
    assert classify_regime("bird", "It is just a silhouette — nothing physical happens.") == "abstract"
