"""Sanity tests that M8d additions to type aliases and label tables exist."""

from __future__ import annotations

from physical_mode.config import FactorialSpec
from physical_mode.inference.prompts import LABELS_BY_SHAPE, labels_for_shape


def test_m8d_shapes_factorial_iter():
    """FactorialSpec accepts the new M8d shapes without error."""
    spec = FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line",),
        bg_levels=("blank",),
        cue_levels=("none",),
        event_templates=("fall",),
        seeds_per_cell=1,
    )
    rows = list(spec.iter())
    assert len(rows) == 3
    assert {r.shape for r in rows} == {"car", "person", "bird"}


def test_m8d_labels_by_shape_present():
    """All three new categories registered in LABELS_BY_SHAPE.

    C6-alt-A (2026-05-01) extends car/person/bird from 3-tuple to 5-tuple
    by adding "subtype" and "style" labels. The first 3 (physical, abstract,
    exotic) are still the original triplet — newer roles are appended.
    """
    for category in ("car", "person", "bird"):
        assert category in LABELS_BY_SHAPE, f"{category!r} missing from LABELS_BY_SHAPE"
        triplet = LABELS_BY_SHAPE[category]
        assert isinstance(triplet, tuple), f"{category!r} entry not a tuple"
        assert len(triplet) >= 3, f"{category!r} entry too short: {triplet}"
        assert all(isinstance(x, str) for x in triplet), f"{category!r} entry has non-str: {triplet}"


def test_m8d_label_triplet_values():
    """Spec-pinned first-3 (physical / abstract / exotic) per category — preserved
    after C6-alt-A 5-tuple extension."""
    assert LABELS_BY_SHAPE["car"][:3]    == ("car",    "silhouette",  "figurine")
    assert LABELS_BY_SHAPE["person"][:3] == ("person", "stick figure", "statue")
    assert LABELS_BY_SHAPE["bird"][:3]   == ("bird",   "silhouette",  "duck")


def test_m8d_labels_for_shape():
    """labels_for_shape() returns the configured triplet (first 3)."""
    assert labels_for_shape("car")[:3]    == ("car",    "silhouette",  "figurine")
    assert labels_for_shape("person")[:3] == ("person", "stick figure", "statue")
    assert labels_for_shape("bird")[:3]   == ("bird",   "silhouette",  "duck")


def test_c6_alt_a_extended_labels():
    """C6-alt-A (2026-05-01): subtype + style roles added at index 3+4."""
    assert LABELS_BY_SHAPE["car"]    == ("car",    "silhouette",   "figurine", "sedan",  "cartoon")
    assert LABELS_BY_SHAPE["person"] == ("person", "stick figure", "statue",   "human",  "sketch")
    assert LABELS_BY_SHAPE["bird"]   == ("bird",   "silhouette",   "duck",     "eagle",  "drawing")


def test_c6_orig_new_categories_present():
    """C6-orig (2026-05-01): boat / fish / plant categories registered."""
    for cat in ("boat", "fish", "plant"):
        assert cat in LABELS_BY_SHAPE, f"{cat!r} missing"
        triplet = LABELS_BY_SHAPE[cat]
        assert len(triplet) == 3, f"{cat!r} should be 3-tuple"
        # First label is the literal physical name.
        assert triplet[0] == cat
