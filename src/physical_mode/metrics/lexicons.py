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
