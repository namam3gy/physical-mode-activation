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
    "nothing will happen",
    "nothing happens",
    "won't move",
    "will not move",
    "cannot move",
    "does not move",
    "doesn't move",
})
