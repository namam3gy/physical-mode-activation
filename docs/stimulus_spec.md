# Stimulus spec — PhysCue (Sub-task 1)

Canonical definitions for the factorial axes + rendering choices. Keep this
file in sync with `src/physical_mode/stimuli/` and `src/physical_mode/config.py`.

## Canvas

- 512 × 512 px, RGB, white background.
- Ground line at y = 0.78 × canvas = 399 px when `bg_level ∈ {ground, scene}`.
- Object default radius: 64 px (128 px diameter).

## Axis A — object abstraction

| Level | Rendering | Cue claim (`references/project.md` §1.3) |
|---|---|---|
| `line` | 3-px black outline, white fill | pure geometric; weakest physical cue |
| `filled` | 2-px outline, uniform medium-gray fill | some mass cue; still geometric |
| `shaded` | radial gradient, light from top-left (Ramachandran 1988 light-from-above prior) | strongest *static* 3D cue |
| `textured` | shaded sphere + meridian arc + 7 dark spots (soccer-ball cue) | highest level rendered programmatically |
| `block_stack` | three stacked jittered rectangles | alternative "clearly physical" stimulus without ball semantics |

The `block_stack` level is intentionally *not* in the pilot or MVP-full configs —
it's reserved for a follow-up that targets the "physical object ≠ ball" axis.

## Axis B — background

| Level | Rendering |
|---|---|
| `blank` | no additions |
| `ground` | single 3-px horizontal black line at y=399 |
| `scene` | ground line + sky/ground shading + small ramp obstacle on the right (Gibson-style surface + optic-flow-compatible support plane) |

## Axis C — context cue

| Level | Rendering |
|---|---|
| `none` | no additions |
| `cast_shadow` | elliptical cast shadow on the ground (Kersten et al. 1997 ground-attachment cue); rendered even when bg = blank |
| `motion_arrow` | red directional arrow whose heading is chosen per event template |
| `both` | cast_shadow + motion_arrow |
| `wind` (legacy) | five clusters of short grey arcs anchored to one side of the object — found invisible to Qwen2.5-VL in the pilot, retained for reproducibility |
| `arrow_shadow` (legacy) | the pilot's combined cue, equivalent to `both` |

## Axis D — object label (prompt-time)

| Level | Prompt phrasing |
|---|---|
| `circle` | "The image shows a circle. …" |
| `ball` | "The image shows a ball. …" |
| `planet` | "The image shows a planet. …" |
| `shape` | "The image shows a shape. …" |
| `object` | "The image shows an object. …" |

Controls whether language prior alone can force physics-mode (research H2).
The pilot uses a single label (`ball`); MVP-full uses `(circle, ball, planet)`.

## Event templates

| Event | Object position | Expected physical answer |
|---|---|---|
| `fall` | center-top (cx=256, cy=128) | falls to ground |
| `horizontal` | mid-left (cx=179, cy=230) | moves sideways (esp. with wind cue) |
| `hover` | center-upper (cx=256, cy=205) | stays in air (tests for over-attribution) |
| `wall_bounce` | right-mid (cx=358, cy=230) | bounces back |
| `roll_slope` | left-lower (cx=128, cy=307) | rolls down the ramp |

Pilot uses `fall` + `horizontal`; MVP-full uses `fall` only (pilot showed
the two were behaviorally indistinguishable). The remaining three are
scaffolded for a later round.

## Seed discipline

`FactorialSpec.base_seed` (default 1000) is the starting value; each
(object, bg, cue, event, variant_index) cell gets a unique seed from a
monotonically incrementing counter. The same factorial spec **always** produces
the same stimuli — verified by `tests/test_stimuli_deterministic.py`.
