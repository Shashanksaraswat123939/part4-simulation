# Part 4 — Components

Everything on the car that is not the milled body, as exact parametric geometry
with regulation gates, CFD patches and a mass rollup.

| module | what |
|---|---|
| `component_contract.py` | every regulation number used (T3, T6, T7, T8, T9) |
| `wheel.py` | parametric wheel: mass and inertia (within 1.1 % of the v2 CAD), stiffness relative to the CAD wheel, designs, CFD wheel surface |
| `wings.py` | NACA front and rear wings, mount and pylon, scrutineer-style gates |
| `nose.py` | parametric printed nose cone (off by default: -0.4 % on the current wings) |
| `support.py` | parametric NACA strut wheel support inside the T7.12.1 cylinder, ~0.5 g vs 1.4-1.5 g CAD pods |
| `assembly.py` | all parts for one car: right-half STL per patch, `extra_surfaces` for Part 2, `fixed_hardware_kwargs` for Part 2's mass rollup, gates |

```bash
python assembly.py --body body_half.stl --out parts/
python run_all_tests.py
```

Evidence behind the defaults (GitHub Actions CFD, 2026-09-25): wheels are 64–75 % of
the car's drag; a flat front wing 5.5 mm ahead of the front wheels is drag-neutral
(it shields the wheels by exactly its own drag); the endplates tried added 3–6 %.

Wheel designs (mean inertia per wheel): `carbon_rim_film` 101.5 g·mm² (default:
open carbon wheel closed by 0.10 mm film faces, so the rotating-wall CFD is honest),
`carbon_rim_capped` 124.9, `carbon_rim` 85.6, `abs_light` 117.0,
`cad_v2` 137.7. Aero shape per wheel: `shoulder_in`/`shoulder_out` (rounded tyre
shoulders beyond the T7.4 contact width) and `dome` (outboard hubcap), priced as
printed shells; `assembly.build` also takes a `(front, rear)` pair of `Wheel`.

The v2 rear wheel-support CAD bottoms out at 1.40 mm, 0.10 mm under T3.7;
`assembly.build` trims it at 1.51 mm. Change the CAD to match.
