# Part 4 — Components

Everything on the car that is not the milled body, as exact parametric geometry
with regulation gates, CFD patches, a mass rollup and the manufacturing files.

| module | what |
|---|---|
| `component_contract.py` | every regulation number used (T3, T6, T7, T8, T9) |
| `wheel.py` | the team's wheel STLs in SLS PA12 with a 6 mm seat for one 3x6x2.5 bearing (`team_stl`, the default): mass and inertia measured from the mesh. Also the closed spinning disc the CFD uses, and an older parametric wheel model with reference designs |
| `beam_support.py` | the team's support architecture, parametric: one PA12 beam per axle through the body, printed 3 mm stub axles, optional wheel-face discs; T7.12.1, T3.7 and cutter-radius gates |
| `support.py` | an alternative NACA strut support (not the team's architecture) |
| `wings.py` | NACA front wing with flap, rear wing and pylon, scrutineer-style gates |
| `nose.py` | printed nose cone ahead of Ref A (a PA12 shell) |
| `joints.py` | printed positives glued into milled negatives: a ball-end-safe pocket per part, the printed part that fills it, and the manufacturing files |
| `assembly.py` | all parts for one car: right-half STL per CFD patch, `extra_surfaces` for Part 2, `fixed_hardware_kwargs` for the mass rollup, gates |

```bash
python assembly.py --body body_half.stl --out parts/
python run_all_tests.py
```

## Manufacturing files (`joints.make_all`)

Written to `manufacture/`, in **millimetres**, both halves, one closed solid per part:

| file | what | material |
|---|---|---|
| `machined_body.stl` | the body as milled: every pocket cut, ending at Ref A | foam, 0.163 g/cm3 |
| `printed_front_assembly.stl` | nose cone (hollow shell with bulkhead and a 3 mm powder drain, or solid when `wall_mm` leaves no cavity), front wing, flap and mount, and the tenon: ONE part, because the wing mount runs through the cone | PA12 |
| `printed_support_front.stl`, `printed_support_rear.stl` | one part per axle: beam, stub axles, disc, the keel that fills the drop-in channel, and that axle's tether guide | PA12 |
| `printed_rwing.stl` | rear wing and pylon, trimmed at the body surface (surface-bonded: no pocket fits over the cartridge wall) | PLA |
| `printed_wheel_front_x2.stl`, `printed_wheel_rear_x2.stl` | the team wheels with the bearing seat (print two of each) | PA12 |

No two of these occupy the same space. Before 2026-09-30 the nose, front wing, front
tether guide and front support were four overlapping files, and 0.85 cm3 of overlap
was counted twice in the mass.

`make_all` re-reads every file it writes and reports whether it is a closed solid
with the expected number of pieces, and its mass. `manufactured_g` is the sum: with
the wheels, bearings and halo it is the car's mass on the scale, which is what
Part 5 sizes the body against and checks T3.6 with.

## Evidence behind the defaults

GitHub Actions CFD, medium mesh, all parts in the flow: wheels are about 64 % of the
car's drag. Front wing at +6 deg with an 8 mm flap at 30 deg; minimum-section rear
wing. The v2 rear wheel-support CAD bottoms out at 1.40 mm, 0.10 mm under T3.7, and
its 12.7 mm disc blocks the T7.13 hang-test claw; the parametric beam support uses a
12.0 mm disc.
