"""
assembly.py -- every non-body part of the car, as CFD patches + a mass rollup.

    build(W, x_front, d_halo, body_half_stl, out_dir) -> dict

writes, in out_dir (right halves, metres, ASCII STL):
    wheelF.stl, wheelR.stl   closed wheels, sunk 0.3 mm into the track, ROTATING
    supports.stl             v2 CAD wheel supports (Part 1 hardware_cad)
    halo.stl                 halo + helmet (Part 1 hardware_cad)
    fwing.stl                front wing + its centreline mount
    rwing.stl                rear wing + pylon
    tethers.stl              T6 tether line guides
    assembly.json            everything below

assembly.json carries:
    extra_surfaces           exactly what Part 2's OpenFOAMRunConfig /
                             AdjointRunConfig.extra_surfaces take
    fixed_hardware_kwargs    exactly what Part 2's FixedHardwareSpec takes, so
                             Part 3's mass rollup sees the real parts
    wheel_moi_kg_m2          the designed wheel, for the race objective
    gates                    every regulation margin Part 4 can measure (mm)

Why each part is in the CFD: measured on GitHub Actions (rnd-cfd, 2026-09-25)
wheels are 64-75 % of the car's drag and the body only 13-22 %.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in ("part1-simulation", "part2-simulation"):
    _d = HERE.parent / _p
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.append(str(_d))

import component_contract as cc   # noqa: E402
import wheel as wh                # noqa: E402
import wings as wg                # noqa: E402

SPEED_MPS = 20.0
SINK_MM = 0.3
PART_MATERIAL = "PLA"           # wings, mounts, tether guides: printed, solid


def _export_half(mesh, path: Path) -> None:
    m = mesh.copy()
    m.vertices[m.vertices[:, 1] < 0.0, 1] = 0.0
    m.export(str(path), file_type="stl_ascii")


def _half_of(mesh):
    """Right half (y >= 0) of a centreline part, capped on the symmetry plane."""
    import trimesh
    return trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], [0, 0, 0], cap=True)


def tether_guide(x_mm: float, id_mm: float = 4.5, od_mm: float = 8.0,
                 length_mm: float = 2.0, z_bottom_mm: float = 2.0):
    import trimesh
    ring = trimesh.creation.annulus(r_min=id_mm / 2e3, r_max=od_mm / 2e3,
                                    height=length_mm / 1e3, sections=48)
    ring.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    ring.apply_translation([x_mm / 1e3, 0.0, (z_bottom_mm + od_mm / 2) / 1e3])
    return _half_of(ring), dict(id_mm=id_mm, od_mm=od_mm, x_mm=x_mm,
                                bottom_mm=z_bottom_mm)


def _full_mass_com(half_meshes, density_g_cm3: float):
    """Mass (kg) and COM (m) of the FULL symmetric part from its right half."""
    vol = sum(abs(m.volume) for m in half_meshes)
    if vol <= 0:
        return 0.0, (0.0, 0.0, 0.0)
    cx = sum(abs(m.volume) * m.center_mass[0] for m in half_meshes) / vol
    cz = sum(abs(m.volume) * m.center_mass[2] for m in half_meshes) / vol
    return 2 * vol * density_g_cm3 * 1e3, (cx, 0.0, cz)


def fixed_hardware_kwargs(W_mm, x_front_mm, d_halo_mm, parts_mass: dict,
                          wheel_design: str) -> dict:
    """Kwargs for Part 2's FixedHardwareSpec from the REAL parts.

    The spec has a single "rear_wing" slot; it carries every aero part here
    (front wing + mount, rear wing + pylon, tether guides) at their combined
    COM, replacing the old 5 g placeholder.
    """
    import fixed_hardware as fh
    from geometry_contract import CO2_MASS_KG, R_WHEEL_M
    ref_a = cc.ref_plane_A(x_front_mm) / 1e3
    ref_b = cc.ref_plane_B(x_front_mm, W_mm) / 1e3
    rear_face = ref_b + 0.040
    hw = fh.compute_default_fixed_hardware_inputs(W_mm, x_front_mm, d_halo_mm,
                                                  ref_a, ref_b, rear_face_x_m=rear_face)
    halo = hw["halo_geometry"]
    hz = [z for _y, z in halo.cross_section_yz_m]
    f, r = wh.design(wheel_design)
    sup_f, sup_r = 1.396e-3, 1.545e-3               # v2 CAD supports, each (fixed_hardware)
    mf = 2 * (f.mass * 1e-3 + sup_f)
    mr = 2 * (r.mass * 1e-3 + sup_r)
    xf, xr = x_front_mm / 1e3, (x_front_mm + W_mm) / 1e3
    aero = [v for k, v in parts_mass.items() if k in ("fwing", "rwing", "tethers")]
    m_aero = sum(m for m, _c in aero)
    c_aero = tuple(sum(m * c[i] for m, c in aero) / m_aero for i in range(3)) if m_aero else (rear_face, 0, 0.05)
    return {
        "co2_cartridge_mass_kg": CO2_MASS_KG,
        "co2_cartridge_com": tuple(v / 1e3 for v in hw["canister_com_mm"]),
        "rear_wing_mass_kg": m_aero,
        "rear_wing_com": c_aero,
        "wheels_axles_mass_kg": mf + mr,
        "wheels_axles_com": ((mf * xf + mr * xr) / (mf + mr), 0.0, R_WHEEL_M),
        "wheels_front_mass_kg": mf, "wheels_front_com": (xf, 0.0, R_WHEEL_M),
        "wheels_rear_mass_kg": mr, "wheels_rear_com": (xr, 0.0, R_WHEEL_M),
        "halo_mass_kg": fh.HALO_MASS_KG,
        "halo_com": (0.5 * (halo.x_front_m + halo.x_rear_m), 0.0, 0.5 * (min(hz) + max(hz))),
    }


def build(W_mm: float, x_front_mm: float, d_halo_mm: float, body_half_stl: str,
          out_dir: str, wheel_design: str = "carbon_rim_capped",
          front: wg.FrontWing = wg.FrontWing(), rear: wg.RearWing = wg.RearWing(),
          rotate_wheels: bool = True) -> dict:
    import trimesh
    import hardware_geometry as hg
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = trimesh.load(str(body_half_stl), force="mesh")
    f, r = wh.design(wheel_design)
    from geometry_contract import FRONT_WHEEL_INNER_Y_MM as FY, REAR_WHEEL_INNER_Y_MM as RY
    x_rear = x_front_mm + W_mm
    R = 0.5 * (f.R + r.R)

    surfaces, gates, parts_mass, info = [], {}, {}, {}

    # -- wheels (rotating closed cylinders) --------------------------------
    for name, x, y_in, w in (("wheelF", x_front_mm, FY, f.w), ("wheelR", x_rear, RY, r.w)):
        cyl = wh.cfd_surface(R, w, x, y_in, SINK_MM)
        _export_half(cyl, out / f"{name}.stl")
        origin = (x / 1e3, (y_in + w / 2) / 1e3, (R - SINK_MM) / 1e3)
        surfaces.append({"name": name, "stl": str(out / f"{name}.stl"),
                         "rotating": ({"origin": origin, "axis": (0, 1, 0),
                                       "omega": -SPEED_MPS / (R / 1e3)} if rotate_wheels else None)})
    gates["T7.5_diameter_min"] = 2 * R - cc.WHEEL_DIA_MIN
    gates["T7.5_diameter_max"] = cc.WHEEL_DIA_MAX - 2 * R
    gates["T7.4_front_contact"] = f.w - cc.FRONT_CONTACT_MIN
    gates["T7.4_rear_contact"] = r.w - cc.REAR_CONTACT_MIN
    gates["T7.2_front_gap"] = 2 * FY - cc.FRONT_GAP_MIN
    gates["T7.2_rear_gap"] = 2 * RY - cc.REAR_GAP_MIN

    # -- supports and halo (CAD) -------------------------------------------
    sup = [hg.build_wheel_assembly(a, x)[f"{a}_wheel_support_right"]
           for a, x in (("front", x_front_mm), ("rear", x_rear))]
    _export_half(trimesh.util.concatenate(sup), out / "supports.stl")
    surfaces.append({"name": "supports", "stl": str(out / "supports.stl"), "rotating": None})
    halo = hg.build_halo(cc.ref_plane_A(x_front_mm) / 1e3, d_halo_mm)["halo_right"]
    _export_half(halo, out / "halo.stl")
    surfaces.append({"name": "halo", "stl": str(out / "halo.stl"), "rotating": None})

    # -- front wing --------------------------------------------------------
    fwm = wg.build_front_wing(front, x_front_mm, R)
    gates.update(wg.front_wing_gates(front, fwm, x_front_mm, R))
    _export_half(trimesh.util.concatenate(list(fwm.values())), out / "fwing.stl")
    surfaces.append({"name": "fwing", "stl": str(out / "fwing.stl"), "rotating": None})
    parts_mass["fwing"] = _full_mass_com(list(fwm.values()), cc.DENSITY_G_CM3[PART_MATERIAL])

    # -- rear wing ---------------------------------------------------------
    rwm = wg.build_rear_wing(rear, x_front_mm, W_mm, body)
    rparts = {k: rwm[k] for k in ("wing", "pylon")}
    forward = [body, halo] + sup
    gates.update(wg.rear_wing_gates(rear, rparts, x_front_mm, W_mm, body, forward))
    _export_half(trimesh.util.concatenate(list(rparts.values())), out / "rwing.stl")
    surfaces.append({"name": "rwing", "stl": str(out / "rwing.stl"), "rotating": None})
    parts_mass["rwing"] = _full_mass_com(list(rparts.values()), cc.DENSITY_G_CM3[PART_MATERIAL])
    info["rear_wing_z_chord_mm"] = rwm["_z_chord_mm"]
    info["body_top_under_rear_wing_mm"] = rwm["_body_top_mm"]

    # -- tether guides (T6) -------------------------------------------------
    tg = []
    for x in (x_front_mm - 5.0, x_rear + 5.0):
        m, meta = tether_guide(x)
        tg.append(m)
        info.setdefault("tethers", []).append(meta)
    _export_half(trimesh.util.concatenate(tg), out / "tethers.stl")
    surfaces.append({"name": "tethers", "stl": str(out / "tethers.stl"), "rotating": None})
    parts_mass["tethers"] = _full_mass_com(tg, cc.DENSITY_G_CM3[PART_MATERIAL])
    gates["T6.1_front_guide"] = cc.TETHER_FROM_AXLE_MAX - 5.0
    gates["T6.1_rear_guide"] = cc.TETHER_FROM_AXLE_MAX - 5.0
    gates["T6.2_id_min"] = 4.5 - cc.TETHER_ID_MIN
    gates["T6.2_id_max"] = cc.TETHER_ID_MAX - 4.5
    gates["T3.7_tether_clearance"] = 2.0 - cc.TRACK_CLEARANCE_MIN

    fhk = fixed_hardware_kwargs(W_mm, x_front_mm, d_halo_mm, parts_mass, wheel_design)
    result = {
        "W_mm": W_mm, "x_front_mm": x_front_mm, "d_halo_mm": d_halo_mm,
        "wheel_design": wh.summary(wheel_design),
        "wheel_moi_kg_m2": wh.mean_inertia_kg_m2(wheel_design),
        "extra_surfaces": surfaces,
        "parts_mass_g": {k: v[0] * 1e3 for k, v in parts_mass.items()},
        "fixed_hardware_kwargs": fhk,
        "gates": gates,
        "failed_gates": sorted(k for k, v in gates.items() if v < -1e-9),
        "front_wing": wg.params_dict(front), "rear_wing": wg.params_dict(rear),
        "info": info,
    }
    (out / "assembly.json").write_text(json.dumps(result, indent=2, default=list))
    return result


def extra_surfaces_from(assembly_json: str) -> tuple:
    """Load the Part 2 extra_surfaces tuple back from assembly.json."""
    d = json.loads(Path(assembly_json).read_text())
    out = []
    for s in d["extra_surfaces"]:
        rot = s["rotating"]
        if rot:
            rot = {"origin": tuple(rot["origin"]), "axis": tuple(rot["axis"]),
                   "omega": rot["omega"]}
        out.append({"name": s["name"], "stl": s["stl"], "rotating": rot})
    return tuple(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=120.3)
    ap.add_argument("--x-front", type=float, default=46.0)
    ap.add_argument("--d-halo", type=float, default=43.72)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wheels", default="carbon_rim_capped")
    a = ap.parse_args()
    res = build(a.W, a.x_front, a.d_halo, a.body, a.out, a.wheels)
    print(json.dumps({k: res[k] for k in ("parts_mass_g", "wheel_moi_kg_m2", "failed_gates")},
                     indent=2))
