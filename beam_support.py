"""
beam_support.py -- the team's wheel-support system, built from numbers.

The format is the team's own support CAD (hardware_cad/*_wheel_support.stl,
measured 2026-10-01): one printed part per axle, wheel to wheel, with the
AXLE BUILT IN and the bearing pressed onto it.

  beam     through the body along y, superellipse section (x width, z height,
           exponent p). It drops into a channel machined from below, so the
           section's tightest curvature must admit the 6.25 mm ball
           (radius >= 3.125 mm).
  disc     at each wheel, covering its open inner side: radius, thickness,
           and how far it sits inside the rim (the team's rear disc sits
           1 mm inside; their front one stands 2.5 mm inboard).
  boss     the flare from the disc down to the axle, a concave fillet (theirs:
           5.8 mm at the disc, ~3 mm radius). Its root diameter is SIZED HERE
           for the loads below.
  stub     the axle itself: a shoulder the bearing's inner race sits against,
           then the 3 mm journal the 3x6x2.5 bearing presses onto, out to the
           wheel's outer face.
  hubcap   a separate cap pressed on the stub's end, flush with the wheel's
           outer face: it closes the wheel's outer side and holds the wheel
           on. It does not turn (it is on the axle, not the wheel).

LOADS ("normal racing loads", assumptions to adjust with the team):
  radial 8 N at the bearing (the whole car at ~33 g on one axle; launch
  load transfer is about 1 N a wheel and the finish catch under 1 N, so this
  is being set down hard or pressed on), and 3 N sideways at the tyre's
  contact (~12 g of the car on the two wheels of one side). SLS PA12 at
  44 MPa with a safety factor of 2, and at most 0.15 mm of sag at the bearing
  so the wheel stays clear of the disc. The 3 mm journal is fixed by the
  bearing and is the weak point: at safety 2 it carries 3.2 N sideways, no
  more (a steel pin there would lift that). Everything thicker is sized to
  these loads.

Material SLS PA12 (1.01 g/cm3, team spec 2026-09-27). One piece left-to-right;
the builder returns the right half (y >= 0), like every other Part 4 part.

Regulations: T7.12.1 -- a support may only exist inside the cylinder through
the two opposing wheels' diameters, i.e. within R of the axle axis; T3.7 track
clearance >= 1.5 mm; T7.13 needs hang-test clearance between the wheel's inner
corner and the BODY (a disc is support, not body).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import component_contract as cc

PA12_G_CM3 = 1.01
TOOL_R_MM = 3.125
STUB_D_MM = 3.0               # the journal: the 3x6x2.5 bearing's bore
BEARING_W_MM = 2.5
RIM_INNER_R_MM = 13.72        # the team wheel's rim, inside (measured on the STL)

# Loads and limits (see the module docstring)
LOAD_RADIAL_N = 8.0
LOAD_AXIAL_N = 3.0
PA12_STRENGTH_MPA = 44.0
PA12_E_MPA = 1650.0
SAFETY = 2.0
SAG_MAX_MM = 0.15


@dataclass(frozen=True)
class BeamSupport:
    # 7 mm round by default: the slot must admit the 6.25 mm ball, so a round
    # beam can be no smaller, and its 38 mm2 matches the v2 CAD's ribbed
    # section (~35 mm2). A solid 14 x 10 mm beam weighed 5.3 g a side.
    beam_w_mm: float = 7.0        # x
    beam_h_mm: float = 7.0        # z
    p: float = 2.0                # 2 = ellipse (smoothest), larger = boxier
    x_offset_mm: float = 0.0      # beam centre relative to the axle
    z_offset_mm: float = 0.0
    disc_front: bool = True       # the team's CAD has a disc at every wheel
    disc_rear: bool = True
    disc_r_mm: float = 12.0       # <= axle height - 1.5 mm (T3.7): 12.25 here
    disc_t_mm: float = 1.0
    disc_recess_mm: float = 1.0   # disc's outer face this far inside the rim
    #                               (negative: standing inboard of the wheel)
    boss_d_mm: float = 5.8        # the flare's diameter at the disc, at least
    fillet_r_mm: float = 3.0      # the flare's radius
    shoulder_d_mm: float = 4.0    # what the bearing's inner race sits against
    hubcap: bool = True
    hubcap_r_mm: float = 12.0      # as the disc: 1.5 mm off the track (T3.7)
    hubcap_t_mm: float = 0.8
    material_g_cm3: float = PA12_G_CM3


def section(bs: BeamSupport, n: int = 64) -> np.ndarray:
    """(x, z) of the beam section about its own centre, mm."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    a, b = bs.beam_w_mm / 2, bs.beam_h_mm / 2
    return np.c_[a * np.sign(c) * np.abs(c) ** (2 / bs.p), b * np.sign(s) * np.abs(s) ** (2 / bs.p)]


def min_curvature_radius_mm(bs: BeamSupport) -> float:
    """Smallest radius of curvature of the section -- the slot's tightest
    corner, which the ball-end must fit."""
    P = section(bs, 720)
    d1 = (np.roll(P, -1, 0) - np.roll(P, 1, 0)) / 2
    d2 = np.roll(P, -1, 0) - 2 * P + np.roll(P, 1, 0)
    k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(
        np.linalg.norm(d1, axis=1) ** 3, 1e-12)
    return float(1.0 / k.max())


def _extrude_xz(poly_xz_mm, y0_mm, y1_mm, x_mm, z_mm):
    import trimesh
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(poly_xz_mm).buffer(0), (y1_mm - y0_mm) / 1e3)
    m.apply_scale([1e-3, 1e-3, 1.0])
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    m.apply_translation([x_mm / 1e3, y0_mm / 1e3, z_mm / 1e3])
    return m


def _revolve_y(profile_ry_mm, x_mm, z_mm, sections: int = 96):
    """Solid of revolution about the axle (the line x, z along y), metres,
    from a closed (radius, y) outline that starts and ends on the axis."""
    import trimesh
    m = trimesh.creation.revolve(np.asarray(profile_ry_mm, float), sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    m.apply_scale(1e-3)
    m.apply_translation([x_mm / 1e3, 0.0, z_mm / 1e3])
    if m.volume < 0:
        m.invert()
    return m


def stub_outline(bs: BeamSupport, y_disc_mm: float, y_bearing_mm: float, y_tip_mm: float,
                 root_d_mm: float) -> list:
    """(radius, y) from the disc's outer face to the stub's tip: the flare
    (a concave arc of fillet_r from root_d at the disc down to the shoulder),
    the shoulder up to the bearing's inner face, the journal to the tip. The
    flare is shortened if the space before the bearing is too small."""
    r_s, r_j = bs.shoulder_d_mm / 2, STUB_D_MM / 2
    r_root = max(root_d_mm / 2, r_s)
    y_b0 = y_bearing_mm - BEARING_W_MM / 2
    room = max(y_b0 - 0.3 - y_disc_mm, 0.2)             # 0.3 mm of plain shoulder
    rf = max(bs.fillet_r_mm, r_root - r_s)
    th = math.acos(max(-1.0, 1 - (r_root - r_s) / rf)) if r_root > r_s else 0.0
    if rf * math.sin(th) > room:                         # too long: a straight taper instead
        pts = [(r_root, y_disc_mm), (r_s, y_disc_mm + room)]
    else:
        y_e = y_disc_mm + rf * math.sin(th)
        pts = [(r_s + rf * (1 - math.cos(t)), y_e - rf * math.sin(t)) for t in np.linspace(th, 0, 12)]
    return pts + [(r_s, y_b0), (r_j, y_b0), (r_j, y_tip_mm)]


def structure(bs: BeamSupport, outline: list, y_bearing_mm: float, y_fixed_mm: float,
              R_mm: float) -> dict:
    """What the loads do to the stub, as a cantilever fixed at y_fixed (the
    disc): the peak bending stress along it and the sag at the bearing. The
    radial load acts at the bearing; the side load at the tyre's contact puts
    its moment (force x wheel radius) into the whole stub."""
    r_of = lambda y: float(np.interp(y, [q[1] for q in outline], [q[0] for q in outline]))
    ys = np.linspace(y_fixed_mm, y_bearing_mm, 200)
    M = LOAD_RADIAL_N * (y_bearing_mm - ys) + LOAD_AXIAL_N * R_mm          # N mm
    d = 2 * np.array([r_of(y) for y in ys])
    sigma = 32 * M / (math.pi * d ** 3)
    # sag: integrate the curvature M / EI twice from the fixed end
    k = M / (PA12_E_MPA * math.pi * d ** 4 / 64)
    slope = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(ys))])
    sag = float(np.sum(0.5 * (slope[1:] + slope[:-1]) * np.diff(ys)))
    return {"stress_mpa": float(sigma.max()), "allow_mpa": PA12_STRENGTH_MPA / SAFETY,
            "stress_at_y_mm": float(ys[int(sigma.argmax())]), "sag_mm": sag,
            "journal_stress_mpa": float(32 * LOAD_AXIAL_N * R_mm / (math.pi * STUB_D_MM ** 3))}


def sized_root_d_mm(bs: BeamSupport, y_disc_mm: float, y_bearing_mm: float, R_mm: float) -> float:
    """The boss diameter at the disc: the larger of the asked boss_d and what
    the root's bending moment needs at the allowed stress."""
    M = LOAD_RADIAL_N * (y_bearing_mm - y_disc_mm) + LOAD_AXIAL_N * R_mm
    need = (32 * M / (math.pi * PA12_STRENGTH_MPA / SAFETY)) ** (1 / 3)
    return max(bs.boss_d_mm, need, bs.shoulder_d_mm)


def build(bs: BeamSupport, x_axle_mm: float, z_axle_mm: float, y_inner_mm: float,
          wheel_width_mm: float, disc: bool, hub_mm: tuple = (5.25, 8.25),
          R_mm: float = 14.12) -> dict:
    """Right half of one axle's support, metres: beam, disc, boss and stub as
    one piece, and the hubcap as its own. `hub_mm` is where the wheel's hub
    (its bearing seat) lies, measured from the wheel's inner face."""
    import trimesh
    y_out = y_inner_mm + wheel_width_mm
    y_disc = y_inner_mm + bs.disc_recess_mm                # the disc's outer face
    y_beam_end = y_disc - (bs.disc_t_mm if disc else 0.0)
    y_bear = y_inner_mm + 0.5 * (hub_mm[0] + hub_mm[1])
    y_tip = y_out - (bs.hubcap_t_mm if bs.hubcap else 0.0)
    xc, zc = x_axle_mm + bs.x_offset_mm, z_axle_mm + bs.z_offset_mm
    root_d = sized_root_d_mm(bs, y_disc, y_bear, R_mm)
    line = stub_outline(bs, y_disc, y_bear, y_tip, root_d)
    # one revolved piece: (the disc,) the flare, the shoulder, the journal
    prof = [(0.0, y_beam_end - 0.05)]
    if disc:
        prof += [(bs.disc_r_mm, y_beam_end - 0.05), (bs.disc_r_mm, y_disc)]
    else:
        prof += [(line[0][0], y_beam_end - 0.05)]
    prof += line + [(0.0, y_tip)]
    parts = [_extrude_xz(section(bs), 0.0, y_beam_end + 0.05, xc, zc),
             _revolve_y(prof, x_axle_mm, z_axle_mm)]
    part = trimesh.boolean.union(parts, engine="manifold")
    cap = None
    if bs.hubcap:
        # a plate flush with the wheel's outer face, on a short hub that the
        # stub's end presses into (a 3 mm blind bore)
        hub_r, hub_len = 2.5, 2.0
        cap = _revolve_y([(0.0, y_tip), (STUB_D_MM / 2, y_tip), (STUB_D_MM / 2, y_tip - hub_len),
                          (hub_r, y_tip - hub_len), (hub_r, y_tip), (bs.hubcap_r_mm, y_tip),
                          (bs.hubcap_r_mm, y_out), (0.0, y_out)], x_axle_mm, z_axle_mm)
    return {"support": part, "hubcap": cap, "_section_centre_mm": (xc, zc),
            "_struct": dict(structure(bs, line, y_bear, y_disc, R_mm), root_d_mm=root_d,
                            disc=disc, y_disc_in_rim_mm=bs.disc_recess_mm)}


def gates(bs: BeamSupport, meshes: dict, x_axle_mm: float, z_axle_mm: float, R_mm: float,
          tag: str) -> dict:
    v = meshes["support"].vertices * 1e3
    r = np.hypot(v[:, 0] - x_axle_mm, v[:, 2] - z_axle_mm).max()
    st = meshes["_struct"]
    g = {f"T7.12.1_{tag}_support_in_cylinder": R_mm - r,
         f"T3.7_{tag}_support_clearance": v[:, 2].min() - cc.TRACK_CLEARANCE_MIN,
         f"machining_{tag}_slot_corner_radius": min_curvature_radius_mm(bs) - TOOL_R_MM,
         # loads: margins in MPa and mm
         f"load_{tag}_stub_stress": st["allow_mpa"] - st["stress_mpa"],
         f"load_{tag}_bearing_sag": SAG_MAX_MM - st["sag_mm"]}
    if st["disc"] and bs.disc_recess_mm > 0:      # a disc inside the rim must clear it
        g[f"fit_{tag}_disc_in_rim"] = RIM_INNER_R_MM - 0.5 - bs.disc_r_mm
    if bs.hubcap:
        g[f"fit_{tag}_hubcap_in_rim"] = RIM_INNER_R_MM - 0.5 - bs.hubcap_r_mm
        g[f"T3.7_{tag}_hubcap_clearance"] = z_axle_mm - bs.hubcap_r_mm - cc.TRACK_CLEARANCE_MIN
    return g


def mass_kg(bs: BeamSupport, meshes: dict) -> float:
    """Full part (both halves), with its two hubcaps."""
    vol = abs(meshes["support"].volume) + (abs(meshes["hubcap"].volume) if meshes.get("hubcap") else 0.0)
    return 2 * vol * 1e6 * bs.material_g_cm3 * 1e-3
