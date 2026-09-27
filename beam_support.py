"""
beam_support.py -- the team's wheel-support architecture, made parametric.

The v2 CAD support (sliced 2026-09-27) is one printed part per axle: a beam
through the body from wheel to wheel, a stub axle at each end that the
3x6x2.5 bearings press onto, and -- on the rear only -- a disc covering the
wheel's inner face. This module builds that part from numbers:

  beam    superellipse section (x width, z height, exponent p) through the
          body along y; it slides into a slot machined from the left and
          right, so the section's tightest curvature must admit the 6.25 mm
          ball (radius >= 3.125 mm)
  disc    optional, front and rear separately: radius, thickness, gap to the
          wheel's inner face
  stub    3 mm (bearing bore), through the wheel's bearings, same material

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
STUB_D_MM = 3.0


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
    disc_front: bool = False
    disc_rear: bool = True        # the v2 CAD has a rear disc only
    disc_r_mm: float = 12.0       # <= axle height - 1.5 mm (T3.7): 12.25 here
    disc_t_mm: float = 1.0
    gap_mm: float = 0.3           # running clearance to the wheel's inner face
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


def build(bs: BeamSupport, x_axle_mm: float, z_axle_mm: float, y_inner_mm: float,
          wheel_width_mm: float, disc: bool) -> dict:
    """Right half of one axle's support: beam + optional disc + stub, metres."""
    import trimesh
    y_face = y_inner_mm - bs.gap_mm                       # support stops here
    y_beam_end = y_face - (bs.disc_t_mm if disc else 0.0)
    xc, zc = x_axle_mm + bs.x_offset_mm, z_axle_mm + bs.z_offset_mm
    parts = [_extrude_xz(section(bs), 0.0, y_beam_end + 0.05, xc, zc)]
    if disc:
        d = trimesh.creation.cylinder(radius=bs.disc_r_mm / 1e3, height=bs.disc_t_mm / 1e3,
                                      sections=96)
        d.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        d.apply_translation([x_axle_mm / 1e3, (y_face - bs.disc_t_mm / 2) / 1e3, z_axle_mm / 1e3])
        parts.append(d)
    stub_len = wheel_width_mm + bs.gap_mm + 0.5
    st = trimesh.creation.cylinder(radius=STUB_D_MM / 2e3, height=stub_len / 1e3, sections=48)
    st.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    st.apply_translation([x_axle_mm / 1e3, (y_face + stub_len / 2 - 0.05) / 1e3, z_axle_mm / 1e3])
    parts.append(st)
    part = trimesh.boolean.union(parts, engine="manifold")
    return {"support": part, "_section_centre_mm": (xc, zc)}


def gates(bs: BeamSupport, meshes: dict, x_axle_mm: float, z_axle_mm: float, R_mm: float,
          tag: str) -> dict:
    v = meshes["support"].vertices * 1e3
    r = np.hypot(v[:, 0] - x_axle_mm, v[:, 2] - z_axle_mm).max()
    return {f"T7.12.1_{tag}_support_in_cylinder": R_mm - r,
            f"T3.7_{tag}_support_clearance": v[:, 2].min() - cc.TRACK_CLEARANCE_MIN,
            f"machining_{tag}_slot_corner_radius": min_curvature_radius_mm(bs) - TOOL_R_MM}


def mass_kg(bs: BeamSupport, meshes: dict) -> float:
    """Full part (both halves)."""
    return 2 * abs(meshes["support"].volume) * 1e6 * bs.material_g_cm3 * 1e-3
