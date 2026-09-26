"""
support.py -- parametric streamlined wheel support (T7.12).

The v2 CAD supports are 1.4-1.5 g pods that fill most of the axle cylinder
from the centreline out to the wheel: about 6 g for four, and 7 % of the car's
drag in CFD (2026-09-26). T7.12.1 lets a support take any shape inside "the
cylindrical volume generated through the diameter of the two opposing wheels",
so this is the minimum one: a symmetric NACA strut on the axle line, from 2 mm
inside the body's side to just short of the wheel's inner face, plus an axle
pin through the wheel (inside the closed CFD wheel, so mass only).

Wheels cannot be faired: T7.9 bans parts over the wheels in top and bottom
view, T7.10 in side view (supports excepted), T7.11 in front view above 20 mm.
The support is the one wheel-area part the rules leave free.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import component_contract as cc
import wings as wg


@dataclass(frozen=True)
class Strut:
    chord_mm: float = 16.0
    t_frac: float = 0.25          # 4 mm at 16 mm: stiff for the T7.13 100 g hang test
    embed_mm: float = 2.0         # into the body side
    gap_to_wheel_mm: float = 0.3  # running clearance to the wheel's inner face
    pin_r_mm: float = 1.5         # axle pin through the wheel
    material: str = "PLA"


def body_side_y(body_half_mesh, x_mm: float, z_mm: float, y_from_mm: float) -> float:
    """Body's outer y at (x, z), by a ray from the wheel toward the centreline."""
    o = np.array([[x_mm, y_from_mm, z_mm]]) / 1e3
    locs, _, _ = body_half_mesh.ray.intersects_location(o, np.array([[0, -1.0, 0]]))
    if not len(locs):
        raise ValueError(f"no body at x={x_mm:.1f}, z={z_mm:.1f} to carry the support")
    return float(locs[:, 1].max() * 1e3)


def build_strut(s: Strut, x_axle_mm: float, z_axle_mm: float, y_inner_mm: float,
                wheel_width_mm: float, body_half_mesh) -> dict:
    """Right-half strut (CFD) and pin (mass only), metres."""
    import trimesh
    yb = body_side_y(body_half_mesh, x_axle_mm, z_axle_mm, y_inner_mm - 0.01)
    y0, y1 = yb - s.embed_mm, y_inner_mm - s.gap_to_wheel_mm
    if yb >= y1:
        raise ValueError(f"body side y={yb:.2f} mm reaches the wheel (inner face "
                         f"{y_inner_mm:.2f} mm): no room for a support")
    strut = wg.extrude_section(wg.naca4(s.chord_mm, s.t_frac), y0, y1,
                               x_axle_mm - s.chord_mm / 2, z_axle_mm)
    pin = trimesh.creation.cylinder(radius=s.pin_r_mm / 1e3,
                                    height=(wheel_width_mm + s.gap_to_wheel_mm) / 1e3)
    pin.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    pin.apply_translation([x_axle_mm / 1e3,
                           (y1 + (wheel_width_mm + s.gap_to_wheel_mm) / 2) / 1e3,
                           z_axle_mm / 1e3])
    return {"strut": strut, "pin": pin, "_span_mm": y1 - yb}


def strut_gates(s: Strut, meshes: dict, x_axle_mm: float, z_axle_mm: float,
                R_mm: float, tag: str) -> dict:
    v = meshes["strut"].vertices * 1e3
    r = np.hypot(v[:, 0] - x_axle_mm, v[:, 2] - z_axle_mm).max()
    return {f"T7.12.1_{tag}_support_in_cylinder": R_mm - r,
            f"T3.7_{tag}_support_clearance": v[:, 2].min() - cc.TRACK_CLEARANCE_MIN,
            f"{tag}_support_span": meshes["_span_mm"]}


def strut_mass_kg(s: Strut, meshes: dict) -> float:
    """One support: body-side embed included (it replaces foam, counted twice
    -- a small conservative error), pin included."""
    rho = cc.DENSITY_G_CM3[s.material] * 1e3                    # kg/m3
    return (abs(meshes["strut"].volume) + abs(meshes["pin"].volume)) * rho
