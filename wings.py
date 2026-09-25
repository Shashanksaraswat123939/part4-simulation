"""
wings.py -- parametric front and rear wings, their supports, and scrutineer gates.

Wings are exact geometry (NACA 4-digit section extruded along y), not level-set
material: a 2-3 mm wing is 4-6 cells on the body's 0.5 mm grid and marching
cubes turns it into a lumpy slab. Every builder returns the RIGHT half
(y >= 0) in metres, which is what the half-car CFD takes; the car is symmetric.

What the numbers are based on (rnd-cfd, 2026-09-25, medium mesh, rotating
wheels): a flat front wing 5.9 mm ahead of the front wheels cut wheel drag
6.5 %, exactly its own drag -- net zero. Endplates as tried added 3-6 %. So the
defaults are a flat, endplate-free front wing and a minimum-drag rear wing;
the shape parameters are exposed for the optimiser (Part 5).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

import component_contract as cc


# ------------------------------------------------------------------ geometry

def naca4(chord_mm: float, t_frac: float, camber: float = 0.0, camber_pos: float = 0.4,
          n: int = 41) -> np.ndarray:
    """Closed NACA 4-digit section, (x, z) in mm, x from 0 (LE) to chord (TE)."""
    beta = np.linspace(0, math.pi, n)
    x = 0.5 * (1 - np.cos(beta))                       # cosine spacing
    yt = 5 * t_frac * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                       + 0.2843 * x**3 - 0.1036 * x**4)
    if camber > 0:
        p = camber_pos
        yc = np.where(x < p, camber / p**2 * (2 * p * x - x**2),
                      camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    else:
        yc = np.zeros_like(x)
    upper = np.c_[x, yc + yt]
    lower = np.c_[x, yc - yt][::-1][1:-1]
    return np.vstack([upper, lower]) * chord_mm


def extrude_section(section_xz_mm: np.ndarray, y0_mm: float, y1_mm: float,
                    x_le_mm: float, z_chord_mm: float, aoa_deg: float = 0.0):
    """Extrude a section along +y. Positive AoA raises the TRAILING edge
    (nose-down, i.e. a downforce incidence for an inverted-free flat wing)."""
    import trimesh
    from shapely.geometry import Polygon
    pts = section_xz_mm.copy()
    if aoa_deg:
        a = math.radians(aoa_deg)
        c = np.array([0.25 * (pts[:, 0].max()), 0.0])
        rel = pts - c
        pts = c + rel @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    poly = Polygon(pts).buffer(0)
    m = trimesh.creation.extrude_polygon(poly, (y1_mm - y0_mm) / 1000.0)
    m.apply_scale([1e-3, 1e-3, 1.0])                   # section in mm -> m
    # section plane (x, z_sec), extrusion along +z  ->  car (x, z), along +y
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    m.apply_translation([x_le_mm / 1000.0, y0_mm / 1000.0, z_chord_mm / 1000.0])
    return m


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box in mm -> trimesh in m."""
    import trimesh
    b = trimesh.creation.box(extents=[(x1 - x0) / 1e3, (y1 - y0) / 1e3, (z1 - z0) / 1e3])
    b.apply_translation([(x0 + x1) / 2e3, (y0 + y1) / 2e3, (z0 + z1) / 2e3])
    return b


# ------------------------------------------------------------------ front wing

@dataclass(frozen=True)
class FrontWing:
    chord_mm: float = 20.0
    t_frac: float = 0.15          # 3.0 mm at 20 mm chord
    camber: float = 0.0
    aoa_deg: float = 0.0
    half_span_mm: float = 38.5    # tip just outboard of the front wheel (36.5)
    z_chord_mm: float = 8.0       # chord line height above the track
    gap_to_wheel_mm: float = 5.5  # TE to front-wheel leading edge (T7.9.1 >= 5)
    endplate_h_mm: float = 0.0    # 0 = none (measured to hurt, see module doc)
    endplate_w_mm: float = 2.0
    mount_t_mm: float = 3.0       # centreline mount to the body at Ref A

    def x_le(self, x_front_mm: float, R_mm: float) -> float:
        return x_front_mm - R_mm - self.gap_to_wheel_mm - self.chord_mm


def build_front_wing(fw: FrontWing, x_front_mm: float, R_mm: float) -> dict:
    """{'wing': mesh, 'mount': mesh[, 'endplate': mesh]} right halves, metres."""
    sec = naca4(fw.chord_mm, fw.t_frac, fw.camber)
    x_le = fw.x_le(x_front_mm, R_mm)
    out = {"wing": extrude_section(sec, 0.0, fw.half_span_mm, x_le, fw.z_chord_mm, fw.aoa_deg)}
    ref_a = cc.ref_plane_A(x_front_mm)
    t = fw.chord_mm * fw.t_frac
    # Mount: from 30 % chord back to 1 mm past Ref A (so it bonds to the body),
    # half-thickness in y, from inside the wing to 4 mm above it.
    out["mount"] = box(x_le + 0.3 * fw.chord_mm, ref_a + 1.0, 0.0, fw.mount_t_mm / 2,
                       fw.z_chord_mm - 0.25 * t, fw.z_chord_mm + t / 2 + 4.0)
    if fw.endplate_h_mm > 0:
        out["endplate"] = box(x_le, x_le + fw.chord_mm, fw.half_span_mm,
                              fw.half_span_mm + fw.endplate_w_mm,
                              fw.z_chord_mm - fw.endplate_h_mm / 2,
                              fw.z_chord_mm + fw.endplate_h_mm / 2)
    return out


def front_wing_gates(fw: FrontWing, meshes: dict, x_front_mm: float, R_mm: float,
                     nose_half_width_mm: float = 0.0) -> dict:
    """Signed margins in mm (>= 0 passes), measured on the meshes."""
    wing = meshes["wing"]
    b = wing.bounds * 1e3
    ref_a = cc.ref_plane_A(x_front_mm)
    allb = np.vstack([m.bounds for m in meshes.values()]) * 1e3
    t = fw.chord_mm * fw.t_frac
    # Span: continuous across the centreline unless a nose wider than the
    # mount splits it; then two segments of (half_span - nose_half_width).
    seg = fw.half_span_mm - max(nose_half_width_mm, fw.mount_t_mm / 2)
    span_margin = (2 * fw.half_span_mm - cc.FRONT_SPAN_SINGLE_MIN
                   if nose_half_width_mm <= fw.mount_t_mm / 2
                   else seg - cc.FRONT_SPAN_SEGMENT_MIN)
    outboard_top = b[1, 2]           # whole wing is outboard of 15 mm somewhere
    endplate = meshes.get("endplate")
    g = {
        "T8.6.1_span": span_margin,
        "T8.6.2_chord_min": fw.chord_mm - cc.FRONT_CHORD_MIN,
        "T8.6.2_chord_max": cc.FRONT_CHORD_MAX - fw.chord_mm,
        "T8.6.3_thick_min": t - cc.WING_THICK_MIN,
        "T8.6.3_thick_max": cc.WING_THICK_MAX - t,
        "T8.5.2_forward_of_ref_A": ref_a - b[1, 0],
        "T8.5.2_height_outboard": cc.FRONT_WING_Z_MAX_OUTBOARD - outboard_top,
        "T8.2_nose_overhang": cc.NOSE_OVERHANG_MAX - (ref_a - allb[:, 0].min()),
        "T8.7_clear_air_track": b[0, 2] - cc.CLEAR_AIR,
        "T8.7_clear_air_front_wheel": (x_front_mm - R_mm) - b[1, 0] - cc.CLEAR_AIR,
        "T7.9.1_zone_ahead_of_wheel": (x_front_mm - R_mm) - b[1, 0] - cc.T79_AHEAD_OF_FRONT_WHEEL,
        "T8.5.1_mount_height": cc.NOSE_SUPPORT_Z_MAX - meshes["mount"].bounds[1, 2] * 1e3,
        "T8.5.1_mount_half_width": cc.NOSE_SUPPORT_HALF_WIDTH_MAX - meshes["mount"].bounds[1, 1] * 1e3,
    }
    if endplate is not None:
        eb = endplate.bounds * 1e3
        g["T8.5.3_endplate_width"] = cc.FRONT_ENDPLATE_WIDTH_MAX - (eb[1, 1] - eb[0, 1])
        g["T8.5.3_endplate_height"] = cc.FRONT_ENDPLATE_Z_MAX - eb[1, 2]
    return g


# ------------------------------------------------------------------ rear wing

@dataclass(frozen=True)
class RearWing:
    chord_mm: float = 16.0
    t_frac: float = 0.15          # 2.4 mm
    camber: float = 0.0
    aoa_deg: float = 0.0
    half_span_mm: float = 26.0    # 52 mm unbroken span (T9.5.1 >= 50)
    x_le_after_ref_b_mm: float = 3.0
    clear_air_margin_mm: float = 0.5   # above the 5 mm T9.6 minimum
    pylon_t_mm: float = 2.0
    pylon_chord_mm: float = 10.0


def body_top_under(body_half_mesh, x0_mm, x1_mm, y1_mm, step_mm: float = 1.0) -> float:
    """Highest body point under a footprint (mm), by vertical ray casts."""
    xs = np.arange(x0_mm, x1_mm + 1e-9, step_mm)
    ys = np.arange(0.0, y1_mm + 1e-9, step_mm)
    X, Y = np.meshgrid(xs, ys)
    origins = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 200.0)] / 1e3
    dirs = np.tile([0, 0, -1.0], (len(origins), 1))
    locs, idx, _ = body_half_mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    return float(locs[:, 2].max() * 1e3) if len(locs) else 0.0


def build_rear_wing(rw: RearWing, x_front_mm: float, W_mm: float, body_half_mesh) -> dict:
    """Wing placed with T9.6 clear air above the body; pylon down to the body."""
    ref_b = cc.ref_plane_B(x_front_mm, W_mm)
    x_le = ref_b + rw.x_le_after_ref_b_mm
    t = rw.chord_mm * rw.t_frac
    top = body_top_under(body_half_mesh, x_le - 1, x_le + rw.chord_mm + 1, rw.half_span_mm + 1)
    z_chord = top + cc.CLEAR_AIR + rw.clear_air_margin_mm + t / 2 + 0.5
    sec = naca4(rw.chord_mm, rw.t_frac, rw.camber)
    wing = extrude_section(sec, 0.0, rw.half_span_mm, x_le, z_chord, rw.aoa_deg)
    xm = x_le + 0.5 * rw.chord_mm
    pylon = box(xm - rw.pylon_chord_mm / 2, xm + rw.pylon_chord_mm / 2, 0.0, rw.pylon_t_mm / 2,
                top - 2.0, z_chord)
    return {"wing": wing, "pylon": pylon, "_body_top_mm": top, "_z_chord_mm": z_chord}


def rear_wing_gates(rw: RearWing, meshes: dict, x_front_mm: float, W_mm: float,
                    body_half_mesh, forward_parts=()) -> dict:
    wing = meshes["wing"]
    b = wing.bounds * 1e3
    ref_b = cc.ref_plane_B(x_front_mm, W_mm)
    t = rw.chord_mm * rw.t_frac
    # T9.6 clear air: sample the wing's lower surface, distance to the body.
    pts, _ = __import__("trimesh").sample.sample_surface_even(wing, 800, seed=1)
    lower = pts[pts[:, 2] < wing.bounds[:, 2].mean()]
    import trimesh
    _, d, _ = trimesh.proximity.closest_point(body_half_mesh, lower if len(lower) else pts)
    # T9.7 front view: nothing ahead of the wing may reach above its underside
    # within its span.
    block = 0.0
    for m in forward_parts:
        mb = m.bounds * 1e3
        if mb[0, 1] < b[1, 1] and mb[0, 0] < b[0, 0]:
            block = max(block, mb[1, 2])
    pb = meshes["pylon"].bounds * 1e3
    return {
        "T9.4.1_aft_of_ref_B": min(b[0, 0], pb[0, 0]) - ref_b,
        "T9.4.2_overhang": cc.REAR_OVERHANG_MAX - (b[1, 0] - ref_b),
        "T9.4.3_height": cc.REAR_Z_MAX - b[1, 2],
        "T9.5.1_span": 2 * rw.half_span_mm - cc.REAR_SPAN_MIN,
        "T9.5.2_chord_min": rw.chord_mm - cc.REAR_CHORD_MIN,
        "T9.5.2_chord_max": cc.REAR_CHORD_MAX - rw.chord_mm,
        "T9.5.3_thick_min": t - cc.WING_THICK_MIN,
        "T9.5.3_thick_max": cc.WING_THICK_MAX - t,
        "T9.5.4_height_deviation": cc.REAR_HEIGHT_DEVIATION_MAX - 0.0,   # straight wing
        "T9.6_clear_air_body": float(d.min() * 1e3) - cc.CLEAR_AIR,
        "T9.7_front_view": b[0, 2] - block,
    }


def params_dict(obj) -> dict:
    return asdict(obj)
