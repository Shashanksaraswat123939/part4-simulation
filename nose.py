"""
nose.py -- parametric printed nose cone ahead of the milled body.

The level-set body is carved from Ref A backwards and its mass term removes
everything forward of it (the "nose carved to zero mass" warning): the milled
body ends in a blunt rounded front at x ~ 31.5 mm. This part is the printed
cone that fairs that front forward of Ref A and carries the front wing.

Shape: superellipse sections |y/b|^p + |(z - zc)/h|^p = 1, from a point tip at
x_tip = Ref A - length to a root at `blend_x`, where it takes the body's own
section (measured by slicing the body STL) scaled by `root_scale` so the cone
tucks just inside the body. Along the cone b, h grow as xi^k (k = 0.5 an
elliptic nose, larger k sharper) and the centre line runs from tip_z to the
body section's centre.

Regulations (component_contract): T8.2 whole nose assembly <= 40 mm ahead of
Ref A; T8.5.1 anything ahead of Ref A <= 25 mm high and <= 15 mm half-width;
T3.7 >= 1.5 mm above the track.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import component_contract as cc


@dataclass(frozen=True)
class NoseCone:
    length_mm: float = 20.0       # tip ahead of Ref A
    k: float = 0.5                # section growth xi^k: 0.5 elliptic, 1 conical
    p: float = 2.5                # superellipse exponent: 2 ellipse, larger boxier
    tip_z_mm: float = 10.0        # tip height above the track
    blend_after_ref_a_mm: float = 10.0   # root station, where the body is full size
    root_scale: float = 0.97      # root section vs the body's own
    wall_mm: float = 0.8          # printed shell, for mass
    material: str = "PLA"


def body_section(body_half_mesh, x_mm: float) -> tuple:
    """(half-width, z_low, z_high) of the body at station x, mm."""
    s = body_half_mesh.section(plane_origin=[x_mm / 1e3, 0, 0], plane_normal=[1, 0, 0])
    if s is None:
        raise ValueError(f"the body has no section at x = {x_mm:.1f} mm")
    v = s.vertices * 1e3
    return float(v[:, 1].max()), float(v[:, 2].min()), float(v[:, 2].max())


def _ring(b, h, zc, p, m):
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.c_[b * np.sign(c) * np.abs(c) ** (2 / p), zc + h * np.sign(s) * np.abs(s) ** (2 / p)]


def build_nose(nc: NoseCone, x_front_mm: float, body_half_mesh, n: int = 40, m: int = 48):
    """Right half (y >= 0) of the cone, metres, plus the full-cone mesh for mass."""
    import trimesh
    ref_a = cc.ref_plane_A(x_front_mm)
    x_tip = ref_a - nc.length_mm
    x_root = ref_a + nc.blend_after_ref_a_mm
    b0, zl, zh = body_section(body_half_mesh, x_root)
    b0, h0, zc0 = nc.root_scale * b0, nc.root_scale * (zh - zl) / 2, (zh + zl) / 2
    xi = (1 - np.cos(np.linspace(0, math.pi / 2, n + 1)[1:]))        # clustered at the tip
    xi = xi / xi[-1]
    verts = [[x_tip, 0.0, nc.tip_z_mm]]
    for q in xi:
        f = q ** nc.k
        ring = _ring(b0 * f, h0 * f, nc.tip_z_mm + (zc0 - nc.tip_z_mm) * q, nc.p, m)
        verts += [[x_tip + q * (x_root - x_tip), y, z] for y, z in ring]
    verts.append([x_root, 0.0, zc0])
    V = np.array(verts) / 1e3
    F = [[0, 1 + (j + 1) % m, 1 + j] for j in range(m)]
    for i in range(n - 1):
        a, b = 1 + i * m, 1 + (i + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            F += [[a + j, a + j1, b + j1], [a + j, b + j1, b + j]]
    last, c = 1 + (n - 1) * m, len(V) - 1
    F += [[c, last + j, last + (j + 1) % m] for j in range(m)]
    full = trimesh.Trimesh(V, np.array(F), process=True)
    if full.volume < 0:
        full.invert()
    half = trimesh.intersections.slice_mesh_plane(full, [0, 1, 0], [0, 0, 0], cap=True)
    return {"nose": half, "_full": full, "_root": (b0, zl, zh)}


def nose_mass_com(nc: NoseCone, full_mesh):
    """Printed shell: side area (root cap excluded) x wall. kg, COM in m."""
    side = full_mesh.face_normals[:, 0] < 0.99
    area = float(full_mesh.area_faces[side].sum())                   # m^2
    m = area * nc.wall_mm / 1e3 * cc.DENSITY_G_CM3[nc.material] * 1e3  # kg
    c = full_mesh.triangles_center[side]
    w = full_mesh.area_faces[side]
    com = tuple((c * w[:, None]).sum(0) / w.sum())
    return m, (com[0], 0.0, com[2])


def nose_gates(nc: NoseCone, meshes: dict, x_front_mm: float) -> dict:
    ref_a = cc.ref_plane_A(x_front_mm)
    v = meshes["nose"].vertices * 1e3
    ahead = v[v[:, 0] < ref_a]
    return {
        "T8.2_nose_cone_overhang": cc.NOSE_OVERHANG_MAX - (ref_a - v[:, 0].min()),
        "T8.5.1_nose_height": cc.NOSE_SUPPORT_Z_MAX - ahead[:, 2].max(),
        "T8.5.1_nose_half_width": cc.NOSE_SUPPORT_HALF_WIDTH_MAX - ahead[:, 1].max(),
        "T3.7_nose_clearance": v[:, 2].min() - cc.TRACK_CLEARANCE_MIN,
    }
