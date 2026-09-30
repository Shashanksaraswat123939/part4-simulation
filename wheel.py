"""
wheel.py -- the team's wheels: their STL geometry in SLS PA12, with a 6 mm seat
for one 3x6x2.5 mm bearing, measured for mass and spin inertia; and the closed
spinning disc the CFD uses in their place.

The wheel geometry is the team's (part1 hardware_cad/front_wheel.stl,
rear_wheel.stl), not optimised here. The CFD cannot resolve 0.6 mm spokes on a
~1 mm mesh, so it models each wheel as a closed disc of the same size with a
rotating-wall boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 3x6x2.5 mm miniature bearing (MR63 size): ~0.30 g, steel.
BEARING_ID_MM, BEARING_OD_MM, BEARING_W_MM, BEARING_G = 3.0, 6.0, 2.5, 0.30
PA12_G_CM3 = 1.01
CAD = Path(__file__).resolve().parent.parent / "part1-simulation" / "hardware_cad"


@dataclass(frozen=True)
class MeasuredWheel:
    """A wheel measured from its mesh (x material density), plus its bearings."""
    R: float                   # mm
    w: float                   # contact width, mm
    mass_part_g: float
    inertia_part_gmm2: float
    bearings: int = 1

    @property
    def total_width(self) -> float:
        return self.w

    @property
    def mass(self) -> float:
        return self.mass_part_g + self.bearings * BEARING_G

    @property
    def inertia(self) -> float:
        # the outer race and half the balls turn with the wheel
        m_b = self.bearings * BEARING_G
        return self.inertia_part_gmm2 + 0.45 * m_b * (BEARING_OD_MM / 2 - 0.4) ** 2


def _seat_ring(m, ax):
    """The 6 mm bearing seat added to the team hub: a PA12 ring from the
    bearing OD out to the STL's own bore, over the hub's own length."""
    import trimesh
    c = (m.bounds[0] + m.bounds[1]) / 2
    v = m.vertices - c
    r = np.hypot(*np.delete(v, ax, axis=1).T)
    hub = r < 5.5
    bore_r = float(r[hub].min())
    a0, a1 = float(v[hub, ax].min()), float(v[hub, ax].max())
    ring = trimesh.creation.annulus(r_min=BEARING_OD_MM / 2, r_max=bore_r + 0.05,
                                    height=a1 - a0, sections=96)
    if ax != 2:
        ring.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)[ax]))
    shift = c.copy()
    shift[ax] += (a0 + a1) / 2
    ring.apply_translation(shift)
    return ring, bore_r, a1 - a0


def team_wheel_meshes() -> list:
    """(front, rear): (mesh with the bearing seat, axis index, bore radius,
    hub length), in the STL's own coordinates (mm)."""
    import trimesh
    out = []
    for name in ("front_wheel.stl", "rear_wheel.stl"):
        m = trimesh.load(str(CAD / name), force="mesh")
        trimesh.repair.fix_winding(m)
        trimesh.repair.fix_normals(m)
        ax = int(np.argmin(m.bounds[1] - m.bounds[0]))
        ring, bore_r, hub_len = _seat_ring(m, ax)
        out.append((trimesh.boolean.union([m, ring], engine="manifold"), ax, bore_r, hub_len))
    return out


def design(name: str = "team_stl") -> tuple:
    """(front, rear) MeasuredWheel. The hub is 3 mm long: ONE 2.5 mm bearing."""
    if name != "team_stl":
        raise ValueError(f"unknown wheel design {name!r}; the wheels are the team's STLs")
    out = []
    rho = PA12_G_CM3 * 1e-3                       # g/mm3
    for seated, ax, _bore, _len in team_wheel_meshes():
        ext = seated.bounds[1] - seated.bounds[0]
        mc = seated.copy()
        mc.apply_translation(-(seated.bounds[0] + seated.bounds[1]) / 2)
        out.append(MeasuredWheel(R=float(max(np.delete(ext, ax)) / 2), w=float(ext[ax]),
                                 mass_part_g=abs(seated.volume) * rho,
                                 inertia_part_gmm2=abs(mc.moment_inertia[ax, ax]) * rho))
    return tuple(out)


def mean_inertia_kg_m2(name: str = "team_stl") -> float:
    """The single wheel MOI the race objective takes (it multiplies by 4)."""
    f, r = design(name)
    return 0.5 * (f.inertia + r.inertia) * 1e-9


def summary(name: str = "team_stl") -> dict:
    f, r = design(name)
    return {"design": name,
            "front": {"R_mm": f.R, "width_mm": f.w, "mass_g": f.mass, "I_gmm2": f.inertia},
            "rear": {"R_mm": r.R, "width_mm": r.w, "mass_g": r.mass, "I_gmm2": r.inertia},
            "mean_I_kg_m2": mean_inertia_kg_m2(name)}


def cfd_surface(R_mm: float, width_mm: float, x_axle_mm: float, y_inner_mm: float,
                sink_mm: float = 0.3, sections: int = 128):
    """Closed disc for CFD, sunk into the track so the contact line meshes (a
    tangent cylinder gives snappy a zero-thickness gap). Metres."""
    import trimesh
    m = trimesh.creation.cylinder(radius=R_mm / 1e3, height=width_mm / 1e3, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([x_axle_mm / 1e3, (y_inner_mm + width_mm / 2) / 1e3,
                         (R_mm - sink_mm) / 1e3])
    return m
