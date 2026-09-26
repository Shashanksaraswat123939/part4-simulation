"""
wheel.py -- parametric STEM Racing wheel: mass, rotational inertia, relative stiffness.

Geometry (matches the team's v2 CAD wheel family, measured in cad_wheels.json):
  rim   : cylindrical shell, outer radius R, radial thickness t_rim, axial width w
          (the whole contact width, T7.4/T7.7), optional inboard lip
  plate : a central spoke plate of axial thickness t_plate spanning r_hub..R-t_rim,
          with N spokes of tangential width b_spoke (straight-spoke approximation)
  hub   : ring r_bore..r_hub over axial length l_hub
  cap   : optional full-face disc of thickness t_cap (hubcap)

All lengths mm, density g/cm^3, results in g and g*mm^2.

Stiffness is RELATIVE to the CAD wheel, which passes the T7.13 100 g hang test.
Two failure modes are tracked, both from thin-shell/beam scaling:
  S_span  rim bending between spokes, a curved beam of span L = 2*pi*R/N:
          stiffness ~ E * w * t^3 / L^3
  S_edge  the unsupported rim overhang either side of the plate, a cylindrical
          shell cantilever of length a = (w - t_plate)/2 under a radial edge load:
          stiffness ~ E * t^3 / a^2 * sqrt(t/R)^-1 ... simplified to E * t^2.5 / (a^2 * R^0.5)
          (shell bending length sqrt(R t) sets the loaded zone)
A design is acceptable when both ratios to the CAD wheel are >= 1. The model
is a scaling law, not an FE result: it ranks designs, and the physical 100 g
hang test on printed samples is the acceptance test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

MATERIALS = {
    # name: (density g/cm3, E GPa, min printable wall mm, note)
    "ABS_FDM":       (1.04, 2.2, 0.40, "current wheels; 0.4 mm nozzle"),
    "PLA_FDM":       (1.24, 3.5, 0.40, "stiffer, denser"),
    "PETG_FDM":      (1.27, 2.1, 0.40, ""),
    "SLA_standard":  (1.15, 2.6, 0.30, "thin walls, brittle"),
    "SLA_tough":     (1.14, 1.9, 0.30, ""),
    "PA12_SLS":      (1.01, 1.7, 0.60, "SLS nylon"),
    "CF_tube_rim":   (1.55, 50.0, 0.15, "rolled carbon tube as the rim only"),
    "PET_film":      (1.39, 4.0, 0.05, "thermoformed film face cover, not structural"),
}


@dataclass(frozen=True)
class Wheel:
    R: float = 14.12
    w: float = 13.25
    t_rim: float = 0.40
    t_plate: float = 2.93
    r_hub: float = 5.0
    r_bore: float = 4.5
    l_hub: float = 2.93
    n_spokes: int = 7
    b_spoke: float = 1.29         # tangential spoke width mid-span (mm), CAD-calibrated
    r_fillet: float = 1.5         # root fillet radius at hub and rim ends (mm)
    plate_fill_extra: float = 0.0 # extra plate area fraction (fillets, webs)
    t_cap: float = 0.0
    rim_material: str = "ABS_FDM"
    body_material: str = "ABS_FDM"
    cap_material: str = ""        # "" = body_material
    # Aero shape, beyond the flat contact width w (so T7.4 is untouched):
    # rounded tyre shoulders of radius s (a printed quarter-torus shell each)
    # and an outboard hubcap bulging `dome` mm (a paraboloid shell).
    shoulder_in: float = 0.0
    shoulder_out: float = 0.0
    dome: float = 0.0

    @property
    def total_width(self):
        return self.w + self.shoulder_in + self.shoulder_out + self.dome

    # ---- mass properties -------------------------------------------------
    def parts(self):
        rho_r = MATERIALS[self.rim_material][0] * 1e-3      # g/mm3
        rho_b = MATERIALS[self.body_material][0] * 1e-3
        R, ri = self.R, self.R - self.t_rim
        out = {}
        v = math.pi * (R**2 - ri**2) * self.w
        out["rim"] = (rho_r * v, rho_r * v * (R**2 + ri**2) / 2)
        # spokes: N spokes of width b plus root fillets of radius rf at both
        # ends (width grows by 2*rf*(1-d/rf)^2 within rf of each root).
        # Integrated numerically over r in the plate of thickness t_plate.
        n = 400
        rr = [self.r_hub + (ri - self.r_hub) * (k + 0.5) / n for k in range(n)]
        dr = (ri - self.r_hub) / n
        m_sp = i_sp = 0.0
        for r in rr:
            b = self.b_spoke
            for d in (r - self.r_hub, ri - r):
                if d < self.r_fillet:
                    b += 2 * self.r_fillet * (1 - d / self.r_fillet) ** 2
            b = min(b, 2 * math.pi * r / max(self.n_spokes, 1))
            dm = rho_b * self.n_spokes * b * self.t_plate * dr
            m_sp += dm
            i_sp += dm * r * r
        out["spokes"] = (m_sp, i_sp)
        if self.plate_fill_extra:
            v2 = math.pi * (ri**2 - self.r_hub**2) * self.t_plate * self.plate_fill_extra
            out["plate_fill"] = (rho_b * v2, rho_b * v2 * (ri**2 + self.r_hub**2) / 2)
        v = math.pi * (self.r_hub**2 - self.r_bore**2) * self.l_hub
        out["hub"] = (rho_b * v, rho_b * v * (self.r_hub**2 + self.r_bore**2) / 2)
        if self.t_cap:
            rho_c = MATERIALS[self.cap_material or self.body_material][0] * 1e-3
            v = math.pi * (ri**2 - self.r_hub**2) * self.t_cap
            out["cap"] = (rho_c * v, rho_c * v * (ri**2 + self.r_hub**2) / 2)
        t_sh = MATERIALS[self.body_material][2]            # thinnest printable shell
        for k in ("shoulder_in", "shoulder_out"):
            s_ = getattr(self, k)
            if s_:
                rc = R - s_ * (1 - 2 / math.pi)            # centroid radius of the arc
                v = (math.pi / 2 * s_) * 2 * math.pi * rc * t_sh
                out[k] = (rho_b * v, rho_b * v * rc * rc)
        if self.dome:
            a = R - self.shoulder_out
            v = math.pi * self.dome**2 * t_sh               # extra area of the cap
            out["dome"] = (rho_b * v, rho_b * v * a * a / 2)
        return out

    @property
    def mass(self):
        return sum(m for m, _ in self.parts().values())

    @property
    def inertia(self):
        return sum(i for _, i in self.parts().values())

    # ---- relative stiffness ----------------------------------------------
    def _E_rim(self):
        return MATERIALS[self.rim_material][1]

    def s_span(self):
        L = 2 * math.pi * self.R / self.n_spokes
        return self._E_rim() * self.w * self.t_rim**3 / L**3

    def s_edge(self):
        a = max((self.w - self.t_plate) / 2, 1e-6)
        return self._E_rim() * self.t_rim**2.5 / (a**2 * self.R**0.5)

    def check(self, ref: "Wheel"):
        return self.s_span() / ref.s_span(), self.s_edge() / ref.s_edge()

    def printable(self):
        return (self.t_rim >= MATERIALS[self.rim_material][2] - 1e-9 and
                self.b_spoke >= MATERIALS[self.body_material][2] and
                (self.t_cap == 0 or
                 self.t_cap >= MATERIALS[self.cap_material or self.body_material][2]))


CAD_FRONT = Wheel(w=13.25)
CAD_REAR = Wheel(w=17.25)


# --------------------------------------------------------------------------
# Designs (rnd/wheels, 2026-09-25). Front and rear share the section; widths
# are T7.4 minimum + 0.1 mm print margin (the CAD used +0.25).
# --------------------------------------------------------------------------
_LIGHT = dict(R=14.05, t_rim=0.40, n_spokes=7, b_spoke=0.6, t_plate=2.93, l_hub=2.93,
              r_fillet=0.5)
DESIGNS = {
    # today's printed v2 wheels, open spokes (model within 1.1 % of the CAD I)
    "cad_v2": (CAD_FRONT, CAD_REAR),
    # lightest ABS wheel at >= the CAD wheel's stiffness, open spokes
    "abs_light": (Wheel(w=13.1, **_LIGHT), Wheel(w=17.1, **_LIGHT)),
    # same with 0.4 mm hubcaps on both faces: a CLOSED wheel, which is what a
    # rotating-wall CFD boundary represents honestly
    "abs_light_capped": (Wheel(w=13.1, t_cap=0.8, **_LIGHT),
                         Wheel(w=17.1, t_cap=0.8, **_LIGHT)),
    # 0.20 mm rolled carbon rim, printed SLA spokes/hub, 2.9x the CAD stiffness
    "carbon_rim": (
        Wheel(w=13.1, rim_material="CF_tube_rim", body_material="SLA_standard",
              **dict(_LIGHT, t_rim=0.20, t_plate=1.2, l_hub=1.2)),
        Wheel(w=17.1, rim_material="CF_tube_rim", body_material="SLA_standard",
              **dict(_LIGHT, t_rim=0.20, t_plate=1.2, l_hub=1.2))),
    # + 1 mm outboard hubcap dome: -1.3 to -1.6 % D20 in CFD (3 mm and 4 mm
    # were worse), 0.1 g.mm2 of inertia
    "carbon_rim_capped": (
        Wheel(w=13.1, rim_material="CF_tube_rim", body_material="SLA_standard", t_cap=0.6,
              dome=1.0, **dict(_LIGHT, t_rim=0.20, t_plate=1.2, l_hub=1.2)),
        Wheel(w=17.1, rim_material="CF_tube_rim", body_material="SLA_standard", t_cap=0.6,
              dome=1.0, **dict(_LIGHT, t_rim=0.20, t_plate=1.2, l_hub=1.2))),
}


# The open carbon wheel with its faces closed by 0.10 mm film covers (both
# faces, dome 1 mm outboard): the CFD shape of carbon_rim_capped at close to
# carbon_rim's inertia. The rotating-wall CFD cannot resolve open spokes (0.6 mm
# spokes, 0.2 mm rim against ~1.1 mm cells at medium), so closing them is the
# measured option.
DESIGNS["carbon_rim_film"] = tuple(
    Wheel(w=w, rim_material="CF_tube_rim", body_material="SLA_standard",
          cap_material="PET_film", t_cap=2 * 0.10, dome=1.0,
          **dict(_LIGHT, t_rim=0.20, t_plate=1.2, l_hub=1.2))
    for w in (13.1, 17.1))


def design(name):
    """A design name, or a (front, rear) pair of Wheel passed straight through."""
    if isinstance(name, tuple):
        return name
    if name not in DESIGNS:
        raise ValueError(f"unknown wheel design {name!r}; known: {sorted(DESIGNS)}")
    return DESIGNS[name]


def mean_inertia_kg_m2(name: str) -> float:
    """The single wheel MOI the race objective takes (it multiplies by 4)."""
    f, r = design(name)
    return 0.5 * (f.inertia + r.inertia) * 1e-9


def is_closed(name: str) -> bool:
    return all(w.t_cap > 0 for w in design(name))


def summary(name: str) -> dict:
    f, r = design(name)
    sf, se = min(f.check(CAD_FRONT)), min(r.check(CAD_REAR))
    return {"design": name if isinstance(name, str) else "custom", "closed": is_closed(name),
            "front": {"R_mm": f.R, "width_mm": f.w, "mass_g": f.mass, "I_gmm2": f.inertia},
            "rear": {"R_mm": r.R, "width_mm": r.w, "mass_g": r.mass, "I_gmm2": r.inertia},
            "mean_I_kg_m2": mean_inertia_kg_m2(name),
            "stiffness_vs_cad_min": min(sf, se), "printable": f.printable() and r.printable()}


def cfd_surface(R_mm: float, width_mm: float, x_axle_mm: float, y_inner_mm: float,
                sink_mm: float = 0.3, sections: int = 128, shoulder_in: float = 0.0,
                shoulder_out: float = 0.0, dome: float = 0.0):
    """Closed wheel of revolution for CFD, sunk into the track so the contact
    line meshes (a tangent cylinder gives snappy a zero-thickness gap). Metres.

    Profile from the inboard face (y_inner) outward: flat face, shoulder arc,
    flat contact width `width_mm`, shoulder arc, outboard face with a
    paraboloid dome. Zero shoulders and dome give the plain cylinder."""
    import numpy as np
    import trimesh

    def arc(cr, ca, s, t0, t1, n=8):     # centre (radius, axial), angles in rad
        t = np.linspace(t0, t1, n)
        return np.c_[cr + s * np.cos(t), ca + s * np.sin(t)]

    R, w, si, so = R_mm, width_mm, shoulder_in, shoulder_out
    pts = [[0.0, 0.0], [R - si, 0.0]]
    if si:
        pts += arc(R - si, si, si, -np.pi / 2, 0.0)[1:].tolist()
    pts += [[R, si], [R, si + w]]
    a_end = si + w + so
    if so:
        pts += arc(R - so, si + w, so, 0.0, np.pi / 2)[1:].tolist()
    pts.append([R - so, a_end])
    if dome:
        r = np.linspace(R - so, 0.0, 12)[1:]
        pts += np.c_[r, a_end + dome * (1 - (r / (R - so)) ** 2)].tolist()
    else:
        pts.append([0.0, a_end])
    prof = np.array(pts)
    prof = prof[np.r_[True, np.any(np.diff(prof, axis=0) != 0, axis=1)]]
    m = trimesh.creation.revolve(prof / 1000.0, sections=sections)
    if m.volume < 0:
        m.invert()
    # revolve axis z -> car y
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([x_axle_mm / 1000, y_inner_mm / 1000, (R - sink_mm) / 1000])
    return m
