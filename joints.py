"""
joints.py -- printed positives glued into machined negatives.

Team process (2026-09-27): wheel supports, nose cone and every other printed
part are POSITIVES glued ONTO NEGATIVES (pockets) in the main body, and the body
is 3-axis milled with a 6.25 mm ball-end cutter from the top and the bottom.

For each printed part P and the aero body B (right halves, metres):

    O      = P ∩ B                          the part's embedded volume
    fp     = O projected on the xy plane    its footprint
    plug   = prism(fp ⊕ r, from the open face to the far end of O)
    pocket = plug ∩ B                       what the mill removes

fp ⊕ r (a 3.125 mm buffer) is where the ball's CENTRE can go with its edge still
covering fp, so the milled pocket contains the plug; a straight prism to the
open face is what a vertical cutter can reach. The printed part is P ∪ pocket:
it fills the pocket exactly, so the outside shape (and the CFD) is unchanged,
while plastic replaces foam inside. Pockets never cut the T5.5 cartridge wall:
parts are trimmed back from a keep-out around it first.

    cut(parts, body_half, open_side, keep_out) -> JointResult per part
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

TOOL_R_MM = 3.125
FOAM_G_CM3 = 0.163


@dataclass
class Joint:
    name: str
    open_side: str
    plug: object          # trimesh, right half, m
    pocket: object        # plug ∩ body
    printed: object       # (part - body) ∪ pocket
    pocket_cm3: float
    embedded_cm3: float

    def mass_delta_kg(self, part_density_g_cm3: float) -> float:
        """Full-car (both halves) mass change vs 'foam body + whole part',
        which is what the mass rollup counted before joints existed."""
        dv = self.pocket_cm3 * (part_density_g_cm3 - FOAM_G_CM3) \
            - self.embedded_cm3 * part_density_g_cm3
        return 2 * dv * 1e-3


def _bool(op, meshes):
    import trimesh
    return getattr(trimesh.boolean, op)(meshes, engine="manifold")


def _footprint(mesh):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    tri = mesh.vertices[mesh.faces][:, :, :2] * 1e3
    polys = [Polygon(t) for t in tri if abs(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-9]
    return unary_union(polys)


def _prism(poly, z0_mm, z1_mm):
    import trimesh
    from shapely.geometry import box
    poly = poly.intersection(box(-1e3, 0.0, 1e3, 1e3))       # right half only
    parts = [poly] if poly.geom_type == "Polygon" else list(getattr(poly, "geoms", []))
    out = []
    for p in parts:
        if p.area < 1e-6:
            continue
        m = trimesh.creation.extrude_polygon(p.simplify(0.05), z1_mm - z0_mm)
        m.apply_translation([0, 0, z0_mm])
        m.apply_scale(1e-3)
        out.append(m)
    return trimesh.util.concatenate(out) if out else None


def t55_keep_out(x0_mm: float, x1_mm: float, z_axis_mm: float = 35.0,
                 r_mm: float = 12.0 + TOOL_R_MM + 0.5):
    """Cylinder around the cartridge chamber a pocket may not enter (T5.5 wall
    plus the cutter radius plus 0.5 mm)."""
    import trimesh
    c = trimesh.creation.cylinder(radius=r_mm / 1e3, height=(x1_mm - x0_mm) / 1e3, sections=64)
    c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0_mm + x1_mm) / 2e3, 0.0, z_axis_mm / 1e3])
    return c


def cut(name: str, part, body_half, open_side: str = "bottom", keep_out=None,
        tool_r_mm: float = TOOL_R_MM, embed_mm: float | None = 4.0) -> Joint | None:
    """embed_mm: how far into the body the printed part reaches, measured in
    from its outermost embedded point (in y for side parts, z for top/bottom).
    The rest of what the CAD buries in the body is dropped: a CAD support pod
    reaching the centreline would otherwise make a 16 g plug."""
    import trimesh
    # Parts are exported as concatenated shells (wing + pylon overlap); make
    # them one manifold first or the booleans see nothing.
    pieces = part.split(only_watertight=False)
    if len(pieces) > 1:
        part = _bool("union", list(pieces))
    if keep_out is not None:
        part = _bool("difference", [part, keep_out])
    O_full = _bool("intersection", [part, body_half])
    if O_full is None or O_full.is_empty or abs(O_full.volume) < 1e-12:
        return None
    O = O_full
    if embed_mm is not None:
        b = O_full.bounds * 1e3
        if open_side == "top":           # parts on top: keep the top embed_mm
            slab = trimesh.creation.box(bounds=[[b[0, 0] - 1, -1, (b[1, 2] - embed_mm)],
                                                [b[1, 0] + 1, 1e3, b[1, 2] + 1]])
        elif b[1, 1] - b[0, 1] > b[1, 2] - b[0, 2]:   # side part: outermost y band
            slab = trimesh.creation.box(bounds=[[b[0, 0] - 1, b[1, 1] - embed_mm, b[0, 2] - 1],
                                                [b[1, 0] + 1, b[1, 1] + 1, b[1, 2] + 1]])
        else:                             # under the body: lowest z band
            slab = trimesh.creation.box(bounds=[[b[0, 0] - 1, -1, b[0, 2] - 1],
                                                [b[1, 0] + 1, 1e3, b[0, 2] + embed_mm]])
        slab.apply_scale(1e-3)
        O = _bool("intersection", [O_full, slab])
        if O is None or O.is_empty:
            O = O_full
    fp = _footprint(O).buffer(tool_r_mm)
    zb, zt = body_half.bounds[:, 2] * 1e3
    oz0, oz1 = O.bounds[:, 2] * 1e3
    plug = _prism(fp, zb - 1.0, oz1) if open_side == "bottom" else _prism(fp, oz0, zt + 1.0)
    if plug is None:
        return None
    if keep_out is not None:
        plug = _bool("difference", [plug, keep_out])
    pocket = _bool("intersection", [plug, body_half])
    outside = _bool("difference", [part, body_half])
    printed = _bool("union", [outside, pocket])
    return Joint(name, open_side, plug, pocket, printed,
                 pocket_cm3=abs(pocket.volume) * 1e6, embedded_cm3=abs(O_full.volume) * 1e6)


def machined_body(body_half, joints, x_ref_a_mm: float | None = None):
    """The body the mill makes: aero body minus every pocket, minus the nose
    ahead of Ref A (printed separately)."""
    import trimesh
    b = body_half
    pockets = [j.pocket for j in joints if j is not None]
    if pockets:
        b = _bool("difference", [b] + pockets)
    if x_ref_a_mm is not None:            # boolean, not slice_mesh_plane: keeps it closed
        keep = trimesh.creation.box(bounds=[[x_ref_a_mm / 1e3, -0.01, -0.01], [1.0, 0.2, 0.2]])
        b = _bool("intersection", [b, keep])
    pieces = sorted(b.split(only_watertight=False), key=lambda m: -abs(m.volume))
    b = pieces[0]
    b.metadata["dropped_fragments_mm3"] = float(sum(abs(m.volume) for m in pieces[1:]) * 1e9)
    return b


# Which way each printed part's pocket opens. The rear wing sits over the
# cartridge chamber, where the body is only the 3 mm T5.5 wall: no pocket is
# possible, so it is surface-bonded (cut() returns None after the keep-out).
OPEN_SIDE = {"supports": "bottom", "fwing": "bottom", "tethers": "bottom",
             "rwing": "top", "nose": "bottom"}
DENSITY_G_CM3 = {"supports": 1.04, "fwing": 1.24, "rwing": 1.24, "tethers": 1.24,
                 "nose": 1.24}


def nose_part(body_half, x_ref_a_mm: float, tenon_mm: float = 8.0, tenon_half_w_mm: float = 4.0):
    """The printed nose: the body ahead of Ref A plus a tenon reaching
    `tenon_mm` aft of it (8 mm wide: the cutter needs >= 6.25)."""
    import trimesh
    ahead = trimesh.creation.box(bounds=[[-1.0, -0.01, -0.01], [x_ref_a_mm / 1e3, 0.2, 0.2]])
    nose = _bool("intersection", [body_half, ahead])
    if nose is None or nose.is_empty or abs(nose.volume) < 1e-12:
        return None
    zc = 0.5 * sum(nose.bounds[:, 2])
    ten = trimesh.creation.box(bounds=[[x_ref_a_mm / 1e3 - 0.001, 0.0, 0.0],
                                       [(x_ref_a_mm + tenon_mm) / 1e3, tenon_half_w_mm / 1e3, zc]])
    return _bool("union", [nose, _bool("intersection", [ten, body_half])])


def make_all(body_half, assembly: dict, x_ref_a_mm: float, rear_face_mm: float,
             out_dir=None) -> dict:
    """Every joint on one car. Returns a report with the machined body, the
    printed parts and the full-car mass change per part (kg)."""
    import trimesh
    from pathlib import Path
    keep = trimesh.creation.box(bounds=[[x_ref_a_mm / 1e3, -0.01, -0.01], [1.0, 0.2, 0.2]])
    milled = _bool("intersection", [body_half, keep])          # aft of Ref A
    ko = t55_keep_out(rear_face_mm - 50.0 - 4.0, rear_face_mm + 1.0)
    parts = {s["name"]: trimesh.load(s["stl"], force="mesh") for s in assembly["extra_surfaces"]
             if not s["name"].startswith("wheel") and s["name"] != "halo"}
    np_ = nose_part(body_half, x_ref_a_mm)
    if np_ is not None:
        parts["nose"] = np_
    joints, report = [], {"parts": {}, "mass_delta_kg": {}}
    for name, mesh in parts.items():
        j = cut(name, mesh, milled, OPEN_SIDE.get(name, "bottom"), ko)
        dm = 0.0
        if j is None:
            report["parts"][name] = {"joint": "surface bond (no pocket possible)"}
        else:
            joints.append(j)
            dm = j.mass_delta_kg(DENSITY_G_CM3.get(name, 1.24))
            if name == "nose":
                # The nose ahead of Ref A is already in the body's mass (Part 1's
                # nose density): only its tenon's pocket is new.
                dm = 2 * j.pocket_cm3 * (DENSITY_G_CM3["nose"] - FOAM_G_CM3) * 1e-3
            report["parts"][name] = {"joint": f"pocket from the {j.open_side}",
                                     "pocket_cm3_per_side": j.pocket_cm3,
                                     "embedded_cm3_per_side": j.embedded_cm3}
        report["mass_delta_kg"][name] = dm
    mb = machined_body(milled, joints)
    report["machined_body_cm3_full"] = 2 * abs(mb.volume) * 1e6
    report["machined_body_watertight"] = bool(mb.is_watertight)
    report["dropped_fragments_mm3"] = mb.metadata.get("dropped_fragments_mm3", 0.0)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _full(mb).export(str(out / "machined_body.stl"))
        report["machined_body_half_stl"] = str(out / "machined_body_half.stl")
        mb.export(report["machined_body_half_stl"])
        for j in joints:
            if j.name == "supports":                 # separate left/right parts
                j.printed.export(str(out / f"printed_{j.name}_right.stl"))
                left = j.printed.copy()
                left.vertices[:, 1] *= -1
                left.invert()
                left.export(str(out / f"printed_{j.name}_left.stl"))
            else:                                    # centreline parts: one piece
                _full(j.printed).export(str(out / f"printed_{j.name}.stl"))
    return report


def _full(half):
    m = half.copy()
    m.vertices[:, 1] *= -1
    m.invert()
    try:
        return _bool("union", [half, m])
    except Exception:  # noqa: BLE001 -- fall back to two touching halves
        import trimesh
        return trimesh.util.concatenate([half, m])
