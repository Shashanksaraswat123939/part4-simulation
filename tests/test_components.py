"""Part 4 checks: wheel model vs CAD, wing gates at the regulation bounds, assembly."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def test_team_wheels_have_the_bearing_seat_and_a_closed_cfd_disc():
    import wheel as wh
    f, r = wh.design()
    assert 14.0 < f.R < 14.2 and 13.2 < f.w < 13.4 and 17.2 < r.w < 17.4          # T7.4/T7.5
    assert 1.0 < f.mass < 1.4 and 1.1 < r.mass < 1.5                          # g, incl. bearing
    for mesh, ax, bore_r, _hub in wh.team_wheel_meshes():
        v = mesh.vertices - (mesh.bounds[0] + mesh.bounds[1]) / 2
        r_min = np.hypot(*np.delete(v, ax, axis=1).T).min()
        assert mesh.is_watertight and abs(r_min - wh.BEARING_OD_MM / 2) < 0.05, r_min
    disc = wh.cfd_surface(f.R, f.w, 46.0, 23.25)
    b = disc.bounds * 1e3
    assert disc.is_watertight and abs(b[0, 2] + 0.3) < 1e-6 and abs(b[0, 1] - 23.25) < 1e-6


def test_naca_section_thickness_and_chord():
    import wings
    s = wings.naca4(20.0, 0.15)
    assert abs(s[:, 0].max() - 20.0) < 1e-9 and abs(s[:, 0].min()) < 1e-9
    assert abs((s[:, 1].max() - s[:, 1].min()) - 3.0) < 0.05


def test_front_wing_gates_pass_at_default_and_fail_past_bounds():
    import wings
    ok = wings.FrontWing()
    g = wings.front_wing_gates(ok, wings.build_front_wing(ok, 46.0, 14.05), 46.0, 14.05)
    assert min(g.values()) >= 0, {k: v for k, v in g.items() if v < 0}
    for bad, key in ((wings.FrontWing(chord_mm=14.0), "T8.6.2_chord_min"),
                     (wings.FrontWing(t_frac=0.08), "T8.6.3_thick_min"),
                     (wings.FrontWing(z_chord_mm=19.0), "T8.5.2_height_outboard"),
                     (wings.FrontWing(gap_to_wheel_mm=4.0), "T7.9.1_zone_ahead_of_wheel"),
                     (wings.FrontWing(half_span_mm=24.0), "T8.6.1_span"),
                     (wings.FrontWing(z_chord_mm=5.0), "T8.7_clear_air_track")):
        gb = wings.front_wing_gates(bad, wings.build_front_wing(bad, 46.0, 14.05), 46.0, 14.05)
        assert gb[key] < 0, (key, gb[key])


def _body():
    import trimesh
    b = trimesh.creation.box(extents=[0.18, 0.05, 0.04])
    b.apply_translation([0.025 + 0.09, 0.0, 0.0015 + 0.02])
    return b


def test_rear_wing_clear_air_and_placement():
    import wings
    body = _body()
    rw = wings.RearWing()
    m = wings.build_rear_wing(rw, 46.0, 120.3, body)
    g = wings.rear_wing_gates(rw, {k: m[k] for k in ("wing", "pylon")}, 46.0, 120.3, body, [body])
    assert min(g.values()) >= 0, {k: v for k, v in g.items() if v < 0}
    assert g["T9.6_clear_air_body"] < 1.5              # placed just above the limit
    short = wings.RearWing(half_span_mm=24.0)
    m2 = wings.build_rear_wing(short, 46.0, 120.3, body)
    assert wings.rear_wing_gates(short, {k: m2[k] for k in ("wing", "pylon")}, 46.0, 120.3,
                                 body, [body])["T9.5.1_span"] < 0


def test_assembly_writes_patches_masses_and_passes_gates():
    import assembly
    import trimesh
    with tempfile.TemporaryDirectory() as td:
        body = _body()
        half = trimesh.intersections.slice_mesh_plane(body, [0, 1, 0], [0, 0, 0], cap=True)
        half.export(f"{td}/body.stl", file_type="stl_ascii")
        r = assembly.build(120.3, 46.0, 43.72, f"{td}/body.stl", f"{td}/asm")
        assert r["failed_gates"] == [], r["failed_gates"]
        names = [s["name"] for s in r["extra_surfaces"]]
        assert names == ["wheelF", "wheelR", "supports", "halo", "fwing", "rwing", "tethers"]
        for s in r["extra_surfaces"]:
            m = trimesh.load(s["stl"])
            assert m.bounds[0, 1] >= -1e-9, s["name"]           # right half only
        rot = r["extra_surfaces"][0]["rotating"]
        R = r["wheel_design"]["front"]["R_mm"]
        assert abs(rot["omega"] + 20.0 / (R * 1e-3)) < 1.0      # rolls, top moves aft
        fhk = r["fixed_hardware_kwargs"]
        assert 0.001 < fhk["rear_wing_mass_kg"] < 0.012
        assert fhk["wheels_axles_mass_kg"] > 0.005
        back = assembly.extra_surfaces_from(f"{td}/asm/assembly.json")
        assert back[0]["rotating"]["origin"] == tuple(rot["origin"])


def test_nose_cone_is_closed_legal_and_in_the_assembly():
    import tempfile
    import trimesh
    import assembly
    import nose as ns
    with tempfile.TemporaryDirectory() as td:
        slim = trimesh.creation.box(extents=[0.18, 0.034, 0.019])   # car-sized front
        slim.apply_translation([0.115, 0.0, 0.013])
        half = trimesh.intersections.slice_mesh_plane(slim, [0, 1, 0], [0, 0, 0], cap=True)
        half.export(f"{td}/body.stl", file_type="stl_ascii")
        a = assembly.build(120.3, 46.0, 43.72, f"{td}/body.stl", f"{td}/asm", nose=ns.NoseCone())
        assert "nose" in [s_["name"] for s_ in a["extra_surfaces"]]
        nose_gates = {k: v for k, v in a["gates"].items() if "nose_" in k and k != "T8.2_nose_overhang"}
        assert len(nose_gates) == 4 and min(nose_gates.values()) >= 0, nose_gates
        assert 1.0 < a["parts_mass_g"]["nose"] < 3.0
        m = trimesh.load(f"{td}/asm/nose.stl")
        assert m.bounds[0, 0] * 1e3 >= 30.0 - 20.0 - 1e-6


def test_manufacturing_files_are_whole_closed_parts_in_mm():
    """What is sent to the mill and the printer: one closed solid per part,
    both halves, in mm, the two supports as two parts, the nose with its cone.
    (2026-09-30: both support files held half of the FRONT support, the nose
    file had no cone, and 5 of 7 files were not closed once re-read.)"""
    import trimesh
    import assembly
    import beam_support as bsm
    import joints
    import nose as ns
    with tempfile.TemporaryDirectory() as td:
        slim = trimesh.creation.box(extents=[0.192, 0.034, 0.024])
        slim.apply_translation([0.030 + 0.096, 0.0, 0.004 + 0.012])     # Ref A at x = 30 mm
        body = trimesh.intersections.slice_mesh_plane(slim, [0, 1, 0], [0, 0, 0], cap=True)
        body.export(f"{td}/body.stl", file_type="stl_ascii")
        a = assembly.build(120.3, 46.0, 43.72, f"{td}/body.stl", f"{td}/asm",
                           support=bsm.BeamSupport(),
                           nose=ns.NoseCone(blend_after_ref_a_mm=1.0, root_scale=1.0,
                                            material="PA12", wall_mm=0.8))
        rep = joints.make_all(body, a, 30.0, 222.0, f"{td}/mfg/manufacture")
        made = rep["manufactured"]
        for k, v in made.items():
            assert v["closed"] and v["pieces"] == v["pieces_expected"], (k, v)
            assert abs(v["file_cm3"] - v["cm3"]) < 0.01 * v["cm3"] + 1e-3, (k, v)
        load = lambda k: trimesh.load(made[k]["file"], force="mesh").bounds   # noqa: E731
        assert load("machined_body")[1, 0] - load("machined_body")[0, 0] > 150     # mm, not m
        f, r = load("printed_support_front"), load("printed_support_rear")
        assert f[1, 0] < 100 < r[0, 0] and f[0, 1] < -20 and f[1, 1] > 20, (f, r)
        # nose cone and front wing are ONE printed part, from the wing's leading
        # edge past the cone tip to the tenon; tether guides ride on the supports
        fa = load("printed_front_assembly")
        assert fa[0, 0] < 30.0 - 15.0 and fa[1, 1] > 30.0 and "printed_tethers" not in made, fa
        assert abs(rep["manufactured_g"] - sum(v["g"] for v in made.values())) < 1e-9
        # no two manufactured parts may occupy the same space (0.85 cm3 of
        # overlap was counted twice in the mass, 2026-09-30)
        solids = {k: trimesh.load(v["file"], force="mesh") for k, v in made.items()}
        keys = sorted(solids)
        for i, a_ in enumerate(keys):
            for b_ in keys[i + 1:]:
                inter = joints._bool("intersection", [solids[a_], solids[b_]])
                assert inter.is_empty or abs(inter.volume) < 1.0, (a_, b_, abs(inter.volume))


def test_support_follows_the_team_format_and_carries_the_loads():
    """Plate, strip, disc, flared boss and stub axle are one piece; the stub
    has a shoulder and a 3 mm journal where the 3x6x2.5 bearing sits; the pod's
    channel comes from the body; the hubcap is its own closed part; and the
    members grow when the loads or the layout ask for it."""
    import dataclasses
    import trimesh
    import beam_support as bsm
    import wheel as wh
    front_hub, _rear_hub = wh.hub_span_mm()
    assert 4.0 < front_hub[0] < front_hub[1] < 9.0 and abs(front_hub[1] - front_hub[0] - 3.0) < 0.1
    bs = bsm.BeamSupport()
    y_in, w, x, z, R = 23.25, 13.25, 46.0, 13.82, 14.12
    body = trimesh.creation.box(bounds=[[0.030, 0.0, 0.005], [0.222, 0.015, 0.028]])
    ch = bsm.pod_channel(bs, body, x, z, R)
    # the channel is the body's: from its floor, 4 mm of foam left above, inside the wheels' cylinder
    assert abs(ch["z_floor_mm"] - 5.0) < 1e-6 and ch["z_arch_mm"] <= 28.0 - 4.0 + 1e-9
    assert ch["length_mm"] <= bs.pod_len_mm and np.hypot(ch["length_mm"] / 2, z - 5.0) <= R - 0.3 + 1e-9
    out = bsm.build(bs, x, z, y_in, w, True, hub_mm=front_hub, R_mm=R, channel=ch)
    part, cap = out["support"], out["hubcap"]
    assert out["_pod"] and part.is_watertight and len(part.split(only_watertight=False)) == 1
    assert cap.is_watertight and abs(cap.bounds[1, 1] * 1e3 - (y_in + w)) < 1e-6      # flush outside
    y_bear = y_in + 0.5 * sum(front_hub)

    def radius_at(y_mm):          # the stub's radius about the axle, on the plane y
        sec = part.section(plane_origin=[0, y_mm / 1e3, 0], plane_normal=[0, 1, 0]).vertices * 1e3
        return np.hypot(sec[:, 0] - x, sec[:, 2] - z).max()
    assert abs(radius_at(y_bear) - 1.5) < 0.02                                        # the journal
    assert abs(radius_at(y_bear - 1.4) - bs.shoulder_d_mm / 2) < 0.02                 # the shoulder
    g = bsm.gates(bs, out, x, z, R, "front")
    assert all(m >= 0 for m in g.values()), {k: m for k, m in g.items() if m < 0}
    # without the strip the plate alone must carry the bending: it is made thicker
    alone = bsm.build(dataclasses.replace(bs, strip=False), x, z, y_in, w, True, hub_mm=front_hub,
                      R_mm=R, channel=ch)
    assert alone["_struct"]["plate_h_mm"] > 1.5 * out["_struct"]["plate_h_mm"]
    assert bsm.SAG_MAX_MM - alone["_struct"]["sag_mm"] >= -1e-9
    # harder loads ask for a thicker boss and a thicker disc
    y_disc = y_in + bs.disc_recess_mm
    bare = dataclasses.replace(bs, boss_d_mm=0.0)
    d0, t0 = bsm.sized_root_d_mm(bare, y_disc, y_bear, R), bsm.sized_disc_t_mm(bs, y_disc, y_bear, R)
    assert t0 >= bsm.PRINT_MIN_MM
    old = bsm.LOAD_RADIAL_N
    try:
        bsm.LOAD_RADIAL_N = 30 * old
        assert bsm.sized_root_d_mm(bare, y_disc, y_bear, R) > d0
        assert bsm.sized_disc_t_mm(bs, y_disc, y_bear, R) > t0
    finally:
        bsm.LOAD_RADIAL_N = old
