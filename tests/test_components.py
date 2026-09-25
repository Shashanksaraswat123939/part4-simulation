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


def test_wheel_model_reproduces_the_cad_wheels():
    import wheel as W
    cad = json.loads((HERE / "cad_wheels.json").read_text())
    for w, key in ((W.CAD_FRONT, "front_wheel"), (W.CAD_REAR, "rear_wheel")):
        assert abs(w.inertia / cad[key]["I_gmm2"] - 1) < 0.03, (key, w.inertia)
        assert abs(w.mass / cad[key]["mass_g"] - 1) < 0.05, (key, w.mass)


def test_thick_ring_closed_form():
    import wheel as W
    w = W.Wheel(R=14.0, t_rim=0.5, w=13.0, n_spokes=0, l_hub=0.0)
    m, i = w.parts()["rim"]
    m_ref = 1.04e-3 * math.pi * (14.0**2 - 13.5**2) * 13.0
    assert abs(m / m_ref - 1) < 1e-9 and abs(i / (m_ref * (14.0**2 + 13.5**2) / 2) - 1) < 1e-9


def test_every_wheel_design_is_legal_and_at_least_as_stiff():
    import component_contract as cc
    import wheel as W
    for name in W.DESIGNS:
        s = W.summary(name)
        assert cc.WHEEL_DIA_MIN <= 2 * s["front"]["R_mm"] <= cc.WHEEL_DIA_MAX
        assert s["front"]["width_mm"] >= cc.FRONT_CONTACT_MIN
        assert s["rear"]["width_mm"] >= cc.REAR_CONTACT_MIN
        assert s["stiffness_vs_cad_min"] >= 0.999, name
        assert s["printable"], name
    assert W.mean_inertia_kg_m2("carbon_rim_capped") < W.mean_inertia_kg_m2("cad_v2")


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
        assert abs(rot["omega"] + 20.0 / (14.05e-3)) < 1.0      # rolls, top moves aft
        fhk = r["fixed_hardware_kwargs"]
        assert 0.001 < fhk["rear_wing_mass_kg"] < 0.012
        assert fhk["wheels_axles_mass_kg"] > 0.005
        back = assembly.extra_surfaces_from(f"{td}/asm/assembly.json")
        assert back[0]["rotating"]["origin"] == tuple(rot["origin"])


if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _fails = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)(); print("PASS", _n)
        except Exception as e:  # noqa: BLE001
            _fails += 1; print("FAIL", _n, "->", repr(e))
    print(f"{_fails} failed")
    sys.exit(1 if _fails else 0)
