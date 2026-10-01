"""
beam_support.py -- the team's wheel-support system, built from numbers.

The format is the team's own support CAD (hardware_cad/*_wheel_support.stl,
measured 2026-10-01): one printed part per axle, wheel to wheel, with the
AXLE BUILT IN and the bearing pressed onto it.

  pod      AT EACH AXLE THE SUPPORT IS THE CAR'S LOWER BODY (as on the team's
           LF1): over a channel `pod_len` long, from the floor up to
           `pod_arch`, the foam is milled away from below across the whole
           width and a hollow printed shell takes its place, its outside the
           body's own surface, so the car's shape is unchanged. The foam is
           an arch over it. The channel is taken from the body each time, so
           the pod follows the body through the search and the mass sizing.
           It shortens itself to stay inside the wheels' cylinder (T7.12.1)
           and lowers its roof to leave 4 mm of foam above.
  plate    wheel to wheel, a flat slab edge-on to the flow (x chord, z
           thickness, exponent p; the team's is 20 x 1.7 mm), lying in the
           pod's roof. With the
  strip    a thinner one near the floor (theirs: 9 x 1 mm), it makes
           a frame the disc ties together: two thin plates far apart are
           stiff where one is not. The plate's thickness is SIZED HERE for
           the loads below.
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

LOADS ("normal racing loads"): 3 N radial at the bearing and 1.2 N sideways
  at the tyre's contact, SLS PA12 at 44 MPa with a safety factor of 2, and at
  most 0.15 mm of sag at the bearing. These are CALIBRATED TO THE TEAM'S OWN
  SUPPORT: their 1 mm disc, which races, carries just this at safety 2 (the
  stub's moment bends the disc between the boss and the two members). Launch
  load transfer and the finish catch are each about 1 N a wheel. (8 N and
  3 N, tried first, would have failed the team's proven disc.)

LIGHTEST THAT PASSES: every thickness here (plate, strip, disc, pod wall,
  hubcap) is the printer's minimum wall unless a load check asks for more,
  and then it is exactly what the check asks. The plate stops just inside
  the pod, whose roof carries its load on. What the air decides (the plate's
  chord, the discs, the pod's length) is left to the search.

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
LOAD_RADIAL_N = 3.0
LOAD_AXIAL_N = 1.2
PRINT_MIN_MM = 0.7            # thinnest wall SLS PA12 prints reliably
PA12_STRENGTH_MPA = 44.0
PA12_E_MPA = 1650.0
SAFETY = 2.0
SAG_MAX_MM = 0.15


@dataclass(frozen=True)
class BeamSupport:
    # The plate (the team's is 20 x 1.7 mm). Its thickness is raised to what
    # the loads need. Without a pod it is a beam in a slot of its own.
    beam_w_mm: float = 12.0       # x: the chord
    beam_h_mm: float = PRINT_MIN_MM   # z: the thickness, at least
    p: float = 2.0                # 2 = ellipse (smoothest), larger = boxier
    x_offset_mm: float = 0.0      # beam centre relative to the axle
    z_offset_mm: float = 0.0
    disc_front: bool = True       # the team's CAD has a disc at every wheel
    disc_rear: bool = True
    disc_r_mm: float = 12.0       # <= axle height - 1.5 mm (T3.7): 12.25 here
    disc_t_mm: float = PRINT_MIN_MM   # at least; raised to carry the stub's moment
    disc_recess_mm: float = 1.0   # disc's outer face this far inside the rim
    #                               (negative: standing inboard of the wheel)
    boss_d_mm: float = 5.8        # the flare's diameter at the disc, at least
    fillet_r_mm: float = 3.0      # the flare's radius
    shoulder_d_mm: float = 4.0    # what the bearing's inner race sits against
    pod: bool = True              # False: a bare beam in its own slot (before 2026-10-01)
    pod_len_mm: float = 14.0      # along the car, at most (the plate's chord and a margin)
    pod_arch_mm: float = 19.0     # the channel's roof, at most
    pod_wall_mm: float = PRINT_MIN_MM
    pod_rim_mm: float = 3.0       # the glue land left around each open end of the pod
    plate_lap_mm: float = 5.0     # how far the plate runs into the pod's roof
    strip: bool = True
    strip_w_mm: float = 7.0
    strip_h_mm: float = PRINT_MIN_MM
    strip_above_floor_mm: float = 2.5   # its mid-height above the body's floor at the axle:
    #                                     high enough to leave through the body's side, as
    #                                     the plate does (at the floor itself it grazes the
    #                                     curved underside and tears the printed file)
    hubcap: bool = True
    hubcap_r_mm: float = 12.0      # as the disc: 1.5 mm off the track (T3.7)
    hubcap_t_mm: float = PRINT_MIN_MM
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


def sized_disc_t_mm(bs: BeamSupport, y_disc_mm: float, y_bearing_mm: float, R_mm: float) -> float:
    """The disc's thickness: the stub's moment leaves the boss through the
    disc, half upward to the plate and half downward to the strip, each a
    strip of disc as wide as the member it feeds, bending out of its plane."""
    M = LOAD_RADIAL_N * (y_bearing_mm - y_disc_mm) + LOAD_AXIAL_N * R_mm
    w = min(bs.beam_w_mm, bs.strip_w_mm) if bs.strip else bs.beam_w_mm
    return max(bs.disc_t_mm, PRINT_MIN_MM, math.sqrt(6 * (M / 2) / (w * PA12_STRENGTH_MPA / SAFETY)))


def sized_root_d_mm(bs: BeamSupport, y_disc_mm: float, y_bearing_mm: float, R_mm: float) -> float:
    """The boss diameter at the disc: the larger of the asked boss_d and what
    the root's bending moment needs at the allowed stress."""
    M = LOAD_RADIAL_N * (y_bearing_mm - y_disc_mm) + LOAD_AXIAL_N * R_mm
    need = (32 * M / (math.pi * PA12_STRENGTH_MPA / SAFETY)) ** (1 / 3)
    return max(bs.boss_d_mm, need, bs.shoulder_d_mm)


def _unit_section_stiffness(p_exp: float) -> float:
    """I / (chord x thickness^3) of a superellipse slab about its chord line
    (bending up and down): pi/64 for an ellipse."""
    P = section(BeamSupport(beam_w_mm=1.0, beam_h_mm=1.0, p=p_exp), 360)
    x, z = P[:, 0], P[:, 1]
    cr = x * np.roll(z, -1) - np.roll(x, -1) * z
    return float(abs(np.sum((z ** 2 + z * np.roll(z, -1) + np.roll(z, -1) ** 2) * cr)) / 12)


def _unit_section_area(p_exp: float) -> float:
    P = section(BeamSupport(beam_w_mm=1.0, beam_h_mm=1.0, p=p_exp), 360)
    x, z = P[:, 0], P[:, 1]
    return float(abs(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)) / 2)


def frame(bs: BeamSupport, plate_h_mm: float, span_mm: float, reach_mm: float, arm_mm: float,
          R_mm: float) -> dict:
    """What the loads do to the members between the body's side and the disc
    (`span`), with the bearing `reach` beyond the disc.

    With the strip, plate and strip are a frame tied by the disc: the moment
    is a push-pull pair `arm` apart, and each member bends as a beam built in
    at both ends, sharing the radial load by stiffness. Without it the plate
    is a cantilever carrying everything in bending. Returns the worst stress
    and the sag this adds at the bearing."""
    k, a = _unit_section_stiffness(bs.p), _unit_section_area(bs.p)
    I1, A1 = k * bs.beam_w_mm * plate_h_mm ** 3, a * bs.beam_w_mm * plate_h_mm
    M_tip = LOAD_RADIAL_N * reach_mm + LOAD_AXIAL_N * R_mm        # at the disc
    M_root = M_tip + LOAD_RADIAL_N * span_mm                      # at the body's side
    E = PA12_E_MPA
    if bs.strip and arm_mm > 2.0:
        I2, A2 = k * bs.strip_w_mm * bs.strip_h_mm ** 3, a * bs.strip_w_mm * bs.strip_h_mm
        N = M_root / arm_mm
        m1 = LOAD_RADIAL_N * I1 / (I1 + I2) * span_mm / 2                 # end moment, plate
        m2 = LOAD_RADIAL_N * I2 / (I1 + I2) * span_mm / 2
        stress = max(N / A1 + m1 * plate_h_mm / 2 / I1, N / A2 + m2 * bs.strip_h_mm / 2 / I2)
        drop = LOAD_RADIAL_N * span_mm ** 3 / (12 * E * (I1 + I2))
        slope = N * span_mm / E * (1 / A1 + 1 / A2) / arm_mm
    else:
        stress = M_root * plate_h_mm / 2 / I1
        drop = (LOAD_RADIAL_N * span_mm ** 3 / 3 + M_tip * span_mm ** 2 / 2) / (E * I1)
        slope = (LOAD_RADIAL_N * span_mm ** 2 / 2 + M_tip * span_mm) / (E * I1)
    return {"stress_mpa": float(stress), "sag_mm": float(drop + slope * reach_mm)}


def pod_channel(bs: BeamSupport, body, x_axle_mm: float, z_axle_mm: float, R_mm: float) -> dict:
    """The channel the pod takes at one axle of this body (half body, metres):
    its ends along the car and its roof. The length is cut back until the
    body's floor corners there are inside the wheels' cylinder (T7.12.1); the
    roof is lowered to leave 4 mm of foam above, and to stay in the cylinder."""
    sec = body.section(plane_origin=[x_axle_mm / 1e3, 0, 0], plane_normal=[1, 0, 0])
    if sec is None:
        return {}
    v = sec.vertices * 1e3
    z_floor, z_top = float(v[:, 2].min()), float(v[:, 2].max())
    reach = (R_mm - 0.3) ** 2 - (z_axle_mm - z_floor) ** 2
    length = min(bs.pod_len_mm, 2 * math.sqrt(reach)) if reach > 0 else 0.0
    z_arch = min(bs.pod_arch_mm, z_top - 4.0,
                 z_axle_mm + math.sqrt(max((R_mm - 0.3) ** 2 - (length / 2) ** 2, 0.0)))
    near = lambda z: v[np.abs(v[:, 2] - z) < 1.5, 1]               # noqa: E731
    return {"x0_mm": x_axle_mm - length / 2, "x1_mm": x_axle_mm + length / 2, "z_arch_mm": z_arch,
            "z_floor_mm": z_floor, "z_body_top_mm": z_top, "wall_mm": bs.pod_wall_mm,
            "rim_mm": bs.pod_rim_mm,
            "length_mm": length, "y_body_mm": float(v[:, 1].max()),
            # where the plate (just under the roof) leaves the body
            "y_body_at_plate_mm": float(near(z_arch - 1.0).max()) if len(near(z_arch - 1.0)) else 0.0}


def channel_outline(ch: dict):
    """The channel's outline in (x, z), mm: straight walls up from below the
    floor, a flat roof, the two roof corners rounded to the cutter's radius
    (a ball-end from below leaves exactly that)."""
    from shapely.geometry import box
    return box(ch["x0_mm"], -5.0, ch["x1_mm"], ch["z_arch_mm"]).buffer(-TOOL_R_MM).buffer(TOOL_R_MM)


def build(bs: BeamSupport, x_axle_mm: float, z_axle_mm: float, y_inner_mm: float,
          wheel_width_mm: float, disc: bool, hub_mm: tuple = (5.25, 8.25),
          R_mm: float = 14.12, channel: dict | None = None) -> dict:
    """Right half of one axle's support, metres: plate, strip, disc, boss and
    stub as one piece, and the hubcap as its own. `hub_mm` is where the
    wheel's hub (its bearing seat) lies, measured from the wheel's inner
    face. `channel` is this body's pod channel (pod_channel): the plate then
    lies in the pod's roof, and the members are sized for the span from the
    body's side. (The pod's shell is cut from the body in joints.py: it is
    the body's own shape.)"""
    import dataclasses
    import trimesh
    y_out = y_inner_mm + wheel_width_mm
    y_disc = y_inner_mm + bs.disc_recess_mm                # the disc's outer face
    y_bear = y_inner_mm + 0.5 * (hub_mm[0] + hub_mm[1])
    disc_t = sized_disc_t_mm(bs, y_disc, y_bear, R_mm) if disc else 0.0
    y_beam_end = y_disc - disc_t
    y_tip = y_out - (bs.hubcap_t_mm if bs.hubcap else 0.0)
    root_d = sized_root_d_mm(bs, y_disc, y_bear, R_mm)
    line = stub_outline(bs, y_disc, y_bear, y_tip, root_d)
    st = dict(structure(bs, line, y_bear, y_disc, R_mm), root_d_mm=root_d, disc=disc,
              disc_t_mm=disc_t)
    if disc:
        w = min(bs.beam_w_mm, bs.strip_w_mm) if bs.strip else bs.beam_w_mm
        st["disc_stress_mpa"] = 6 * (LOAD_RADIAL_N * (y_bear - y_disc) + LOAD_AXIAL_N * R_mm) / 2             / (w * disc_t ** 2)
    pod = bs.pod and bool(channel) and channel["length_mm"] > 2 * TOOL_R_MM + 1.0
    plate_h, xc = bs.beam_h_mm, x_axle_mm + bs.x_offset_mm
    zc = z_axle_mm + bs.z_offset_mm
    z_strip = None
    if pod:
        # the plate lies in the pod's roof (its top half a wall into it); raise
        # its thickness until the frame carries the loads and the bearing sags
        # no more than allowed
        span = max(y_beam_end - channel["y_body_at_plate_mm"], 0.5)
        z_strip = channel["z_floor_mm"] + bs.strip_above_floor_mm
        for _ in range(40):
            zc = channel["z_arch_mm"] - bs.pod_wall_mm / 2 - plate_h / 2
            fr = frame(bs, plate_h, span, y_bear - y_beam_end, zc - z_strip, R_mm)
            if fr["stress_mpa"] <= st["allow_mpa"] and st["sag_mm"] + fr["sag_mm"] <= SAG_MAX_MM:
                break
            plate_h *= 1.1
        st.update(frame_stress_mpa=fr["stress_mpa"], sag_mm=st["sag_mm"] + fr["sag_mm"],
                  plate_h_mm=plate_h, plate_h_asked_mm=bs.beam_h_mm, span_mm=span)
    bs_sized = dataclasses.replace(bs, beam_h_mm=plate_h)
    # one revolved piece: (the disc,) the flare, the shoulder, the journal
    prof = [(0.0, y_beam_end - 0.05)]
    if disc:
        prof += [(bs.disc_r_mm, y_beam_end - 0.05), (bs.disc_r_mm, y_disc)]
    else:
        prof += [(line[0][0], y_beam_end - 0.05)]
    prof += line + [(0.0, y_tip)]
    # with a pod the plate stops a lap inside it: the pod's roof carries on
    y_plate0 = max(0.0, channel["y_body_at_plate_mm"] - bs.plate_lap_mm) if pod else 0.0
    parts = [_extrude_xz(section(bs_sized), y_plate0, y_beam_end + 0.05, xc, zc),
             _revolve_y(prof, x_axle_mm, z_axle_mm)]
    if pod and bs.strip:
        parts.append(_extrude_xz(section(dataclasses.replace(bs, beam_w_mm=bs.strip_w_mm,
                                                             beam_h_mm=bs.strip_h_mm)),
                                 0.0, y_beam_end + 0.05, x_axle_mm, z_strip))
    part = trimesh.boolean.union(parts, engine="manifold")
    cap = None
    if bs.hubcap:
        # a plate flush with the wheel's outer face, on a short hub that the
        # stub's end presses into (a 3 mm blind bore)
        hub_r, hub_len = 2.5, 2.0
        cap = _revolve_y([(0.0, y_tip), (STUB_D_MM / 2, y_tip), (STUB_D_MM / 2, y_tip - hub_len),
                          (hub_r, y_tip - hub_len), (hub_r, y_tip), (bs.hubcap_r_mm, y_tip),
                          (bs.hubcap_r_mm, y_out), (0.0, y_out)], x_axle_mm, z_axle_mm)
    return {"support": part, "hubcap": cap, "_section_centre_mm": (xc, zc), "_struct": st,
            "_pod": pod}


def gates(bs: BeamSupport, meshes: dict, x_axle_mm: float, z_axle_mm: float, R_mm: float,
          tag: str) -> dict:
    v = meshes["support"].vertices * 1e3
    r = np.hypot(v[:, 0] - x_axle_mm, v[:, 2] - z_axle_mm).max()
    st = meshes["_struct"]
    g = {f"T7.12.1_{tag}_support_in_cylinder": R_mm - r,
         f"T3.7_{tag}_support_clearance": v[:, 2].min() - cc.TRACK_CLEARANCE_MIN,
         # loads: margins in MPa and mm
         f"load_{tag}_stub_stress": st["allow_mpa"] - st["stress_mpa"],
         f"load_{tag}_bearing_sag": SAG_MAX_MM - st["sag_mm"]}
    if "frame_stress_mpa" in st:
        g[f"load_{tag}_frame_stress"] = st["allow_mpa"] - st["frame_stress_mpa"]
    if "disc_stress_mpa" in st:
        g[f"load_{tag}_disc_stress"] = st["allow_mpa"] - st["disc_stress_mpa"]
    if not meshes.get("_pod"):   # a bare beam sits in a slot of its own, which the ball must cut
        g[f"machining_{tag}_slot_corner_radius"] = min_curvature_radius_mm(bs) - TOOL_R_MM
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
