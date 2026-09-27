"""
component_contract.py -- every regulation number Part 4 uses, in one place.

Source: STEM Racing UAE Professional Class Technical Regulations 2025-26.
All lengths mm unless the name ends in _M. Coordinates match Parts 1-3:
x nose->tail (nose tip x=0), y centreline->right, z up from the track.
"""
from __future__ import annotations

# ---- reference planes (T1.17) -------------------------------------------
REF_A_AHEAD_OF_FRONT_AXLE = 16.0
REF_B_BEHIND_REAR_AXLE = 16.0

# ---- assembled car (T3) --------------------------------------------------
WIDTH_MIN, WIDTH_MAX = 65.0, 85.0          # T3.4
HEIGHT_MAX = 65.0                          # T3.5
TRACK_CLEARANCE_MIN = 1.5                  # T3.7

# ---- wheels (T7) ----------------------------------------------------------
WHEEL_DIA_MIN, WHEEL_DIA_MAX = 28.0, 32.0  # T7.5
FRONT_CONTACT_MIN, REAR_CONTACT_MIN = 13.0, 17.0   # T7.4
FRONT_GAP_MIN, REAR_GAP_MIN = 38.0, 30.0   # T7.2
T79_AHEAD_OF_FRONT_WHEEL = 5.0             # T7.9.1
T711_FRONT_WHEEL_HIDE_MAX_Z = 20.0         # T7.11

# ---- front wing and nose (T8) --------------------------------------------
NOSE_OVERHANG_MAX = 40.0                   # T8.2, from Ref A, whole nose assembly
NOSE_SUPPORT_Z_MAX = 25.0                  # T8.5.1
NOSE_SUPPORT_HALF_WIDTH_MAX = 15.0         # T8.5.1
FRONT_WING_Z_MAX_OUTBOARD = 20.0           # T8.5.2, where |y| > 15
FRONT_ENDPLATE_WIDTH_MAX = 10.0            # T8.5.3
FRONT_ENDPLATE_Z_MAX = 25.0                # T8.5.3
FRONT_SPAN_SINGLE_MIN = 50.0               # T8.6.1
FRONT_SPAN_SEGMENT_MIN = 25.0              # T8.6.1, two segments
FRONT_CHORD_MIN, FRONT_CHORD_MAX = 15.0, 25.0     # T8.6.2
FRONT_ELEMENTS_MAX = 3
WING_THICK_MIN, WING_THICK_MAX = 2.0, 6.0  # T8.6.3 / T9.5.3
CLEAR_AIR = 5.0                            # T8.7 / T9.6

# ---- rear wing (T9) -------------------------------------------------------
REAR_OVERHANG_MAX = 40.0                   # T9.4.2, from Ref B
REAR_Z_MAX = 65.0                          # T9.4.3
REAR_SPAN_MIN = 50.0                       # T9.5.1, single unbroken
REAR_CHORD_MIN, REAR_CHORD_MAX = 15.0, 25.0       # T9.5.2
REAR_ELEMENTS_MAX = 2
REAR_HEIGHT_DEVIATION_MAX = 15.0           # T9.5.4

# ---- tether guides (T6) ---------------------------------------------------
TETHER_FROM_AXLE_MAX = 10.0                # T6.1
TETHER_ID_MIN, TETHER_ID_MAX = 3.5, 6.0    # T6.2

# ---- materials (g/cm3) ----------------------------------------------------
DENSITY_G_CM3 = {
    "PLA": 1.24, "ABS": 1.04, "PETG": 1.27, "SLA_resin": 1.15,
    "nylon_PA12": 1.01, "PA12": 1.01, "titanium": 4.43,
}


def ref_plane_A(x_front_mm: float) -> float:
    return x_front_mm - REF_A_AHEAD_OF_FRONT_AXLE


def ref_plane_B(x_front_mm: float, W_mm: float) -> float:
    return x_front_mm + W_mm + REF_B_BEHIND_REAR_AXLE
