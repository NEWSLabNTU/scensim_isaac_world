"""086 C6c1's reference: R0's quarter car at 4 kHz (`fixture_course.reference`) with R5's cams
(`contact_offline.cam_road`'s envelope) and R2's damper, the front-left corner of aib's Model 3,
on C6c's fixtures: C6a's washboard at each sweep speed (its load amplitude, `c6c_check`'s
window), and the class C profile (`road_detail.Profile`'s defaults) at 10 and 20 m/s (its
spectrum's peak, `c6c_check.welch_peak`). Run before the Isaac runs; its output goes into
c6c_check's REF_WB and REF_CC. No Isaac.

    python3 spikes/c6c_reference.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import contact_offline as co  # noqa: E402
import fixture_course as fc  # noqa: E402
from scensim_isaac_world import road_contact as rc  # noqa: E402
from scensim_isaac_world import road_detail as rd  # noqa: E402
from scensim_isaac_world.course import c6c_check as cc  # noqa: E402

SPEEDS = (5, 7, 8, 9, 10, 12, 14, 20)
WB_LEN = 30.0


def corner():
    return rc.Corner(co.SPRUNG, co.K, co.C, co.TRAVEL, co.R, co.WIDTH)


def ref(name, v, road):
    return fc.reference(name, v, co.SPRUNG, co.K, co.C, travel=co.TRAVEL, sag=co.SAG,
                        radius=co.R, road=road, rim=rc.Corner.rim_clearance,
                        damper=corner().damper)[0]


def profile_road(p):
    """The cams' effective height over the profile, x from fc.X0 (the profile's start)."""
    tc = rc.TireContact([corner()])
    x0 = fc.X0 - 5.0

    def f(x, y):
        return p.h(np.asarray(x) - fc.X0 + p.u0)
    s, w, _ = tc.envelope(f, np.array([[x0, 0.0]]), np.array([[1.0, 0.0]]), p.length + 10.0)
    return x0 + s, w[0]


if __name__ == "__main__":
    out = {"washboard": {}, "class_c": {}}
    road = co.cam_road("washboard")
    for v in SPEEDS:
        o = ref("washboard", v, road)
        x, load = np.array(o["x"]) - fc.X0, np.array(o["load"])
        m = (x >= cc.WB_IN) & (x <= WB_LEN - cc.WB_IN)
        out["washboard"][v] = round(cc.amplitude(load[m]), 1)
        print("washboard", v, out["washboard"][v], flush=True)
    p = rd.Profile(u0=0.0)
    road = profile_road(p)
    for v in (10, 20):
        o = ref("washboard", v, road)
        x, load = np.array(o["x"]) - fc.X0, np.array(o["load"])
        m = (x >= cc.CC_IN) & (x <= p.length - cc.CC_IN)
        hz, height, _, _ = cc.welch_peak(load[m], 500.0)
        out["class_c"][v] = [round(hz, 2), round(height, 2)]
        print("class_c", v, out["class_c"][v], flush=True)
    print("C6C-REFERENCE " + json.dumps(out))
