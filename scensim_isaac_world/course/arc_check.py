"""086 C5b: a drive through a course's banked bend, its body roll and lateral specific force
against the record. Pure numpy; reads a trace in the ego_trace attachment's format (ego.csv).

    python3 -m scensim_isaac_world.course.arc_check <ego.csv> <map dir> [--json out.json]

Each frame is placed on the centreline (its nearest station). On a straight window and an arc
window (m along the course) it reports the mean roll (map frame, ROS: left side down
negative) and, on the arc, the lateral specific force along the body's y axis (the truth's
world velocities smoothed over 0.5 s and differenced, gravity included: what an IMU reads)
against v w cos(theta) - g sin(theta), theta the record's bank there, v the ground speed and w
the yaw rate; and v w, what it would be were the road level. The checks (086 C5b's
EXPECTED.md): the straight's roll is the crown under the car (the record's, on the car's side)
within 0.5 deg; the arc's is the bank within 1.0 deg, and less the straight's within 1.0 deg;
the lateral force within 0.2 m/s^2 on average.
"""
import argparse
import json
import os
import sys

import numpy as np

G = 9.81
ROLL_STRAIGHT_TOL = 0.5
ROLL_ARC_TOL = 1.0
LATERAL_TOL = 0.2


def check(csv, map_dir, straight=(50.0, 140.0), arc=(220.0, 290.0)):
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial import cKDTree

    from .. import road_detail as rd
    from . import record
    from .geometry import _normals

    d = np.genfromtxt(csv, delimiter=",", names=True)
    c = np.load(os.path.join(map_dir, "course", "centreline.npz"))
    st = record.load(os.path.join(map_dir, "course"))
    sec = rd.StationSection(st)
    _, j = cKDTree(np.stack([c["x"], c["y"]], 1)).query(np.stack([d["x"], d["y"]], 1))
    s = np.asarray(c["s"], float)[j]
    keep = np.concatenate([[True], np.diff(d["t"]) > 0])
    d, s, j = d[keep], s[keep], j[keep]
    t = d["t"]
    nrm = _normals(c["x"], c["y"])
    off = (d["x"] - c["x"][j]) * nrm[j, 0] + (d["y"] - c["y"][j]) * nrm[j, 1]
    u = s - float(c["s"][0])
    n = max(1, int(round(0.5 / np.median(np.diff(t)))))
    vs = uniform_filter1d(np.stack([d["vx"], d["vy"], d["vz"]], 1), n, axis=0)
    f = np.gradient(vs, t, axis=0) + np.array([0.0, 0.0, G])
    cr, sr, cp, sp, cy, sy = (np.cos(d["roll"]), np.sin(d["roll"]), np.cos(d["pitch"]),
                              np.sin(d["pitch"]), np.cos(d["yaw"]), np.sin(d["yaw"]))
    ybody = np.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr], 1)
    fy = uniform_filter1d(np.sum(f * ybody, 1), n)
    speed = np.hypot(vs[:, 0], vs[:, 1])
    vw = speed * uniform_filter1d(d["wz"], n)
    bank = np.interp(u, sec.u, sec.common["bank"])
    want = vw * np.cos(bank) - G * np.sin(bank)
    # the road's own roll under the car: the bank, and the crown on the car's side of the
    # crown line (a fall to the left is a negative roll)
    side = np.where(off >= np.interp(u, sec.u, sec.common["crown_offset"]), 0, 1)
    crown = np.where(side == 0, np.interp(u, sec.u, st.side["crown"][0]),
                     -np.interp(u, sec.u, st.side["crown"][1]))
    road_roll = np.degrees(-bank - np.arctan(crown))
    roll = np.degrees(d["roll"])
    ws = (s >= straight[0]) & (s <= straight[1])
    wa = (s >= arc[0]) & (s <= arc[1])
    if not ws.any() or not wa.any():
        raise ValueError(f"the trace covers s {s.min():.0f} to {s.max():.0f} m, not both windows")
    r = lambda v: round(float(v), 3)  # noqa: E731
    out = {"frames": int(len(t)), "s_m": [r(s.min()), r(s.max())],
           "lane_offset_m": [r(off[ws | wa].min()), r(off[ws | wa].max())],
           "speed_arc_mps": r(speed[wa].mean()),
           "road_roll_straight_deg": r(road_roll[ws].mean()), "road_roll_arc_deg": r(road_roll[wa].mean()),
           "roll_straight_deg": r(roll[ws].mean()), "roll_arc_deg": r(roll[wa].mean()),
           "fy_arc": r(fy[wa].mean()), "fy_want": r(want[wa].mean()),
           "fy_err_mean": r((fy - want)[wa].mean()), "fy_err_worst": r(np.abs(fy - want)[wa].max()),
           "fy_if_level": r(vw[wa].mean())}
    out["checks"] = {
        "straight_roll": abs(out["roll_straight_deg"] - out["road_roll_straight_deg"]) <= ROLL_STRAIGHT_TOL,
        "arc_roll": abs(out["roll_arc_deg"] - out["road_roll_arc_deg"]) <= ROLL_ARC_TOL,
        "arc_less_straight": abs((out["roll_arc_deg"] - out["roll_straight_deg"])
                                 - (out["road_roll_arc_deg"] - out["road_roll_straight_deg"])) <= ROLL_ARC_TOL,
        "lateral": abs(out["fy_err_mean"]) <= LATERAL_TOL}
    out["pass"] = all(out["checks"].values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("map_dir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = check(a.csv, a.map_dir)
    text = json.dumps(out, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
