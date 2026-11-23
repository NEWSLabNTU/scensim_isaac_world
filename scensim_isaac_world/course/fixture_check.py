"""086 C6a: a `fixture_drive` run judged against R0's reference (the spec's table, computed
from `spikes/fixture_course.reference` before the runs), the front-left wheel's.

All from the run's `fixture.csv`, one row a PhysX step (60 Hz; a load is our contact's mean
over the step). The front-left hub's s is `fl_s`; the fixture's face is at its `fixture_s`.

- static: the wheel's mean load with its hub STATIC_FROM to STATIC_TO m before the face;
- a cleat: its peak load (hub within the face less 1 m and the cleat's end plus 3 m) and its
  body rise (the front-left corner's attachment height `fl_z` over its static mean, to 10 m
  past) within TOL of the reference's, and its load rising ahead of the hub: the hub's s less
  the face's at the first step over RISE x static is negative. Its sample count (steps over
  twice static) and airborne time are reported;
- the pothole: the load falls under UNLOAD x static over the hole, and its far-edge peak
  (the maximum after that minimum) comes within FAR_EDGE_S s and within TOL of the
  reference's;
- each landing, any wheel, any case: after a step whose load is LANDED_N or less, the time
  from the load's return (within its first loaded step, by that step's airborne share) to
  the landing's peak (the middle of the step that holds it) within LANDING_S;
- washboard: the load's minimum and maximum over the washboard each within TOL;
- cleat_10 at 20 m/s from each start offset (`#k`): no miss (a peak at least twice static)
  and no throw (body rise over THROW x the reference's, or pitch over THROW_PITCH deg).

A case whose start check failed (`start_ok`) is refused, not judged, and fails the run.

    python3 -m scensim_isaac_world.course.fixture_check <run dir> [--json out.json]
"""
import argparse
import csv
import json
import math
import sys

import numpy as np

TOL = 0.30
STATIC_FROM, STATIC_TO = 8.0, 2.0  # m before the face
RISE = 1.05  # the reference's first rise: load over 1.05 x static
UNLOAD = 0.30
FAR_EDGE_S = 0.5
LANDED_N = 1.0
LANDING_S = (0.010, 0.030)
THROW = 3.0
THROW_PITCH = 20.0  # deg
PEAK_BEFORE, PEAK_AFTER, RISE_AFTER = 1.0, 3.0, 10.0  # m about the fixture
# the spec's table: peak N, body rise m, first rise m, airborne s; washboard (min, max) N
REF = {"cleat_05@10": dict(peak=13.3e3, rise=0.026, first=-0.18, air=0.04),
       "cleat_10@10": dict(peak=20.2e3, rise=0.078, first=-0.24, air=0.12),
       "cleat_10@20": dict(peak=24.0e3, rise=0.051, first=-0.24, air=0.09),
       "cleat_15@10": dict(peak=26.6e3, rise=0.202, first=-0.28, air=0.30),
       "pothole@10": dict(peak=10.1e3, rise=0.002, first=0.38, air=0.02),
       "washboard@10": dict(range=(1.16e3, 7.36e3)),
       "washboard@20": dict(range=(1.43e3, 8.24e3))}


def within(x, ref, tol=TOL):
    return bool(abs(x - ref) <= tol * abs(ref))


def read(run):
    with open(f"{run}/fixture.csv") as f:
        rows = list(csv.DictReader(f))
    with open(f"{run}/cases.json") as f:
        meta = json.load(f)
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    cols = {}
    for case, rs in by.items():
        rs = [r for r in rs if r["load0"] != ""]
        c = {k: np.array([float(r[k]) for r in rs]) for k in
             ("t", "fl_s", "fl_z", "pitch", "load0", "load1", "load2", "load3",
              "air0", "air1", "air2", "air3")}
        cols[case] = c
    return meta, cols


def static(c, fs):
    m = (c["fl_s"] >= fs - STATIC_FROM) & (c["fl_s"] <= fs - STATIC_TO)
    return float(c["load0"][m].mean()), float(c["fl_z"][m].mean())


def cleat(c, fs, flen, dt):
    """The front-left wheel over a cleat: peak, its sample count, first rise, body rise,
    airborne time, the deepest pitch."""
    st, z0 = static(c, fs)
    w = (c["fl_s"] >= fs - PEAK_BEFORE) & (c["fl_s"] <= fs + flen + PEAK_AFTER)
    wz = (c["fl_s"] >= fs - PEAK_BEFORE) & (c["fl_s"] <= fs + flen + RISE_AFTER)
    load = c["load0"][w]
    over = np.nonzero(load > RISE * st)[0]
    return {"static_N": round(st, 1), "peak_N": round(float(load.max()), 1),
            "samples_over_2x": int((load > 2 * st).sum()),
            "first_rise_m": None if not len(over) else round(float(c["fl_s"][w][over[0]] - fs), 3),
            "body_rise_m": round(float(c["fl_z"][wz].max() - z0), 4),
            "airborne_s": round(float(c["air0"][wz].sum() * dt), 3),
            "pitch_max_deg": round(math.degrees(float(np.abs(c["pitch"][wz]).max())), 2)}


def pothole(c, fs, flen, dt):
    st, _ = static(c, fs)
    w = (c["fl_s"] >= fs - PEAK_BEFORE) & (c["fl_s"] <= fs + flen + PEAK_AFTER)
    t, load = c["t"][w], c["load0"][w]
    hole = (c["fl_s"][w] >= fs) & (c["fl_s"][w] <= fs + flen + 0.4)
    i = int(np.nonzero(hole)[0][np.argmin(load[hole])]) if hole.any() else int(np.argmin(load))
    after = (t > t[i]) & (t <= t[i] + FAR_EDGE_S)
    j = int(np.nonzero(after)[0][np.argmax(load[after])]) if after.any() else None
    out = {"static_N": round(st, 1), "min_N": round(float(load[i]), 1),
           "min_share": round(float(load[i] / st), 3)}
    if j is not None:
        out.update(peak_N=round(float(load[j]), 1), peak_after_s=round(float(t[j] - t[i]), 4),
                   samples_over_2x=int((load > 2 * st).sum()))
    return out


def landings(c, dt):
    """Each landing of any wheel: (wheel, t, spread s)."""
    out = []
    for k in range(4):
        load, air, t = c[f"load{k}"], c[f"air{k}"], c["t"]
        for i in range(1, len(load)):
            if load[i - 1] <= LANDED_N < load[i]:
                back = t[i] - dt + air[i] * dt  # within the step, after its airborne share
                j = i
                while j + 1 < len(load) and load[j + 1] > load[j]:
                    j += 1
                out.append((k, round(float(t[i]), 4), round(float(t[j] - dt / 2 - back), 4)))
    return out


def washboard(c, fs, flen):
    w = (c["fl_s"] >= fs) & (c["fl_s"] <= fs + flen)
    return {"min_N": round(float(c["load0"][w].min()), 1),
            "max_N": round(float(c["load0"][w].max()), 1), "samples": int(w.sum())}


def judge(meta, cols):
    dt = 1.0 / meta["physics_hz"]
    res, ok = [], True
    for info in meta["cases"]:
        case = info["case"]
        name, base = info["fixture"], case.split("#")[0]
        r = {"case": case}
        if not info.get("start_ok") or case not in cols:
            r.update(verdict="refused", message=f"start check: {info.get('start_speed')} m/s, "
                                                f"{info.get('start_offset')} m")
            ok = False
            res.append(r)
            continue
        c, fs, flen = cols[case], info["fixture_s"], info["fixture_len"]
        ref = REF.get(base, {})
        items = {}
        if name.startswith("cleat"):
            m = cleat(c, fs, flen, dt)
            r.update(m)
            if info.get("offset") is None:
                items["peak"] = within(m["peak_N"], ref["peak"])
                items["body_rise"] = within(m["body_rise_m"], ref["rise"])
                items["ahead_of_hub"] = m["first_rise_m"] is not None and m["first_rise_m"] < 0
            else:
                items["no_miss"] = m["peak_N"] >= 2 * m["static_N"]
                items["no_throw"] = (m["body_rise_m"] <= THROW * ref["rise"]
                                     and m["pitch_max_deg"] <= THROW_PITCH)
        elif name == "pothole":
            m = pothole(c, fs, flen, dt)
            r.update(m)
            items["unloads"] = m["min_share"] < UNLOAD
            items["far_edge"] = "peak_N" in m and m["peak_after_s"] <= FAR_EDGE_S \
                and within(m["peak_N"], ref["peak"])
        elif name == "washboard":
            m = washboard(c, fs, flen)
            r.update(m)
            items["min"] = within(m["min_N"], ref["range"][0])
            items["max"] = within(m["max_N"], ref["range"][1])
        land = landings(c, dt)
        r["landings"] = land
        items["landings"] = all(LANDING_S[0] <= s <= LANDING_S[1] for _, _, s in land)
        r["items"] = items
        r["verdict"] = "passed" if all(items.values()) else "failed"
        ok = ok and r["verdict"] == "passed"
        res.append(r)
    return {"verdict": "passed" if ok else "failed", "query": meta["query"],
            "aib": meta.get("aib", ""), "physics_hz": meta["physics_hz"], "cases": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = judge(*read(a.run))
    for r in out["cases"]:
        print(json.dumps({k: v for k, v in r.items() if k != "landings"}
                         | {"landings": len(r.get("landings", []))}))
    print(json.dumps({"verdict": out["verdict"], "query": out["query"], "aib": out["aib"]}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0 if out["verdict"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
