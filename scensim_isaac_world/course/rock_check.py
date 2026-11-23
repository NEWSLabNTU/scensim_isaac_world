"""086 C6b: a `fixture_drive --fixtures c6b` run judged by C6b's checks (roadmap 086, "C6b's
rocks"), and the placement on the built course.

From the run's `fixture.csv` (one row a PhysX step) and `cases.json`, the front-left wheel's:

- embedded_15: its excess impulse J (C6a3's measure, the hub from 1 m before the rock to 1 m
  past it) within TOL of R0's quarter car at 4 kHz over the same rock's top along the track
  (REF_J, computed before the runs: the rock as `rocks.fixed_rock` makes it, its tyre
  envelope); the wheel's centre rising at least WHEEL_RISE and the body pitching at least
  PITCH (each against its mean 8 to 2 m before); no step at PhysX's rigid limit (any wheel's
  jounce within RIGID of its travel). The peak load (at the run's rate) is reported beside
  R4's 57.5 kN, not judged.
- loose_15, loose_8: the case's rock moves at least MOVED and is not lost (its centre within
  LOST_R of where it was first seen, and its z above that less LOST_Z); its largest speed at
  most the car's; no rigid-limit step.
- boulder_35: the car high-centres: under STOP_V within STOP_S of the front axle reaching the
  boulder's face, the front axle then past the face; no rigid-limit step. The largest
  deceleration is reported.
- placement (from the map's road record, offline): no placed rock on the carriageway, and
  each strip's placed cover (rocks' plan area over the strip's area) within a factor of COVER
  of its k.

A case whose start check failed is refused. The rocks in the data, in the stage and awake are
reported per case (133: measured, not inferred).

    python3 -m scensim_isaac_world.course.rock_check <run dir> [--map <map dir>] [--json out]
"""
import argparse
import csv
import json
import math
import sys

import numpy as np

TOL = 0.30
WHEEL_RISE = 0.05  # m
PITCH = 0.3  # deg
RIGID = 0.001  # m
MOVED = 0.05  # m
LOST_R, LOST_Z = 20.0, 0.1  # m
STOP_V, STOP_S = 0.5, 3.0  # m/s, s
COVER = 2.0
# R0's quarter car (C6a's constants) at 11.11 m/s over embedded_15 as placed: J and peak
REF_J, REF_PEAK = 653.9, 25482.6
R4_PEAK = 57.5e3


def read(run):
    with open(f"{run}/cases.json") as f:
        meta = json.load(f)
    by = {}
    with open(f"{run}/fixture.csv") as f:
        for r in csv.DictReader(f):
            if r["load0"] != "":
                by.setdefault(r["case"], []).append(r)
    return meta, by


def col(rs, k):
    return np.array([float(r[k]) if r[k] != "" else np.nan for r in rs])


def rigid_steps(rs, px):
    j = np.stack([col(rs, f"jounce{i}") for i in range(4)], axis=1)
    return int((j >= px - RIGID).any(axis=1).sum())


def pre(x, s, face):
    m = (s >= face - 8) & (s <= face - 2)
    return float(np.nanmean(x[m]))


def rocks_seen(rs):
    return {"loaded_max": int(np.nanmax(col(rs, "rocks_loaded"))),
            "prims_max": int(np.nanmax(col(rs, "rock_prims"))),
            "awake_max": int(np.nanmax(col(rs, "rocks_awake")))}


def embedded(rs, info, dt, px):
    from scensim_isaac_world import rocks as rk

    u = info["fixture_s"]
    half = rk.fixed_rock(0.15, 0.1).scale / 2  # its long axis along the road
    s, load = col(rs, "fl_s"), col(rs, "load0")
    st = pre(load, s, u - half)
    w = (s >= u - half - 1) & (s <= u + half + 1)
    wide = (s >= u - half - 1) & (s <= u + half + 10)
    zu, pitch = col(rs, "zu0"), np.degrees(col(rs, "pitch"))
    out = {"J": round(float(np.clip(load[w] - st, 0, None).sum() * dt), 1),
           "peak_N": round(float(load[w].max()), 1),
           "wheel_rise_m": round(float(np.nanmax(zu[wide]) - pre(zu, s, u - half)), 4),
           "pitch_deg": round(float(np.nanmax(np.abs(pitch[wide] - pre(pitch, s, u - half)))), 3),
           "rigid_steps": rigid_steps(rs, px)}
    items = {"J": abs(out["J"] - REF_J) <= TOL * REF_J,
             "wheel_rises": out["wheel_rise_m"] >= WHEEL_RISE,
             "body_pitches": out["pitch_deg"] >= PITCH,
             "no_rigid_limit": out["rigid_steps"] == 0}
    out.update(ref_J=REF_J, ref_peak_N=REF_PEAK, r4_peak_N=R4_PEAK)
    return out, items


def loose(rs, info, px):
    x, y, z = col(rs, "rock_x"), col(rs, "rock_y"), col(rs, "rock_z")
    seen = np.nonzero(~np.isnan(x))[0]
    if not len(seen):
        return {"rock": "never seen", "rigid_steps": rigid_steps(rs, px)}, {"rock_seen": False}
    i0, i1 = seen[0], seen[-1]
    moved = float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]))
    far = float(np.nanmax(np.hypot(x[seen] - x[i0], y[seen] - y[i0])))
    vmax = float(np.nanmax(col(rs, "rock_speed")))
    car = float(np.nanmax(np.abs(col(rs, "speed"))))
    out = {"moved_m": round(moved, 3), "farthest_m": round(far, 3),
           "z_drop_m": round(float(z[i0] - np.nanmin(z[seen])), 3),
           "rock_speed_max": round(vmax, 3), "car_speed_max": round(car, 3),
           "rigid_steps": rigid_steps(rs, px)}
    items = {"moves": moved >= MOVED,
             "not_lost": far <= LOST_R and out["z_drop_m"] <= LOST_Z,
             "no_faster_than_the_car": vmax <= car,
             "no_rigid_limit": out["rigid_steps"] == 0}
    return out, items


def boulder(rs, info, dt, px):
    from scensim_isaac_world import rocks as rk

    u = info["fixture_s"]
    face = u - rk.fixed_rock(0.35, 0.2).scale / 2
    t, s, v = col(rs, "t"), col(rs, "fl_s"), col(rs, "speed")
    reach = np.nonzero(s >= face - 0.6)[0]  # the plate's front, 0.6 m behind the front axle
    out = {"face_s": round(face, 3), "rigid_steps": rigid_steps(rs, px),
           "decel_max": round(float(-np.nanmin(col(rs, "ax"))), 2),
           "stuck_at_s": info.get("stuck_at_s")}
    if not len(reach):
        return out, {"reaches": False}
    t0 = t[reach[0]]
    after = (t >= t0) & (t <= t0 + STOP_S)
    stop = np.nonzero(after & (np.abs(v) < STOP_V))[0]
    out.update(stopped_after_s=round(float(t[stop[0]] - t0), 3) if len(stop) else None,
               front_axle_at_stop=round(float(s[stop[0]]), 3) if len(stop) else None)
    items = {"high_centres": bool(len(stop)) and s[stop[0]] > face - 0.6,
             "no_rigid_limit": out["rigid_steps"] == 0}
    return out, items


def placement(map_dir):
    """Each strip's placed cover over the course, from the map's road record (offline)."""
    import os

    from scensim_isaac_world import road_detail as rd
    from scensim_isaac_world import road_layer
    from scensim_isaac_world import rocks as rk

    c, = road_layer.load(os.path.join(map_dir, "course", road_layer.FILE))
    d = rd.Detail([c], workers=0)
    area, n = {}, {}
    fixed = {(round(r[0], 3), round(r[1], 3)) for r in c.rocks.at}
    t = d.tile
    xs, ys = c.centreline[:, 0], c.centreline[:, 1]
    for i in range(int(np.floor((xs.min() - 10) / t)), int(np.floor((xs.max() + 10) / t)) + 1):
        for j in range(int(np.floor((ys.min() - 10) / t)), int(np.floor((ys.max() + 10) / t)) + 1):
            for rock, p, yaw, lo in rk.place(d, i, j):
                u, v, _ = c.uv(np.array([p[0]]), np.array([p[1]]))
                if (round(float(u[0]), 3), round(float(v[0]), 3)) in fixed:
                    continue
                z = c.rocks.zone(c.section, v[0], u[0], c.stations)
                a, b = rock.scale / 2, rock.scale * rk.AXES[1] / 2  # plan semi-axes
                area[z] = area.get(z, 0.0) + math.pi * a * b
                n[z] = n.get(z, 0) + 1
    # each strip's area over the course, both sides, from the record at 1 m along it
    st = c.stations
    us = np.arange(0.0, c.length, 1.0)
    strip = {"carriageway": 0.0, "shoulder": 0.0, "runoff": 0.0, "ditch": 0.0}
    ext = c.section.extent()
    for side in (0, 1):
        w = np.interp(us, st.u, st.common["half_width"])
        sh = np.interp(us, st.u, st.side["shoulder"][side])
        ro = np.interp(us, st.u, st.side["runoff_len"][side])
        strip["carriageway"] += float(w.sum())
        strip["shoulder"] += float(sh.sum())
        strip["runoff"] += float(ro.sum())
        strip["ditch"] += float(np.clip(ext - (w + sh + ro), 0, None).sum())
    out = {}
    for z, a_ in strip.items():
        k = getattr(c.rocks, z)
        cover = area.get(z, 0.0) / a_ if a_ > 0 else 0.0
        out[z] = {"k": k, "placed_cover": round(cover, 5), "rocks": n.get(z, 0),
                  "area_m2": round(a_, 1)}
    items = {"none_on_the_carriageway": n.get("carriageway", 0) == 0,
             "cover_within_2x": all(1 / COVER <= v["placed_cover"] / v["k"] <= COVER
                                    for z, v in out.items() if v["k"] > 0 and v["area_m2"] > 0)}
    return out, items


def judge(meta, by, map_dir=None):
    dt = 1.0 / meta["physics_hz"]
    px = meta["px_travel"]
    res, ok = [], True
    for info in meta["cases"]:
        case, name = info["case"], info["fixture"]
        r = {"case": case}
        rs = by.get(case, [])
        if not info.get("start_ok") or not rs:
            r.update(verdict="refused", message=f"start check: {info.get('start_speed')} m/s")
            res.append(r)
            ok = False
            continue
        if name == "embedded_15":
            m, items = embedded(rs, info, dt, px)
        elif name.startswith("loose"):
            m, items = loose(rs, info, px)
        else:
            m, items = boulder(rs, info, dt, px)
        items = {k: bool(v) for k, v in items.items()}
        r.update(m, items=items, rocks=rocks_seen(rs),
                 verdict="passed" if all(items.values()) else "failed")
        ok = ok and r["verdict"] == "passed"
        res.append(r)
    out = {"query": meta["query"], "aib": meta.get("aib", ""), "physics_hz": meta["physics_hz"],
           "cases": res}
    if map_dir:
        p, items = placement(map_dir)
        items = {k: bool(v) for k, v in items.items()}
        out["placement"] = dict(p, items=items,
                                verdict="passed" if all(items.values()) else "failed")
        ok = ok and out["placement"]["verdict"] == "passed"
    out["verdict"] = "passed" if ok else "failed"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--map")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = judge(*read(a.run), map_dir=a.map)
    for r in out["cases"]:
        print(json.dumps(r))
    if "placement" in out:
        print(json.dumps({"placement": out["placement"]}))
    print(json.dumps({"verdict": out["verdict"], "query": out["query"], "aib": out["aib"]}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0 if out["verdict"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
