"""086 C5c: each lip case of a `lip_drive` run classified against its control, and the strips'
surface classes checked.

A case's delay is the time its right front wheel's centre crosses the carriageway's edge after
its turn-in. Less the control's (segment 0, no lip): **re-enters** when that is REENTER_TOL s or
less, **held** when more or when the wheel has not crossed within HELD_WINDOW s of the turn-in.
Each is set beside the course's expected outcome (`cases:`, from the record's rule): an
agreement, a mismatch, or (`boundary`) neither. Reported, not judged: the peak yaw rate and
lateral acceleration in the SPIKE_WINDOW s after the crossing, less the control's, and the
body's roll range about it.

The check passes when every case ran and started right (`lip_drive`'s start check), the
control crossed within CONTROL_TOL s of where its line says (the base following it exactly),
and on the strips' segment the right wheels stood on the record's shoulder zone's class and
the left ones on none. A mismatch with the rule is a result, not a failure.

    python3 -m scensim_isaac_world.course.lip_check <run dir> <map dir> [--json out.json]
"""
import argparse
import csv
import json
import math
import sys

import numpy as np

REENTER_TOL = 0.3  # s
HELD_WINDOW = 4.0  # s
CONTROL_TOL = 0.3  # s
SPIKE_WINDOW = 1.0  # s
CONTROL = 0
STRIPS = 10


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out.setdefault(int(r["case"]), []).append(r)
    return out


def col(rows, k, cast=float):
    return np.array([cast(r[k]) for r in rows])


def crossing(rows, half, turn_t, start_t):
    """The first time at or after `start_t` the right front wheel's centre is on the
    carriageway (v >= -half); None if not within HELD_WINDOW s of `turn_t`."""
    t, fv = col(rows, "t"), col(rows, "fr_v")
    on = np.nonzero((t >= start_t) & (fv >= -half))[0]
    if len(on) == 0 or t[on[0]] > turn_t + HELD_WINDOW:
        return None
    i = on[0]
    if i == 0:
        return float(t[0])
    # linear between the frames either side
    f = (-half - fv[i - 1]) / (fv[i] - fv[i - 1])
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def path_crossing(case, half, tread, base, speed):
    """The delay the line itself gives: the time after the turn-in the right front wheel's
    centre reaches the edge with the base following the line exactly."""
    a = case["angle"]  # rad
    ds = np.arange(0.0, 60.0, 0.005)
    vb = np.minimum(case["v0"] + np.tan(a) * ds, case["v1"])
    psi = np.where(vb < case["v1"], a, 0.0)
    fv = vb + base * np.sin(psi) - tread / 2 * np.cos(psi)
    i = int(np.argmax(fv >= -half))
    return float(ds[i] / speed)


def window(rows, t0, t1, key):
    t, x = col(rows, "t"), col(rows, key)
    m = (t >= t0) & (t <= t1)
    return x[m]


def check(run, map_dir, out=None):
    from scensim_isaac_world.course import record

    st = record.load(f"{map_dir}/course")
    half = float(st.common["half_width"][0])
    rep = json.load(open(f"{run}/cases.json"))
    data = read(f"{run}/lip.csv")
    tread, base = rep["wheel_tread"], rep["wheel_base"]
    speed = rep["speed"]
    cases = {c["segment"]: c for c in rep["cases"]}
    res, problems = {"query": rep["query"], "cases": []}, []
    if CONTROL not in cases or CONTROL not in data:
        raise SystemExit("lip_check: no control (segment 0) in the run")

    def delay(k):
        c = cases[k]
        x = crossing(data[k], half, c["turn_t"], c["start_t"])
        return None if x is None else x - c["turn_t"], x

    d0, x0 = delay(CONTROL)
    want = path_crossing(dict(cases[CONTROL], angle=math.radians(rep["angle_deg"])), half,
                         tread, base, speed)
    res["control"] = {"delay_s": None if d0 is None else round(d0, 3),
                      "line_says_s": round(want, 3)}
    if d0 is None or abs(d0 - want) > CONTROL_TOL:
        problems.append(f"control: crossed {d0} s after the turn-in, its line says "
                        f"{want:.3f} (+-{CONTROL_TOL})")
    c0 = data[CONTROL]
    base_yaw = (np.abs(window(c0, x0, x0 + SPIKE_WINDOW, "yaw_rate")).max()
                if x0 is not None else math.nan)
    base_ay = (np.abs(window(c0, x0, x0 + SPIKE_WINDOW, "ay")).max()
               if x0 is not None else math.nan)
    agree = mismatch = boundary = 0
    for k in sorted(cases):
        c = cases[k]
        if not c.get("start_ok"):
            problems.append(f"case {k}: refused at its start (speed {c.get('start_speed')}, "
                            f"offset {c.get('start_offset')})")
        if not all(c.get("right_on_shoulder") or [False]):
            problems.append(f"case {k}: the right wheels not on the shoulder at the start "
                            f"({c.get('right_on_shoulder')})")
        d, x = delay(k)
        rel = None if (d is None or d0 is None) else d - d0
        got = "held" if rel is None or rel > REENTER_TOL else "re-enters"
        line = {"segment": k, "lip_h": c["lip_h"], "lip_angle": c["lip_angle"],
                "expect": c["expect"], "got": got,
                "delay_s": None if d is None else round(d, 3),
                "less_control_s": None if rel is None else round(rel, 3),
                "start_speed": c.get("start_speed"), "start_offset": c.get("start_offset"),
                "right_on_shoulder": c.get("right_on_shoulder")}
        if x is not None:
            rows = data[k]
            line["yaw_rate_peak_less_control"] = round(
                float(np.abs(window(rows, x, x + SPIKE_WINDOW, "yaw_rate")).max() - base_yaw), 4)
            line["ay_peak_less_control"] = round(
                float(np.abs(window(rows, x, x + SPIKE_WINDOW, "ay")).max() - base_ay), 3)
            roll = window(rows, x - 0.5, x + SPIKE_WINDOW, "roll")
            line["roll_range_deg"] = round(math.degrees(float(roll.max() - roll.min())), 3)
        if c["expect"] == "boundary":
            line["verdict"] = "boundary"
            boundary += 1
        elif c["expect"] == got:
            line["verdict"] = "agrees"
            agree += 1
        else:
            line["verdict"] = "mismatch"
            mismatch += 1
        res["cases"].append(line)
    if STRIPS in data:
        rows = [r for r in data[STRIPS]
                if cases[STRIPS]["start_t"] <= float(r["t"]) <= cases[STRIPS]["turn_t"]]
        right = {(r["w1_cls"], r["w1_zone"][:22]) for r in rows} | \
                {(r["w3_cls"], r["w3_zone"][:22]) for r in rows}
        left = {(r["w0_cls"], r["w0_zone"]) for r in rows} | {(r["w2_cls"], r["w2_zone"])
                                                               for r in rows}
        res["strips"] = {"right": sorted(map(list, right)), "left": sorted(map(list, left))}
        if right != {("gravel", "record_right_shoulder_")}:
            problems.append(f"strips: the right wheels read {sorted(right)}, not gravel on "
                            f"the record's shoulder zone")
        if any(z for _, z in left):
            problems.append(f"strips: the left wheels read a zone: {sorted(left)}")
    res.update(agree=agree, mismatch=mismatch, boundary=boundary, problems=problems,
               ok=not problems)
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("map")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = check(a.run, a.map, a.json)
    print(f"query {res['query']}: control {res['control']}")
    for c in res["cases"]:
        print(f"  {c['segment']:2d} {c['lip_h']:.2f} m {str(c['lip_angle']):>4} deg  "
              f"expect {c['expect']:9s} got {c['got']:9s} {c['verdict']:8s} "
              f"less control {c['less_control_s']} s  yaw+ {c.get('yaw_rate_peak_less_control')}"
              f"  ay+ {c.get('ay_peak_less_control')}  roll {c.get('roll_range_deg')}")
    print(f"agree {res['agree']}, mismatch {res['mismatch']}, boundary {res['boundary']}")
    for p in res["problems"]:
        print("PROBLEM", p)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
