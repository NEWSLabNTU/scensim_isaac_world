"""086 C6c1: a `fixture_drive --fixtures c6c` run judged against the quarter car with R5's cams
and R2's damper (`spikes/c6c_reference.py`, computed before the runs), the front-left wheel's
tire load at our contact's inner steps (`inner.csv`, 480 Hz at 60 Hz).

- washboard at each speed: the load's amplitude, (max - min) / 2 with the hub from WB_IN m past
  the washboard's start to WB_IN m before its end (its ramps are two 0.7 m waves), within TOL
  of the reference's at that speed; and the amplitudes rising to their plateau as the
  reference's do: the largest at 13-15 Hz or above (v / 0.7 m), none after it more than TOL
  under it (reported);
- class_c at 10 and 20 m/s: the load's spectrum (`welch_peak`) with the hub from CC_IN m into
  the profile to CC_IN m before its end; its peak in 4-40 Hz within PEAK_HZ of the reference's
  and its height over its 4-6 Hz level within TOL of the reference's.

A case whose start check failed (`start_ok`) is refused, not judged, and fails the run.

    python3 -m scensim_isaac_world.course.c6c_check <run dir> [--json out.json]
"""
import argparse
import csv
import json

import numpy as np

TOL = 0.30
PEAK_HZ = 2.0
WB_IN = 2.0
CC_IN = 10.0
SEG_S = 2.048  # s, Welch's segment (1024 samples at 500 Hz)
BAND = (4.0, 40.0)
LOW = (4.0, 6.0)
# The reference (spikes/c6c_reference.py at siw 086-c6c, before any run): washboard amplitude
# (N) by speed (m/s); class C peak (Hz) and its height over the 4-6 Hz level by speed.
REF_WB = {5.0: 1495.8, 7.0: 2520.6, 8.0: 3313.6, 9.0: 3882.9, 10.0: 3923.0, 12.0: 3383.9,
          14.0: 3045.8, 20.0: 2573.7}
REF_CC = {10.0: (9.77, 1.66), 20.0: (12.7, 2.24)}


def welch_peak(load, rate):
    """Welch's method on `load` (sampled at `rate` Hz) less its mean: Hann windows of SEG_S s,
    50 % overlap, averaged periodograms. Returns (peak Hz in BAND, its height over the mean of
    the LOW bins, the frequencies, the spectrum)."""
    y = np.asarray(load, dtype=float)
    y = y - y.mean()
    seg = int(round(SEG_S * rate))
    if len(y) < seg:
        raise ValueError(f"{len(y)} samples, under one {seg}-sample segment")
    win = np.hanning(seg)
    P = np.mean([np.abs(np.fft.rfft(y[i:i + seg] * win)) ** 2
                 for i in range(0, len(y) - seg + 1, seg // 2)], axis=0)
    f = np.fft.rfftfreq(seg, 1.0 / rate)
    m = (f > BAND[0]) & (f < BAND[1])
    k = int(np.argmax(P[m]))
    lo = (f > LOW[0]) & (f < LOW[1])
    return float(f[m][k]), float(P[m][k] / P[lo].mean()), f, P


def amplitude(load):
    load = np.asarray(load, dtype=float)
    return float((load.max() - load.min()) / 2.0)


def judge(run_dir, ref_wb=None, ref_cc=None, dt=1.0 / 60.0):
    ref_wb = REF_WB if ref_wb is None else ref_wb
    ref_cc = REF_CC if ref_cc is None else ref_cc
    cases = json.load(open(f"{run_dir}/cases.json"))["cases"]
    rows = {}
    with open(f"{run_dir}/inner.csv") as f:
        r = csv.reader(f)
        head = next(r)
        n_in = len(head) - 3
        for row in r:
            rows.setdefault(row[0], []).append((float(row[2]), [float(x) for x in row[3:]]))
    rate = n_in / dt
    out, ok = {"cases": {}, "rate_hz": rate}, True
    wb = {}
    for c in cases:
        name, speed = c["fixture"], float(c["speed"])
        res = {"start_ok": c.get("start_ok", False)}
        out["cases"][c["case"]] = res
        if not res["start_ok"]:
            res["verdict"] = "refused"
            ok = False
            continue
        fs, flen = c["fixture_s"], c["fixture_len"]
        pad = WB_IN if name == "washboard" else CC_IN
        load = [x for s, ls in rows.get(c["case"], []) if fs + pad <= s <= fs + flen - pad
                for x in ls]
        if name == "washboard":
            a = amplitude(load)
            ref = ref_wb.get(speed)
            res.update(amplitude_N=round(a, 1), ref_N=ref, hz=round(speed / 0.7, 1))
            res["pass"] = ref is not None and abs(a - ref) <= TOL * ref
            wb[speed] = a
        else:
            hz, height, _, _ = welch_peak(load, rate)
            ref = ref_cc.get(speed)
            res.update(peak_hz=round(hz, 2), height=round(height, 2), ref=ref)
            res["pass"] = ref is not None and abs(hz - ref[0]) <= PEAK_HZ \
                and abs(height - ref[1]) <= TOL * ref[1]
        ok = ok and res["pass"]
    if wb:  # reported: the plateau's shape
        top = max(wb, key=wb.get)
        out["washboard_largest_at_hz"] = round(top / 0.7, 1)
        out["washboard_after_largest_min_share"] = round(
            min(wb[v] for v in wb if v >= top) / wb[top], 3)
    out["verdict"] = "pass" if ok else "fail"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = judge(a.run)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
