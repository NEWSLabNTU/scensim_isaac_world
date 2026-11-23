"""086 C5b: Isaac's road against a course's record, by PhysX's scene queries (Isaac's Python).

    python -m scensim_isaac_world.course.isaac_check <map dir> [--json out.json]

Opens the course's world (`<map dir>/isaac/world.usdc`), lets PhysX cook it, and casts a ray
straight down at every point of `check.grid` (every station and midway between, across the
paved width and beside each breakline). The road's height there is the hit's; the record's is
the centreline's height plus the section's. The same tolerances as `check`: 5 mm in height,
lip faces left out, and 0.1 deg in each half carriageway's cross slope. A ray that hits no
road, or hits something else first, is counted as missing.
"""
import argparse
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map_dir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    from aib import app_exit
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    code = 1  # a check that fails is a failed run, as one that raises
    try:
        out = run(a.map_dir)
        text = json.dumps(out, indent=1)
        print(text, flush=True)
        if a.json:
            with open(a.json, "w") as f:
                f.write(text + "\n")
        code = 0 if out["pass"] else 1
    except BaseException as e:  # app.close() would end the process with 0 (scensim issue 118)
        code = app_exit.code_of(e) or 1
    finally:
        app_exit.close(app, code)  # its status: Kit's close would end the process with 0
    return code


def run(map_dir):
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from pxr import UsdPhysics

    from .. import road_detail as rd
    from . import check, record

    work = os.path.join(map_dir, "course")
    c = dict(np.load(os.path.join(work, "centreline.npz")))
    m = np.load(os.path.join(work, "meshes.npz"))
    section = rd.StationSection(record.load(work))
    half = float(m["half_width"])

    omni.usd.get_context().open_stage(os.path.join(map_dir, "isaac", "world.usdc"))
    stage = omni.usd.get_context().get_stage()
    if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    world.reset()
    world.step(render=False)
    query = omni.physx.get_physx_scene_query_interface()

    seen = {}

    def heights(u, v):
        x, y, z, nx, ny, _, _ = check._interp_line(c, u)
        px, py = x + v * nx, y + v * ny
        h = np.full(len(u), np.nan)
        for k in range(len(u)):
            hit = query.raycast_closest((float(px[k]), float(py[k]), float(z[k]) + 2.0),
                                        (0.0, 0.0, -1.0), 4.0)
            what = str(hit.get("collision", "")) if hit and hit.get("hit") else "nothing"
            if what.endswith("/Road"):
                h[k] = float(hit["position"][2])
            else:
                seen.setdefault((round(float(u[k]), 2), round(float(v[k]), 3)), what)
        return h

    u, v, kind = check.grid(c, section, half)
    want = check.record_height(c, section, u, v)
    got = heights(u, v)
    skip = check.face_band(section, u, v, check.FACE_SKIP)
    err = got - want
    missing = int(np.sum(np.isnan(got) & ~skip))
    ok = ~skip & ~np.isnan(got)
    s = np.asarray(c["s"], float) - float(c["s"][0])
    us = np.concatenate([s, 0.5 * (s[1:] + s[:-1])])
    sk = np.concatenate([np.zeros(len(s), int), np.ones(len(s) - 1, int)])
    rec_at = lambda uu, vv: check.record_height(c, section, uu, vv)  # noqa: E731
    ds = check.slopes(c, section, heights, us) - check.slopes(c, section, rec_at, us)
    out = {"points": int(len(u)), "skipped_on_a_face": int(skip.sum()), "missing": missing}
    miss = [(uu, vv, w) for (uu, vv), w in seen.items()]
    out["missing_v"] = sorted({vv for _, vv, _ in miss})[:20]
    out["missing_hit"] = sorted({w for _, _, w in miss})
    out["missing_first"] = miss[:10]
    for name, k in (("stations", 0), ("midway", 1)):
        e = np.abs(err[ok & (kind == k)])
        d = np.abs(ds[:, sk == k])
        j = int(np.argmax(np.where(ok & (kind == k), np.abs(err), -1)))
        out[name] = {"height_max_mm": round(float(e.max()) * 1e3, 3),
                     "height_p99_mm": round(float(np.percentile(e, 99)) * 1e3, 3),
                     "height_ok": bool(e.max() <= check.HEIGHT_TOL),
                     "slope_max_deg": round(float(np.nanmax(d)), 4),
                     "slope_ok": bool(np.nanmax(d) <= check.SLOPE_TOL_DEG),
                     "worst_at": {"u": round(float(u[j]), 2), "v": round(float(v[j]), 3)}}
    out["pass"] = (missing == 0 and all(out[n]["height_ok"] and out[n]["slope_ok"]
                                        for n in ("stations", "midway")))
    return out


if __name__ == "__main__":
    sys.exit(main())
