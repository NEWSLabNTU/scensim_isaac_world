"""086 C5b: a course's road mesh, and the road layer's tiles, against its record.

    python3 -m scensim_isaac_world.course.check <map dir> [--layer] [--json out.json]

The grid: at every station of the record and midway between each two; across the paved width
every `ACROSS` m and `NEAR` m each side of every breakline (the crown line, the carriageway's
edges, the foot of each lip's face). At each point the record's road is the centreline's height
plus the section's (`road_detail.StationSection.height_uv`, the road layer's own interpolation
between stations), and the mesh's is the height of the road mesh's triangle under it (its quads
split from the first corner to the third, as PhysX's cooking and the renderer split them).

- height: within `HEIGHT_TOL` m; points within `FACE_SKIP` m of a lip's face in plan are left out
  (a face is a step, its height there is either side's);
- cross slope: each half carriageway's, from `EDGE_IN` m off the crown line to `EDGE_IN` m in
  from its edge, within `SLOPE_TOL_DEG`;
- with `--layer`, the road layer's tiles (`road_detail.Detail`, the record as a corridor of the
  map's road detail, no roughness) against the section's height within `LAYER_TOL` m (C5a's
  check), a tile cell and a half from a lip's face and from the paved edge, where the record's
  backslope back to the ground begins (a kink a tile's cell rounds by up to 7 mm).

The same grid, given to Isaac's scene queries, is `isaac_check.py`'s (the GPU half).
"""
import argparse
import json
import os
import sys

import numpy as np

ACROSS = 0.25
NEAR = 0.02
HEIGHT_TOL = 0.005
SLOPE_TOL_DEG = 0.1
EDGE_IN = 0.10
FACE_SKIP = 0.01
LAYER_TOL = 2e-3


def _interp_line(c, u):
    """The centreline's point, height and left normal at u (linear between stations, the
    normals as the ribbon's)."""
    from .geometry import _normals, is_closed

    s = np.asarray(c["s"], float) - float(c["s"][0])
    nrm = _normals(c["x"], c["y"], is_closed(c))
    i = np.clip(np.searchsorted(s, u, side="right") - 1, 0, len(s) - 2)
    t = (u - s[i]) / (s[i + 1] - s[i])
    lerp = lambda a: a[i] + t * (a[i + 1] - a[i])  # noqa: E731
    return lerp(c["x"]), lerp(c["y"]), lerp(c["z"]), lerp(nrm[:, 0]), lerp(nrm[:, 1]), i, t


def grid(c, section, half):
    """The check's points: (u, v, kind) arrays; kind 0 a station, 1 midway."""
    from .geometry import section_rows

    s = np.asarray(c["s"], float) - float(c["s"][0])
    us = np.concatenate([s, 0.5 * (s[1:] + s[:-1])])
    kind = np.concatenate([np.zeros(len(s), int), np.ones(len(s) - 1, int)])
    base = np.arange(-half, half + 1e-9, ACROSS)
    rows = section_rows(section, us, half, -half, 2 * half)  # the breaklines alone, per u
    U, V, K = [], [], []
    for r in rows[1:-1]:
        for d in (-NEAR, 0.0, NEAR):
            U.append(us)
            V.append(np.clip(r + d, -half, half))
            K.append(kind)
    for v in base:
        U.append(us)
        V.append(np.full(len(us), v))
        K.append(kind)
    return np.concatenate(U), np.concatenate(V), np.concatenate(K)


def mesh_height(c, pts, faces, u, v, half):
    """The road mesh's height under each (u, v), its triangles split 0-2 (nan: none under)."""
    x, y, _, nx, ny, i, _ = _interp_line(c, u)
    px, py = x + v * nx, y + v * ny
    n = len(c["x"])
    rows = len(pts) // n
    out = np.full(len(u), np.nan)
    for di in (0, -1, 1):  # the point's station segment first, then its neighbours
        seg = np.clip(i + di, 0, n - 2)
        todo = np.isnan(out)
        if not todo.any():
            break
        for r in range(rows - 1):
            q = np.stack([r * n + seg, r * n + seg + 1, (r + 1) * n + seg + 1,
                          (r + 1) * n + seg], axis=1)[todo]
            P = [pts[q[:, k]] for k in range(4)]
            for a, b, cc in ((P[0], P[1], P[2]), (P[0], P[2], P[3])):
                z = _bary(a, b, cc, px[todo], py[todo])
                hit = ~np.isnan(z)
                idx = np.flatnonzero(todo)[hit]
                out[idx] = np.where(np.isnan(out[idx]), z[hit], out[idx])
    return out


def _bary(a, b, c, x, y):
    """z of triangle abc at (x, y), nan outside it (a hair's tolerance)."""
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
    l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
    l3 = 1.0 - l1 - l2
    e = -1e-9
    inside = ok & (l1 >= e) & (l2 >= e) & (l3 >= e)
    return np.where(inside, l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2], np.nan)


def face_band(section, u, v, pad):
    """Whether (u, v) lies within `pad` m of a lip's face in plan (a side with a lip there)."""
    f = section._fields(u, v)  # noqa: SLF001
    face = np.where(f["lip_angle"] >= 90.0, 0.0,
                    f["lip_h"] / np.tan(np.radians(np.minimum(f["lip_angle"], 89.999))))
    x = np.abs(v) - f["half_width"]
    return (f["lip_h"] > 0) & (x >= -pad) & (x <= face + pad)


def record_height(c, section, u, v):
    _, _, z, _, _, _, _ = _interp_line(c, u)
    return z + section.height_uv(u, v)


def slopes(c, section, heights_at, u):
    """Each half carriageway's cross slope (deg, left then right), from heights_at(u, v)."""
    off = np.interp(u, section.u, section.common["crown_offset"])
    w = section.carriageway(u)
    out = []
    for sign in (1.0, -1.0):
        a = off + sign * EDGE_IN
        b = sign * (w - EDGE_IN)
        out.append(np.degrees(np.arctan((heights_at(u, b) - heights_at(u, a)) / np.abs(b - a))))
    return np.stack(out)


def check_mesh(c, m, section):
    half = float(m["half_width"])
    pts, faces = m["road_points"], m["road_faces"]
    u, v, kind = grid(c, section, half)
    want = record_height(c, section, u, v)
    got = mesh_height(c, pts, faces, u, v, half)
    skip = face_band(section, u, v, FACE_SKIP)
    err = got - want
    missing = int(np.sum(np.isnan(got)))
    ok = ~skip & ~np.isnan(got)
    s = np.asarray(c["s"], float) - float(c["s"][0])
    us = np.concatenate([s, 0.5 * (s[1:] + s[:-1])])
    sk = np.concatenate([np.zeros(len(s), int), np.ones(len(s) - 1, int)])
    mesh_at = lambda uu, vv: mesh_height(c, pts, faces, uu, vv, half)  # noqa: E731
    rec_at = lambda uu, vv: record_height(c, section, uu, vv)  # noqa: E731
    ds = slopes(c, section, mesh_at, us) - slopes(c, section, rec_at, us)
    out = {"points": int(len(u)), "skipped_on_a_face": int(skip.sum()), "missing": missing}
    for name, k in (("stations", 0), ("midway", 1)):
        e = np.abs(err[ok & (kind == k)])
        d = np.abs(ds[:, sk == k])
        out[name] = {"height_max_mm": round(float(e.max()) * 1e3, 3),
                     "height_ok": bool(e.max() <= HEIGHT_TOL),
                     "slope_max_deg": round(float(np.nanmax(d)), 4),
                     "slope_ok": bool(np.nanmax(d) <= SLOPE_TOL_DEG)}
        j = int(np.argmax(np.where(ok & (kind == k), np.abs(err), -1)))
        out[name]["worst_at"] = {"u": round(float(u[j]), 2), "v": round(float(v[j]), 3)}
    out["pass"] = (missing == 0 and all(out[n]["height_ok"] and out[n]["slope_ok"]
                                        for n in ("stations", "midway")))
    return out


def check_layer(work, section, c, m):
    """The road layer's tiles of the record against the section's height (no roughness)."""
    from .. import road_detail as rd
    from .. import road_layer as rl
    from .record import NPZ

    half = float(m["half_width"])
    cor, = rl.parse({"corridors": [{"stations": NPZ, "seed": 1,
                                    "roughness": {"gd0": 1e-12}}]}, base=work)
    det = rd.Detail([cor])
    try:
        u, v, kind = grid(c, section, half)
        x, y, _, nx, ny, _, _ = _interp_line(c, u)
        got = det.height(x + v * nx, y + v * ny)
    finally:
        det.close()
    want = section.height_uv(u, v)
    # a tile cell from a face or from the paved edge, where the record's backslope begins
    # (1V:2H: a tile's bilinear cell rounds that kink by up to 7 mm)
    skip = face_band(section, u, v, rd.SPACING * 1.5) | (np.abs(v) > half - rd.SPACING * 1.5)
    err = np.abs(got - want)[~skip]
    k = kind[~skip]
    out = {"points": int(len(err))}
    for name, kk in (("stations", 0), ("midway", 1)):
        out[name + "_max_mm"] = round(float(err[k == kk].max()) * 1e3, 3)
    out["pass"] = bool(err.max() <= LAYER_TOL)
    return out


def run(map_dir, layer=False):
    from .. import road_detail as rd
    from . import record

    work = os.path.join(map_dir, "course")
    c = dict(np.load(os.path.join(work, "centreline.npz")))
    m = np.load(os.path.join(work, "meshes.npz"))
    section = rd.StationSection(record.load(work))
    out = {"mesh": check_mesh(c, m, section)}
    if layer:
        out["layer"] = check_layer(work, section, c, m)
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map_dir")
    ap.add_argument("--layer", action="store_true", help="the road layer's tiles too")
    ap.add_argument("--json", help="write the result here too")
    a = ap.parse_args(argv)
    out = run(a.map_dir, a.layer)
    text = json.dumps(out, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
