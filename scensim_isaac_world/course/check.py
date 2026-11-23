"""086 C5b, C5d: a course's road mesh, the road layer's tiles and Autoware's maps, against its
record.

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

- Autoware's maps (086 C5d), when the map has them: every lanelet bound node's `ele`, and the
  point midway between each two consecutive nodes of a bound, within `HEIGHT_TOL` m of the
  road mesh there; the point cloud's count the build's, and a divided cloud's tiles each in
  its cell, the metadata naming every tile.

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


def osm_bounds(path):
    """The Lanelet2 map's bound linestrings: a list of (n, 3) arrays of each way's nodes'
    local_x, local_y and ele, in order (every way of the course's map is a bound)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    nodes = {}
    for n in root.iter("node"):
        t = {k.get("k"): k.get("v") for k in n.iter("tag")}
        nodes[n.get("id")] = (float(t["local_x"]), float(t["local_y"]), float(t["ele"]))
    return [np.array([nodes[r.get("ref")] for r in w.iter("nd")]) for w in root.iter("way")]


def check_lanelets(map_dir, c, m, section=None):
    """086 C5d: each lanelet bound node's `ele`, and the point midway between each two
    consecutive nodes of a bound (linear, as Lanelet2 reads it), against the road mesh's height
    there, within HEIGHT_TOL. A bound along a lip runs on its top, where the face is a step (a
    tenth of a millimetre in plan is the step's height): a point within FACE_SKIP of a face is
    compared with the mesh FACE_SKIP m inside it, toward the centreline."""
    from .. import road_detail as rd

    half = float(m["half_width"])
    cor = rd.Corridor(np.stack([c["x"], c["y"], c["z"]], 1))
    out = {}
    ways = osm_bounds(os.path.join(map_dir, "lanelet2_map.osm"))
    nodes = np.unique(np.concatenate(ways), axis=0)
    mids = np.concatenate([0.5 * (w[1:] + w[:-1]) for w in ways if len(w) > 1])
    for name, p in (("nodes", nodes), ("midway", mids)):
        u, v, _ = cor.uv(p[:, 0], p[:, 1])
        on_face = face_band(section, u, v, FACE_SKIP) if section is not None else \
            np.zeros(len(u), bool)
        v = np.where(on_face, v - np.sign(v) * FACE_SKIP, v)
        got = mesh_height(c, m["road_points"], m["road_faces"], u, v, half)
        err = np.abs(p[:, 2] - got)
        j = int(np.nanargmax(err))
        out[name] = {"points": int(len(p)), "on_a_lip": int(on_face.sum()),
                     "missing": int(np.isnan(got).sum()),
                     "height_max_mm": round(float(np.nanmax(err)) * 1e3, 3),
                     "worst_at": {"u": round(float(u[j]), 2), "v": round(float(v[j]), 3)}}
    out["pass"] = all(out[k]["missing"] == 0 and out[k]["height_max_mm"] <= HEIGHT_TOL * 1e3
                      for k in ("nodes", "midway"))
    return out


def check_cloud(map_dir):
    """086 C5d: a divided cloud's tiles each inside their TILE m cell, the metadata naming
    every tile and no other, and their points together the build's count (`course/maps.json`'s
    `pcd_points`). One undivided file: its count."""
    from . import maps

    path = os.path.join(map_dir, "pointcloud_map.pcd")
    want = None
    info = os.path.join(map_dir, "course", "maps.json")
    if os.path.exists(info):
        want = json.load(open(info)).get("pcd_points")
    if not os.path.isdir(path):
        n = len(maps.read_pcd(path))
        return {"divided": False, "points": n, "pass": want is None or n == want}
    import yaml

    meta = yaml.safe_load(open(os.path.join(map_dir, maps.METADATA)))
    res = float(meta.pop("x_resolution"))
    meta.pop("y_resolution")
    files = sorted(f for f in os.listdir(path) if f.endswith(".pcd"))
    problems = []
    if sorted(meta) != files:
        problems.append(f"metadata names {len(meta)} tiles, the directory holds {len(files)}")
    n = 0
    for f in files:
        p = maps.read_pcd(os.path.join(path, f)).astype(np.float64)
        n += len(p)
        x0, y0 = meta.get(f, (np.nan, np.nan))
        if not (np.all(p[:, 0] >= x0) and np.all(p[:, 0] < x0 + res)
                and np.all(p[:, 1] >= y0) and np.all(p[:, 1] < y0 + res)):
            problems.append(f"{f}: points outside its cell [{x0}, {y0}] + {res}")
    if want is not None and n != want:
        problems.append(f"the tiles hold {n} points, the build wrote {want}")
    return {"divided": True, "tiles": len(files), "points": n, "problems": problems,
            "pass": not problems}


def run(map_dir, layer=False):
    from .. import road_detail as rd
    from . import record

    work = os.path.join(map_dir, "course")
    c = dict(np.load(os.path.join(work, "centreline.npz")))
    m = np.load(os.path.join(work, "meshes.npz"))
    section = rd.StationSection(record.load(work))
    out = {"mesh": check_mesh(c, m, section)}
    if os.path.exists(os.path.join(map_dir, "lanelet2_map.osm")):  # Autoware's maps (C5d)
        out["lanelets"] = check_lanelets(map_dir, c, m, section)
        out["cloud"] = check_cloud(map_dir)
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
