"""How far the world's own road render goes under the road detail (scensim issue 168).

The road layer draws its detail as its own render mesh (`road_layer.MeshStream`), the base road
plus the detail, and the world's own road mesh stays drawn under it. A ray meets whichever is
higher, so wherever the detail is below the base road (a pothole, a rut, the lower half of the
roughness) the lidar and the cameras saw the base road instead. The world server therefore
draws the world's road (its render only; its collision stays) lowered by the deepest detail below
the base, plus MARGIN, so the detail mesh is above it wherever the mesh is drawn.

The depth per corridor:
- `base_drop: {depth, sha256}` in the corridor's entry of road_detail.yaml, sampled once by this
  module's CLI over every road sample the corridor's tiles hold (exact);
- else, or when the entry or its stations file changed since (the sha256 differs), the features'
  bound (`bound`): the deepest pothole, the ruts' depth and 3 sd of the roughness, with a WARN
  naming the CLI. A stale exact value is worse than a safe bound.

    python -m scensim_isaac_world.base_drop <map dir | road_detail.yaml> [--write]

Pure numpy and yaml; no Isaac.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys

import numpy as np

from . import road_detail as rd

log = logging.getLogger("scensim_isaac_world")

KEY = "base_drop"
RENDER_KEY = "base_render"  # top level: the prims whose render is lowered (default: `render_paths`)
MARGIN = 0.01  # m below the deepest detail
SIGMAS = 3.0   # the bound's share of the roughness
CLI = "python -m scensim_isaac_world.base_drop <map> --write"


def entry_hash(entry, base_dir):
    """sha256 of a corridor's entry in road_detail.yaml (every key but `base_drop`, as canonical
    JSON) and of the bytes of its stations file, when it names one."""
    h = hashlib.sha256()
    body = {k: v for k, v in (entry or {}).items() if k != KEY}
    h.update(json.dumps(body, sort_keys=True, default=str).encode())
    st = body.get("stations")
    if st:
        with open(os.path.join(base_dir, str(st)), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def roughness_sd(r):
    """The roughness's sd, m: its line PSD's integral over the band the road detail makes
    (1 / (2 TILE) to N_MAX cycles/m)."""
    n = np.geomspace(1.0 / (2 * rd.TILE), rd.N_MAX, 4000)
    p = r.psd(n)
    return float(math.sqrt(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(n))))


def bound(c):
    """The features' bound on corridor `c`'s deepest detail below the base, m (>= 0): the deepest
    pothole (its range's top, or a fixed one's), the ruts' depth, SIGMAS sd of the roughness, the
    deepest point of its random profiles, and the section's fall where the world's road does not
    carry it. Washboard, cleats and water bars are above the base."""
    d = 0.0
    if c.potholes is not None:
        fixed = [float(p[2]) for p in (c.potholes.at or ())]
        poisson = [float(c.potholes.depth[1])] if len(getattr(c, "_holes", ())) else []
        d += max(fixed + poisson + [0.0])
    if c.ruts is not None:
        d += float(c.ruts.depth)
    if c.roughness is not None:
        d += SIGMAS * roughness_sd(c.roughness)
    lows = [-float(np.min(p.h(np.arange(p.u0, p.u0 + p.length, p.du)))) for p in c.profiles]
    return d + max(lows + [0.0]) + section_low(c)


def section_low(c, n=64):
    """How far the cross-section goes below the centreline over the carriageway and shoulders,
    m (>= 0), on a corridor whose world road does not carry it (`section_in_base` false: the
    detail mesh draws the crown and the shoulders' fall, the world's road is level); 0 when it
    does."""
    if c.section_in_base:
        return 0.0
    sec = c.section
    if hasattr(sec, "side"):  # a StationSection: the widest shoulder of either side
        shoulder = float(np.max(sec.side["shoulder"]))
    else:
        shoulder = float(getattr(sec, "shoulder", 0.0))
    lo = 0.0
    for u in np.linspace(0.0, c.length, n):
        w = float(sec.carriageway(u)) + shoulder
        v = np.linspace(-w, w, 81)
        lo = min(lo, float(np.min(sec.height_uv(np.full_like(v, u), v))))
    return -lo


def sampled(c, reach=30.0):
    """Corridor `c`'s deepest detail below the base, m (>= 0), over every carriageway and
    shoulder sample of every tile within `reach` m of its centreline: what its render mesh
    draws there."""
    d = rd.Detail([c], workers=0)
    xs, ys = c.centreline[:, 0], c.centreline[:, 1]
    lo = 0.0
    for (i, j) in sorted(k for k in d.tiles_near(xs, ys, reach) if d.touches(*k)):
        m = d.zones_tile(i, j) == rd.Detail.ROAD
        if m.any():
            lo = min(lo, float(d.tile_of(i, j)[m].min()))
    return -lo


def entries(path):
    """road_detail.yaml's corridor entries, as written."""
    import yaml

    with open(path) as f:
        return list((yaml.safe_load(f) or {}).get("corridors") or [])


def depths(path, corridors):
    """Per corridor, (depth m, where it came from: "stored" or "bound"), with a WARN for each
    that falls back to the bound."""
    base_dir = os.path.dirname(os.path.abspath(path))
    out = []
    for n, (c, e) in enumerate(zip(corridors, entries(path))):
        stored = (e or {}).get(KEY)
        if isinstance(stored, dict) and stored.get("sha256") == entry_hash(e, base_dir):
            out.append((float(stored["depth"]), "stored"))
            continue
        why = "none stored" if not stored else "stale (the corridor or its stations changed)"
        b = bound(c)
        log.warning("road detail %s corridor %d: base_drop %s; the features' bound %.3f m used "
                    "(issue 168: run `%s` for the exact depth)", path, n, why, b, CLI)
        out.append((b, "bound"))
    return out


def drop(path, corridors):
    """How far the world's road render goes down, m: the deepest corridor's depth plus MARGIN."""
    ds = depths(path, corridors)
    return (max(d for d, _ in ds) if ds else 0.0) + MARGIN


def render_paths(path, ground):
    """The prims whose render is lowered: road_detail.yaml's `base_render` (`[]`: none), else
    those of the world's ground prims (`ground`) named `Road` (the course builder's road mesh,
    `/World/env/Road`). The terrain stays: the detail does not cover it. So do the lane lines:
    drawn a little above the road, they stay visible where the detail is at or below them, as
    before."""
    import yaml

    with open(path) as f:
        given = (yaml.safe_load(f) or {}).get(RENDER_KEY)
    if given is not None:  # an empty list: nothing lowered (the world's road drawn as built)
        return [str(p) for p in given]
    return [p for p in ground if p.rstrip("/").rsplit("/", 1)[-1] == "Road"]


def write(path, values):
    """Write each corridor's `base_drop: {depth, sha256}` into road_detail.yaml in place,
    keeping every other line (its comments too): an existing `base_drop:` line in the entry is
    replaced, else one is added as the entry's first key. `values`: per corridor (depth,
    sha256)."""
    with open(path) as f:
        lines = f.read().split("\n")
    # the corridors' list items: "- " at the list's indent after the top-level `corridors:`
    top = next(i for i, ln in enumerate(lines) if re.match(r"^corridors\s*:", ln))
    starts, indent = [], None
    for i in range(top + 1, len(lines)):
        ln = lines[i]
        if ln and not ln.startswith((" ", "-", "#")):
            break  # the next top-level key
        m = re.match(r"^(\s*)- ", ln)
        if m and (indent is None or len(m.group(1)) == indent):
            indent = len(m.group(1))
            starts.append(i)
    if len(starts) != len(values):
        raise ValueError(f"{path}: {len(starts)} corridor entries found, {len(values)} values")
    ends = starts[1:] + [next((i for i in range(starts[-1] + 1, len(lines))
                               if lines[i] and not lines[i].startswith((" ", "-", "#"))),
                              len(lines))]
    key_indent = " " * (indent + 2)
    for (s, e), (depth, sha) in reversed(list(zip(zip(starts, ends), values))):
        new = f"{key_indent}{KEY}: {{depth: {depth:.4f}, sha256: {sha}}}  # issue 168, by {CLI}"
        at = next((i for i in range(s, e) if re.match(rf"^\s*(- )?{KEY}\s*:", lines[i])), None)
        if at is not None:
            if at == s:  # the entry's first key: keep its "- "
                new = " " * indent + "- " + new.lstrip()
            lines[at] = new
        else:
            first = lines[s]
            rest = first[indent + 2:]
            lines[s] = " " * indent + "- " + new.lstrip()
            lines.insert(s + 1, key_indent + rest)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    from . import road_layer

    ap = argparse.ArgumentParser(prog="base_drop", description=__doc__.split("\n\n")[0])
    ap.add_argument("map", help="a map directory (its road_detail.yaml) or a road_detail.yaml")
    ap.add_argument("--write", action="store_true",
                    help="write each corridor's sampled depth and hash into the file")
    a = ap.parse_args(argv)
    path = road_layer.find(a.map)
    corridors = road_layer.load(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    values = []
    for n, (c, e) in enumerate(zip(corridors, entries(path))):
        d = sampled(c)
        values.append((d, entry_hash(e, base_dir)))
        print(f"{path} corridor {n}: deepest detail below the base {d:.4f} m (the features' "
              f"bound {bound(c):.4f} m)")
    if a.write:
        write(path, values)
        print(f"wrote {KEY} into {path}; the world's road render drops "
              f"{max(v[0] for v in values) + MARGIN:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
