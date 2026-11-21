"""The minimal Lanelet2 loop of an authored field (scensim roadmap 091 W17; design
roboracer-indoor-twin.md 3.1).

SSv2 needs a Lanelet2 map even when every entity is placed by WorldPosition. An indoor field has
none, so this writes one: a single one-way lane along a closed centreline the user draws, in
lanelets of about `chunk` m sharing their end points, the last one ending on the first one's
first points.

    python -m scensim_isaac_world.loop <map dir> --centreline track.csv [--width 1.0]
        [--direction forward|reverse] [--chunk 5] [--speed-limit 10] [--origin 0,0]
    python -m scensim_isaac_world.loop <map dir> --prim /field/centreline    # a BasisCurves prim

The centreline: a CSV of `x,y` rows in the map frame (blank lines, `#` comments and a header
row are skipped), or a `BasisCurves` prim of the map's stage (`isaac/world.yaml`'s `usd`; its
first curve, in the stage's world frame, moved by `map_origin` into the map's). It is closed by
the tool: a last point equal to the first is dropped. The driving direction is the centreline's
order (`forward`) or the reverse; a route against it is impossible, so it is written in the
map's header. The lane is `width` m wide, its bounds `width / 2` either side; a bend tighter than
that half width would fold a bound over itself, and is refused, naming where.

The projection: TransverseMercator about `origin` (default 0, 0) with scale factor 1, as the
CARLA towns' maps declare, since SSv2 takes TransverseMercator and MGRS only (not Local). Each
node's latitude and longitude are the inverse of its map-frame position, so Autoware, SSv2 and
the simple pilot all compute the position back; `local_x`/`local_y` carry it too. The latitude
and longitude are synthetic, an indoor field placed at the origin by convention, and the map's
header and `map_projector_info.yaml` say so.

Writes `lanelet2_map.osm`, `map_projector_info.yaml`, `map_config.yaml` and `pointcloud_map.pcd`
in the map directory. The point cloud is the lane's flat surface on a `PCD_STEP` grid: SSv2's
map loader refuses a map directory with no `*.pcd`, and an authored field has none; it is not a
localisation map (a RoboRacer stack localises on its own occupancy map), and its header says
so.
The standard library only, but for `--prim` (pxr and PyYAML).
"""
import argparse
import math
import sys
from pathlib import Path

GENERATOR = "scensim_isaac_world.loop"
STEP = 0.5          # m between bound points
PCD_STEP = 0.5      # m between the lane's surface points in pointcloud_map.pcd
MIN_POINTS = 3

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
_E2 = WGS84_F * (2 - WGS84_F)
_EP2 = _E2 / (1 - _E2)
K0 = 1.0            # the scale factor the CARLA towns' maps declare


class LoopError(ValueError):
    """A centreline the loop cannot be made from, or a bad argument; the message says which."""


# --- the projection (acb's carla_map_gen/projection.py, as scensim_agent's simple_route) ------

def _meridian_arc(lat):
    e2, e4, e6 = _E2, _E2 ** 2, _E2 ** 3
    return WGS84_A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
                      - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat)
                      + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat)
                      - (35 * e6 / 3072) * math.sin(6 * lat))


class Projector:
    """Autoware's and SSv2's TransverseMercatorProjector: Snyder's series forward, Newton back."""

    def __init__(self, lat0=0.0, lon0=0.0, k0=K0):
        self.lat0, self.lon0, self.k0 = float(lat0), float(lon0), float(k0)

    def project(self, lat_deg, lon_deg):
        lat = math.radians(lat_deg)
        s, c = math.sin(lat), math.cos(lat)
        n = WGS84_A / math.sqrt(1 - _E2 * s * s)
        t = (s / c) ** 2
        cc = _EP2 * c * c
        a = math.radians(lon_deg - self.lon0) * c
        x = self.k0 * n * (a + (1 - t + cc) * a ** 3 / 6
                           + (5 - 18 * t + t * t + 72 * cc - 58 * _EP2) * a ** 5 / 120)
        y = self.k0 * (_meridian_arc(lat) - _meridian_arc(math.radians(self.lat0))
                       + n * (s / c) * (a * a / 2 + (5 - t + 9 * cc + 4 * cc * cc) * a ** 4 / 24
                                        + (61 - 58 * t + t * t + 600 * cc - 330 * _EP2)
                                        * a ** 6 / 720))
        return x, y

    def unproject(self, x, y):
        lat, lon = self.lat0 + y / 111000.0, self.lon0 + x / 111000.0
        h = 1e-7
        for _ in range(20):
            px, py = self.project(lat, lon)
            ex, ey = x - px, y - py
            if math.hypot(ex, ey) < 1e-7:
                break
            ax, ay = self.project(lat + h, lon)
            bx, by = self.project(lat, lon + h)
            j11, j12 = (ax - px) / h, (bx - px) / h
            j21, j22 = (ay - py) / h, (by - py) / h
            det = j11 * j22 - j12 * j21
            lat += (j22 * ex - j12 * ey) / det
            lon += (j11 * ey - j21 * ex) / det
        return lat, lon


# --- the centreline ---------------------------------------------------------------------------

def read_csv(path):
    """[(x, y)] of a CSV centreline in the map frame."""
    pts = []
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise LoopError(f"{path}: {e}") from None
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cells = [c.strip() for c in line.replace(";", ",").split(",")]
        try:
            x, y = float(cells[0]), float(cells[1])
        except (ValueError, IndexError):
            if not pts:
                continue  # a header row
            raise LoopError(f"{path}:{n}: not an x,y row: {line!r}") from None
        pts.append((x, y))
    return pts


def read_prim(map_dir, prim_path):
    """[(x, y)] of a BasisCurves prim of the map's stage, in the map frame."""
    import yaml
    from pxr import Usd, UsdGeom

    from . import names
    wy = Path(map_dir) / "isaac" / "world.yaml"
    try:
        w = yaml.safe_load(wy.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise LoopError(f"{wy}: {e}") from None
    usd = names.world_usd_of(str(wy), w)
    stage = Usd.Stage.Open(usd)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.BasisCurves):
        raise LoopError(f"{usd}: {prim_path} is not a BasisCurves prim")
    curves = UsdGeom.BasisCurves(prim)
    points = curves.GetPointsAttr().Get() or []
    counts = curves.GetCurveVertexCountsAttr().Get() or [len(points)]
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ox, oy = (list(w.get("map_origin") or (0.0, 0.0, 0.0)) + [0.0, 0.0])[:2]
    out = []
    for p in list(points)[:counts[0]]:
        q = xf.Transform(p)
        out.append((q[0] + float(ox), q[1] + float(oy)))  # map = world + map_origin
    return out


def _close(pts, where):
    pts = [tuple(map(float, p)) for p in pts]
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-6:
        pts = pts[:-1]
    clean = [pts[0]] if pts else []
    for p in pts[1:]:
        if math.dist(p, clean[-1]) > 1e-6:
            clean.append(p)
    if len(clean) < MIN_POINTS:
        raise LoopError(f"{where}: {len(clean)} distinct point(s); a loop needs {MIN_POINTS}")
    return clean


def _crossing(pts):
    """The first pair of non-adjacent segments of the closed polyline that cross, or None."""
    n = len(pts)

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # they share the closing point
            c, d = pts[j], pts[(j + 1) % n]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                return i, j
    return None


def _resample(pts, step):
    """Points every `step` m along the closed polyline (its last segment back to the first)."""
    ring = pts + [pts[0]]
    seg = [math.dist(ring[k], ring[k + 1]) for k in range(len(pts))]
    total = sum(seg)
    n = max(MIN_POINTS, round(total / step))
    out, k, acc = [], 0, 0.0
    for m in range(n):
        s = total * m / n
        while acc + seg[k] < s:
            acc += seg[k]
            k += 1
        u = (s - acc) / seg[k] if seg[k] else 0.0
        a, b = ring[k], ring[k + 1]
        out.append((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
    return out, total


def _bounds(pts, half):
    """(left, right) bound points `half` m either side of the closed centreline, along its order;
    refuses a bend tighter than `half`."""
    n = len(pts)
    left, right = [], []
    for k in range(n):
        p0, p1, p2 = pts[k - 1], pts[k], pts[(k + 1) % n]
        tx, ty = p2[0] - p0[0], p2[1] - p0[1]
        norm = math.hypot(tx, ty)
        nx, ny = -ty / norm, tx / norm  # left of the direction
        # the bend's radius at p1: the circumradius of p0, p1, p2
        a, b, c = math.dist(p0, p1), math.dist(p1, p2), math.dist(p0, p2)
        area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        if area2 > 1e-12 and a * b * c / (2 * area2) < half:
            raise LoopError(f"the centreline bends at radius {a * b * c / (2 * area2):.2f} m near "
                            f"({p1[0]:.2f}, {p1[1]:.2f}), tighter than the lane's half width "
                            f"{half:g} m: a bound would fold over itself")
        left.append((p1[0] + half * nx, p1[1] + half * ny))
        right.append((p1[0] - half * nx, p1[1] - half * ny))
    return left, right


# --- the map ----------------------------------------------------------------------------------

def build(centreline, width=1.0, direction="forward", chunk=5.0, speed_limit_kmh=10.0,
          where="centreline"):
    """The loop's geometry: {"left", "right", "centre", "cuts", "length"}, bounds along the
    driving direction, `cuts` the point indices where lanelets begin (the last ends on 0)."""
    if direction not in ("forward", "reverse"):
        raise LoopError(f"direction {direction!r}: forward (the centreline's order) or reverse")
    for name, v in (("width", width), ("chunk", chunk), ("speed limit", speed_limit_kmh)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise LoopError(f"{name} {v!r}: a positive number")
    pts = _close(centreline, where)
    if direction == "reverse":
        pts = pts[::-1]
    hit = _crossing(pts)
    if hit:
        raise LoopError(f"{where}: the centreline crosses itself (its segments {hit[0]} and "
                        f"{hit[1]})")
    pts, length = _resample(pts, STEP)
    if length < 2 * chunk:
        raise LoopError(f"{where}: a loop {length:.1f} m long; at least two lanelets of "
                        f"--chunk {chunk:g} m are needed")
    left, right = _bounds(pts, width / 2)
    per = max(2, round(chunk / (length / len(pts))))
    cuts = list(range(0, len(pts), per))
    if len(pts) - cuts[-1] < per / 2 and len(cuts) > 2:
        cuts.pop()  # a short last lanelet joins the one before
    return {"left": left, "right": right, "centre": pts, "cuts": cuts, "length": length}


def write(map_dir, geometry, width, direction, speed_limit_kmh=10.0, origin=(0.0, 0.0),
          source=""):
    """lanelet2_map.osm, map_projector_info.yaml and map_config.yaml in `map_dir`; returns the
    osm's path."""
    map_dir = Path(map_dir)
    proj = Projector(origin[0], origin[1])
    left, right, cuts = geometry["left"], geometry["right"], geometry["cuts"]
    nodes, ways, rels = [], [], []

    def node_line(points, first_id):
        ids = []
        for k, (x, y) in enumerate(points):
            lat, lon = proj.unproject(x, y)
            nodes.append((first_id + k, lat, lon, x, y))
            ids.append(first_id + k)
        return ids

    left_ids = node_line(left, 1)
    right_ids = node_line(right, 1 + len(left))
    white = {"type": "line_thin", "subtype": "solid", "color": "white"}
    tags = {"type": "lanelet", "subtype": "road", "location": "private", "one_way": "yes",
            "speed_limit": f"{speed_limit_kmh:g}"}
    bounds = list(zip(cuts, cuts[1:] + [len(left)]))
    wid, rid = 10 ** 6, 2 * 10 ** 6
    for a, b in bounds:
        idx = [k % len(left) for k in range(a, b + 1)]  # the last lanelet ends on the first point
        wid += 1
        ways.append((wid, [left_ids[k] for k in idx], white))
        wid += 1
        ways.append((wid, [right_ids[k] for k in idx], white))
        rid += 1
        rels.append((rid, wid - 1, wid, tags))
    header = (f"generated by {GENERATOR} (scensim 091 W17): one one-way lane, {len(rels)} "
              f"lanelets, {geometry['length']:.2f} m, width {width:g} m, direction {direction} "
              f"(the centreline's order: forward){'; centreline ' + source if source else ''}. "
              f"Latitude and longitude are synthetic: an indoor field placed at "
              f"({origin[0]:g}, {origin[1]:g}) by convention, a TransverseMercator with scale "
              f"factor 1; local_x/local_y are the map frame")
    osm = map_dir / "lanelet2_map.osm"
    with open(osm, "w") as f:
        f.write("<?xml version='1.0' encoding='UTF-8'?>\n")
        f.write(f"<!-- {header} -->\n")
        f.write(f'<osm version="0.6" generator="{GENERATOR}">\n')
        for n, lat, lon, x, y in nodes:
            f.write(f'  <node id="{n}" visible="true" version="1" lat="{lat:.11f}" '
                    f'lon="{lon:.11f}">\n    <tag k="local_x" v="{x:.4f}"/>\n'
                    f'    <tag k="local_y" v="{y:.4f}"/>\n    <tag k="ele" v="0.0000"/>\n'
                    "  </node>\n")
        for w, refs, t in ways:
            f.write(f'  <way id="{w}" visible="true" version="1">\n')
            for r in refs:
                f.write(f'    <nd ref="{r}"/>\n')
            for k, v in t.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </way>\n")
        for r, lw, rw, t in rels:
            f.write(f'  <relation id="{r}" visible="true" version="1">\n'
                    f'    <member type="way" ref="{lw}" role="left"/>\n'
                    f'    <member type="way" ref="{rw}" role="right"/>\n')
            for k, v in t.items():
                f.write(f'    <tag k="{k}" v="{v}"/>\n')
            f.write("  </relation>\n")
        f.write("</osm>\n")
    write_pcd(map_dir / "pointcloud_map.pcd", geometry["centre"], width)
    note = (f"# Generated by {GENERATOR} (scensim 091 W17). The latitude and longitude are synthetic:\n"
            f"# an indoor field placed at this origin by convention, not a real place.\n")
    (map_dir / "map_projector_info.yaml").write_text(
        note + "projector_type: TransverseMercator\nvertical_datum: WGS84\nmap_origin:\n"
        f"  latitude: {origin[0]}\n  longitude: {origin[1]}\n  altitude: 0.0\nscale_factor: 1.0\n")
    (map_dir / "map_config.yaml").write_text(
        note + "/**:\n  ros__parameters:\n    map_origin:\n"
        f"      latitude: {origin[0]}\n      longitude: {origin[1]}\n      elevation: 0.0\n"
        "      roll: 0.0\n      pitch: 0.0\n      yaw: 0.0\n")
    return osm


def write_pcd(path, centre, width, step=PCD_STEP):
    """The lane's flat surface (z 0) as an ASCII PCD of x, y, z: points every `step` m across
    the lane at every centreline point. Returns the point count."""
    pts = []
    n = len(centre)
    across = max(1, int(width / step))
    for k in range(n):
        p0, p1, p2 = centre[k - 1], centre[k], centre[(k + 1) % n]
        tx, ty = p2[0] - p0[0], p2[1] - p0[1]
        norm = math.hypot(tx, ty)
        nx, ny = -ty / norm, tx / norm
        for j in range(across + 1):
            o = -width / 2 + width * j / across
            pts.append((p1[0] + o * nx, p1[1] + o * ny, 0.0))
    with open(path, "w") as f:
        f.write(f"# {GENERATOR}: the lane's flat surface for SSv2's map loader, not a "
                "localisation map\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
                f"COUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\nDATA ascii\n")
        for x, y, z in pts:
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    return len(pts)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scensim_isaac_world.loop",
                                 description="The minimal Lanelet2 loop of an authored field.")
    ap.add_argument("map_dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--centreline", help="a CSV of x,y rows in the map frame")
    src.add_argument("--prim", help="a BasisCurves prim of the map's stage")
    ap.add_argument("--width", type=float, default=1.0, help="the lane's width, m (default 1.0)")
    ap.add_argument("--direction", default="forward", choices=("forward", "reverse"),
                    help="forward: the centreline's order (default); reverse")
    ap.add_argument("--chunk", type=float, default=5.0, help="m per lanelet (default 5)")
    ap.add_argument("--speed-limit", type=float, default=10.0, help="km/h (default 10)")
    ap.add_argument("--origin", default="0,0", help="lat,lon of the synthetic origin (default 0,0)")
    a = ap.parse_args(argv)
    try:
        try:
            origin = tuple(float(v) for v in a.origin.split(","))
            assert len(origin) == 2
        except (ValueError, AssertionError):
            raise LoopError(f"--origin {a.origin!r}: lat,lon") from None
        if not Path(a.map_dir).is_dir():
            raise LoopError(f"{a.map_dir}: not a directory")
        if a.centreline:
            pts, source = read_csv(a.centreline), str(a.centreline)
        else:
            pts, source = read_prim(a.map_dir, a.prim), a.prim
        g = build(pts, a.width, a.direction, a.chunk, a.speed_limit, where=source)
        osm = write(a.map_dir, g, a.width, a.direction, a.speed_limit, origin, source)
    except LoopError as e:
        print(f"loop: {e}", file=sys.stderr)
        return 2
    print(f"{osm}: {len(g['cuts'])} lanelets, {g['length']:.2f} m, direction {a.direction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
