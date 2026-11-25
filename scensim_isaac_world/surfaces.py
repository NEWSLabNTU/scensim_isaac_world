"""The surface layer's map and model (roadmap 047 S1; docs/design/off-road-surfaces.md).
Pure Python (numpy, yaml; Pillow for PNG rasters): no Isaac, so the check and the plot run
anywhere and the maths is unit-tested.

`<map>/surfaces.yaml` (047 D1) says what every piece of ground is:

    classes: surface_classes.yaml      # optional: a class table beside the map; default the
                                       # one every world server shares (scensim_world_server's)
    allow_unvalidated: [mud_wet]       # optional: classes whose soil parameters are a stand-in
                                       # (the table's `unvalidated:`) that soil zones here may
                                       # use; any other such zone is refused (047 D6)
    defaults: {Road: asphalt, RoadLine: asphalt, SideWalk: concrete, Terrain: grass,
               Ground: dirt, unlabelled: grass}
    zones:
      - id: sand_pit                   # [a-z0-9_]+: scenarios name it (scensim.zone.<id>.<field>)
        class: sand
        soil: grid                     # the soil layer (S3), not only friction: grid (or true:
                                       # ruts, multipass) or closed_form (no memory); the class
                                       # needs soil parameters
        ground_z: 0.0                  # optional, with soil: grid: the zone's ground is the
                                       # server's own tiles at this height (world frame), so the
                                       # ruts are in the collision; the map must have no ground
                                       # collider inside the polygon's bounding box
        polygon: [[120.0, -40.0], [160.0, -40.0], [160.0, -10.0], [120.0, -10.0]]
      - id: ford_1
        water_depth: raster/ford_1_depth.png    # or a number, m
        scale: 0.0001                  # m per pixel count (PNG; a .npy holds metres)
        origin: [300.0, 55.0]          # the raster's lower-left corner, map frame
        resolution: 0.25               # m per pixel

`defaults` maps SimReady's semantic labels to classes (Road, RoadLine, SideWalk, Terrain,
Ground; `unlabelled` for ground without one, default the reference class). A zone is a
map-frame polygon, or a raster's extent when it has no polygon; every field but `id` is
optional and falls through to the zone below it, the label's default and the world's road
condition. Zones listed later lie on top.

A wheel's friction, every physics step (`WheelSurfaces`):

    mu = tire_friction * friction_scale * mu_class * wet_class(wetness) * f_hydroplane(v, depth)
         / mu_reference

filtered over `blend_length` of travel: the class's peak. Then the tire (054 T1): PhysX's tire
builds force at its stiffness and saturates at friction x load near a slip of
2.3 mu F_z / C, flat beyond (054 step 4a). Per class, `peak_slip` and `peak_slip_angle` say
where the peak should be (absent: the rig's own), and each wheel's stiffness is scaled so
PhysX's lands there (`WheelSurfaces.stiffness`). Past the peak the friction written falls
with the combined slip rho = |(kappa / kappa_peak, tan alpha / tan alpha_peak)|, in both
directions, toward the class's `sliding` share: g = sliding + (1 - sliding) / rho, lowered
further with the sliding speed by an optional `stribeck`, and lagged over the class's
`relaxation` length of travel. The value written divides out the rig's friction-vs-slip graph,
which PhysX applies on top. Below the peak on dry reference ground (asphalt) that is the rig's
own tire_friction and stiffness.

    python -m scensim_isaac_world.surfaces check <map dir | surfaces.yaml>
    python -m scensim_isaac_world.surfaces plot <map dir | surfaces.yaml> [-o out.png]
"""
import argparse
import dataclasses
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
# the surface classes every world server shares (scensim roadmap 086 C2)
from scensim_world_server.classes import SURFACE_CLASSES

from . import soil as soil_model

HERE = Path(__file__).resolve().parent
DEFAULT_CLASSES = SURFACE_CLASSES
FILE = "surfaces.yaml"
WETNESS = ("dry", "moist", "wetWithPuddles", "lowFlooded", "highFlooded")
LABELS = ("Road", "RoadLine", "SideWalk", "Terrain", "Ground")
UNLABELLED = "unlabelled"
# SimReady's Wikidata Q-codes for CARLA's ground classes (as scensim_isaac_world.usd_world reads them).
QCODE_LABELS = {"Q34442": "Road", "Q177749": "SideWalk", "Q1392287": "RoadLine",
                "Q186131": "Terrain", "Q36133": "Ground"}
ZONE_ID = re.compile(r"^[a-z0-9_]+$")
# A zone's `soil:` (047 S3, D2): the grid (ruts, multipass) or the closed-form wheel (no memory).
SOIL_MODELS = {True: "grid", "grid": "grid", "closed_form": "closed_form", False: None,
               None: None}
ZONE_KEYS = {"id", "class", "wetness", "water_depth", "soil", "ground_z", "polygon", "scale",
             "origin", "resolution"}
# A scenario's RoadCondition properties (047 S2): `scensim.zone.<id>.<field>` sets a zone's field;
# names outside `scensim.` belong to other tools.
PROPERTY_PREFIX = "scensim."
ZONE_PROPERTY = re.compile(r"^scensim\.zone\.([^.]+)\.([^.]+)$")
ZONE_FIELDS = {"class": "cls", "wetness": "wetness", "water_depth": "water_depth"}
# `scensim.ruts` (047 D5): the soil zones' ruts at the start, a set saved beside the map
# (`<map>/ruts/<name>.npz`), or `fresh`. An instruction, not road state: a later
# RoadCondition without it leaves the ground as the wheels made it.
RUTS_PROPERTY = "scensim.ruts"
RUTS_DIR = "ruts"
RUTS_FRESH = "fresh"
RUTS_NAME = re.compile(r"^[a-z0-9_-]+$")
# The map of a world with no surfaces.yaml: every label the reference class, no zones.
NO_MAP = "(no surfaces.yaml)"
# `scensim.surface_map` (roadmap 056, 086 C4): the scenario's surface map, by name, from the
# map's `surfaces/<name>.yaml` (its state's first); `scensim.default.<Label>`: one ground
# label's class for this scenario (friction only).
SURFACE_MAP_PROPERTY = "scensim.surface_map"
SURFACES_DIR = "surfaces"
DEFAULT_PROPERTY = re.compile(r"^scensim\.default\.([^.]+)$")


class SurfaceError(ValueError):
    """A surface map or class table that cannot be used; the message lists every problem."""


# --- Class table ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceClass:
    name: str
    mu: float
    wet: tuple  # factor per WETNESS level
    hydroplaning: bool = False
    soil: object = None  # soil.SoilParams: the class can be a soil zone's (047 S3)
    unvalidated: str = None  # why its soil parameters are a stand-in (047 D6), or None
    calibration: str = None  # what its soil parameters were fitted to, measured (053)
    peak_slip: float = None  # longitudinal slip at the tire's peak; None: the rig's
    peak_slip_angle: float = None  # rad, the slip angle at the peak; None: the rig's
    sliding: float = 1.0  # the friction past the peak, sliding, as a share of the peak's
    relaxation: float = 0.0  # m of travel the drop past the peak lags by; 0: none
    stribeck: object = None  # Stribeck: the sliding share falling with sliding speed


@dataclass(frozen=True)
class Stribeck:
    speed: float  # m/s, the sliding speed the drop is spread over
    sliding: float  # the sliding share at high sliding speed


@dataclass(frozen=True)
class Hydroplaning:
    tire_pressure_kpa: float = 250.0
    onset: float = 0.5
    residual: float = 0.1
    depth_min: float = 0.0005
    depth_full: float = 0.0025

    @property
    def speed(self):
        """Horne's full hydroplaning speed, m/s."""
        return 6.35 * math.sqrt(self.tire_pressure_kpa) / 3.6


@dataclass(frozen=True)
class ClassTable:
    classes: dict
    reference: str
    wetness_depth: dict
    hydroplaning: Hydroplaning
    blend_length: float

    def __getitem__(self, name):
        return self.classes[name]


def _number(c, key, where, problems, lo, hi, default=None):
    v = c.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        problems.append(f"{where}: {key} must be a number in [{lo}, {hi}]")
        return default
    return float(v)


def _tire(c, where, problems):
    """A class's tire curve (054 T1): SurfaceClass's peak_slip, peak_slip_angle (degrees in
    the table), sliding, relaxation and stribeck."""
    out = {"peak_slip": _number(c, "peak_slip", where, problems, 0.01, 1.0),
           "sliding": _number(c, "sliding", where, problems, 0.05, 1.5, 1.0),
           "relaxation": _number(c, "relaxation", where, problems, 0.0, 10.0, 0.0)}
    angle = _number(c, "peak_slip_angle", where, problems, 0.5, 45.0)
    out["peak_slip_angle"] = None if angle is None else math.radians(angle)
    st = c.get("stribeck")
    if st is not None:
        if not isinstance(st, dict) or set(st) != {"speed", "sliding"}:
            problems.append(f"{where}: stribeck is {{speed: m/s, sliding: share}}")
        else:
            speed = _number(st, "speed", f"{where}: stribeck", problems, 0.01, 100.0)
            share = _number(st, "sliding", f"{where}: stribeck", problems, 0.05, 1.5)
            if speed is not None and share is not None:
                out["stribeck"] = Stribeck(speed, share)
    return out


def parse_classes(data, where="classes"):
    problems = []
    if not isinstance(data, dict) or not isinstance(data.get("classes"), dict):
        raise SurfaceError(f"{where}: no `classes:` mapping")
    classes = {}
    for name, c in data["classes"].items():
        if not isinstance(c, dict) or not isinstance(c.get("mu"), (int, float)) or c["mu"] <= 0:
            problems.append(f"{where}: class {name}: needs mu > 0")
            continue
        wet_in = c.get("wet") or {}
        unknown = set(wet_in) - set(WETNESS)
        if unknown:
            problems.append(f"{where}: class {name}: unknown wetness {sorted(unknown)} "
                            f"(one of {', '.join(WETNESS)})")
        wet, last = [], 1.0
        for level in WETNESS:
            last = float(wet_in.get(level, last if level != "dry" else 1.0))
            wet.append(last)
        if "slip" in c:
            problems.append(f"{where}: class {name}: `slip` is gone (054 T1): the tire's curve "
                            f"is peak_slip, peak_slip_angle, sliding, relaxation and stribeck")
        tire = _tire(c, f"{where}: class {name}", problems)
        params = None
        if c.get("soil") is not None:
            try:
                params = soil_model.parse_params(c["soil"], f"{where}: class {name}: soil")
            except soil_model.SoilError as e:
                problems.append(str(e))
        unvalidated = c.get("unvalidated")
        if unvalidated is not None and (not isinstance(unvalidated, str) or not unvalidated):
            problems.append(f"{where}: class {name}: unvalidated is the reason, a string")
            unvalidated = None
        calibration = c.get("calibration")
        if calibration is not None and (not isinstance(calibration, str) or not calibration):
            problems.append(f"{where}: class {name}: calibration is the reference, a string")
            calibration = None
        if params is not None and calibration is None:
            problems.append(f"{where}: class {name}: a soil class names its `calibration:`, "
                            f"the measurements its soil row was fitted to or why none (053)")
        classes[name] = SurfaceClass(name, float(c["mu"]), tuple(wet),
                                     bool(c.get("hydroplaning", False)), params, unvalidated,
                                     calibration, **tire)
    reference = data.get("reference", "asphalt")
    if reference not in classes:
        problems.append(f"{where}: reference class {reference!r} is not in the table")
    depth = {w: float((data.get("wetness_depth") or {}).get(w, 0.0)) for w in WETNESS}
    hp = Hydroplaning(**{k: float(v) for k, v in (data.get("hydroplaning") or {}).items()
                         if k in Hydroplaning.__dataclass_fields__})
    if problems:
        raise SurfaceError("\n".join(problems))
    return ClassTable(classes, reference, depth, hp, float(data.get("blend_length", 0.2)))


def load_classes(path=DEFAULT_CLASSES):
    import yaml

    with open(path) as f:
        return parse_classes(yaml.safe_load(f), str(path))


# --- Rasters and zones ---------------------------------------------------------------------


@dataclass
class Raster:
    """Values on a grid: row 0 is the top (+y) edge, column 0 the left (-x); `origin` is the
    lower-left corner in the map frame, `resolution` m per pixel. Sampled bilinearly at pixel
    centres; 0 outside."""

    values: np.ndarray
    origin: tuple
    resolution: float

    @property
    def extent(self):
        h, w = self.values.shape
        x0, y0 = self.origin
        return (x0, x0 + w * self.resolution, y0, y0 + h * self.resolution)

    def polygon(self):
        x0, x1, y0, y1 = self.extent
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def sample(self, x, y):
        h, w = self.values.shape
        u = (x - self.origin[0]) / self.resolution - 0.5
        v = (self.origin[1] + h * self.resolution - y) / self.resolution - 0.5  # row
        if u < -0.5 or v < -0.5 or u > w - 0.5 or v > h - 0.5:
            return 0.0
        u = min(max(u, 0.0), w - 1.0)
        v = min(max(v, 0.0), h - 1.0)
        c0, r0 = int(u), int(v)
        c1, r1 = min(c0 + 1, w - 1), min(r0 + 1, h - 1)
        fu, fv = u - c0, v - r0
        a = self.values
        top = a[r0, c0] * (1 - fu) + a[r0, c1] * fu
        bottom = a[r1, c0] * (1 - fu) + a[r1, c1] * fu
        return float(top * (1 - fv) + bottom * fv)


def load_raster(path, scale, origin, resolution):
    if str(path).endswith(".npy"):
        values = np.load(path).astype(np.float64)
    else:
        from PIL import Image

        with Image.open(path) as im:
            values = np.asarray(im, dtype=np.float64)
        if values.ndim == 3:
            values = values[..., 0]
        values = values * scale
    if values.ndim != 2:
        raise ValueError(f"{path}: not a one-channel image")
    return Raster(values, (float(origin[0]), float(origin[1])), float(resolution))


def inside(polygon, x, y):
    """Point in polygon (even-odd), `polygon` an (n, 2) array."""
    px, py = polygon[:, 0], polygon[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    crosses = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.count_nonzero(crosses & (x < xs)) % 2)


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def self_intersects(polygon):
    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return True
    return False


@dataclass
class Zone:
    id: str
    polygon: np.ndarray
    cls: str = None
    wetness: str = None
    water_depth: object = None  # float, Raster or None
    soil: str = None  # the soil layer's model there (SOIL_MODELS), or None: friction only
    ground_z: float = None  # the server's own ground here at this height (ruts), or None
    bbox: tuple = field(init=False)

    def __post_init__(self):
        lo, hi = self.polygon.min(axis=0), self.polygon.max(axis=0)
        self.bbox = (lo[0], hi[0], lo[1], hi[1])

    def contains(self, x, y):
        x0, x1, y0, y1 = self.bbox
        return x0 <= x <= x1 and y0 <= y <= y1 and inside(self.polygon, x, y)

    def depth_at(self, x, y):
        if isinstance(self.water_depth, Raster):
            return self.water_depth.sample(x, y)
        return self.water_depth


# --- The map -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """The ground at one point: its class, and the water on it (None: the world's)."""

    cls: str
    wetness: str = None
    water_depth: float = None
    zone: str = None  # the top zone there, if any
    soil: str = None  # the soil model of the zone that gave the class, if any
    soil_zone: str = None  # that zone, when it has one
    # 091 W13: an authored material's peak friction, absolute as a class's `mu` (the tire's on
    # that ground), on `cls` the reference class's curve; None: the class's
    mu: float = None
    # what decided the class: `zone`, or on an authored map `token`, `material` or `no_map`
    # (`authored.SOURCES`); on any other map `label`. A record, not the ground: two surfaces
    # that differ only here are the same ground
    source: str = field(default=None, compare=False)


@dataclass
class SurfaceMap:
    path: str
    table: ClassTable
    defaults: dict  # label (or UNLABELLED) -> class
    zones: list
    allow_unvalidated: tuple = ()  # classes with stand-in soil parameters soil zones may use
    # Where its rut sets are, in lookup order; the first is where the server saves. Empty:
    # `<map dir>/ruts` only. The world server sets `(<state>/ruts, <map dir>/ruts)` for a map
    # with a state directory (`names`, roadmap 076 A9).
    ruts_dirs: tuple = ()

    @property
    def unvalidated(self):
        """`(zone id, class, reason)` for every soil zone on stand-in parameters (047 D6)."""
        return [(z.id, z.cls, self.table[z.cls].unvalidated) for z in self.zones
                if z.soil and z.cls in self.table.classes and self.table[z.cls].unvalidated]

    def at(self, x, y, label=None, ground=None):
        """The surface at map point (x, y) over ground labelled `label` (None: unlabelled).
        `ground`: an authored map's `authored.Ground` of the collider there (091 W13), which
        decides where no zone gives the class: its token's class, or its material's friction on
        the reference class's curve, or (`no_map`) the label's default as without it."""
        cls = wetness = depth = top = None
        soil = soil_zone = None
        for z in reversed(self.zones):
            if not z.contains(x, y):
                continue
            top = top or z.id
            cls = cls or z.cls
            wetness = wetness or z.wetness
            if depth is None and z.water_depth is not None:
                depth = z.depth_at(x, y)
            if soil is None and z.cls is not None:
                soil = z.soil or False
                soil_zone = z.id if z.soil else None
            if cls and wetness and depth is not None:
                break
        mu = None
        if cls is not None:
            source = "zone"
        elif ground is not None and ground.source == "token":
            cls, source = ground.cls, "token"
        elif ground is not None and ground.source == "material":
            cls, mu, source = self.table.reference, ground.mu, "material"
        else:
            cls = self.defaults.get(label or UNLABELLED, self.defaults[UNLABELLED])
            source = "label" if ground is None else "no_map"
        return Surface(cls, wetness, depth, top, soil or None, soil_zone, mu, source)


def find(map_or_file):
    """`surfaces.yaml` for a map directory, or the file itself."""
    p = Path(map_or_file)
    return p / FILE if p.is_dir() else p


def parse(data, path=FILE, table=None, base=None):
    """A SurfaceMap from surfaces.yaml's contents; SurfaceError lists every problem."""
    base = Path(base or Path(path).parent)
    if not isinstance(data, dict):
        raise SurfaceError(f"{path}: not a mapping")
    problems = []
    unknown = set(data) - {"classes", "defaults", "zones", "allow_unvalidated"}
    if unknown:
        problems.append(f"{path}: unknown key(s) {sorted(unknown)} (classes, defaults, zones, "
                        f"allow_unvalidated)")
    if table is None:
        classes = data.get("classes")
        try:
            table = load_classes(base / classes if classes else DEFAULT_CLASSES)
        except (OSError, SurfaceError) as e:
            raise SurfaceError(f"{path}: class table: {e}") from e

    def known(cls, where):
        if cls not in table.classes:
            problems.append(f"{where}: unknown class {cls!r} (the table has "
                            f"{', '.join(sorted(table.classes))})")

    allow = data.get("allow_unvalidated") or []
    if not isinstance(allow, list) or not all(isinstance(c, str) for c in allow):
        problems.append(f"{path}: allow_unvalidated is a list of class names")
        allow = []
    for cls in allow:
        known(cls, f"{path}: allow_unvalidated")
    defaults = dict(data.get("defaults") or {})
    for label, cls in defaults.items():
        if label not in LABELS and label != UNLABELLED:
            problems.append(f"{path}: defaults: unknown label {label!r} "
                            f"({', '.join(LABELS)}, {UNLABELLED})")
        known(cls, f"{path}: defaults.{label}")
    defaults.setdefault(UNLABELLED, table.reference)
    for label in LABELS:
        defaults.setdefault(label, defaults[UNLABELLED])

    zones, ids = [], set()
    for n, z in enumerate(data.get("zones") or []):
        where = f"{path}: zones[{n}]"
        if not isinstance(z, dict):
            problems.append(f"{where}: not a mapping")
            continue
        zid = str(z.get("id", ""))
        where = f"{path}: zone {zid or n}"
        if not ZONE_ID.match(zid):
            problems.append(f"{where}: id must match [a-z0-9_]+ (scenarios name it)")
        elif zid in ids:
            problems.append(f"{where}: id used twice")
        ids.add(zid)
        unknown = set(z) - ZONE_KEYS
        if unknown:
            problems.append(f"{where}: unknown key(s) {sorted(unknown)}")
        if z.get("class") is not None:
            known(z["class"], where)
        if z.get("wetness") is not None and z["wetness"] not in WETNESS:
            problems.append(f"{where}: wetness {z['wetness']!r} is not one of {', '.join(WETNESS)}")
        depth = z.get("water_depth")
        raster = None
        if isinstance(depth, str):
            missing = [k for k in ("origin", "resolution") if k not in z]
            if not str(depth).endswith(".npy") and "scale" not in z:
                missing.append("scale")
            if missing:
                problems.append(f"{where}: a raster needs {', '.join(missing)}")
            else:
                try:
                    raster = load_raster(base / depth, float(z.get("scale", 1.0)), z["origin"],
                                         z["resolution"])
                    depth = raster
                except (OSError, ValueError) as e:
                    problems.append(f"{where}: raster {depth}: {e}")
        elif depth is not None:
            if not isinstance(depth, (int, float)) or depth < 0:
                problems.append(f"{where}: water_depth is metres >= 0, or a raster file")
            else:
                depth = float(depth)
        if "polygon" in z:
            try:
                poly = np.array(z["polygon"], dtype=np.float64)
            except (TypeError, ValueError):
                poly = np.zeros((0,))
            if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
                problems.append(f"{where}: polygon needs three or more [x, y] points")
                continue
            if self_intersects(poly):
                problems.append(f"{where}: polygon crosses itself")
        elif raster is not None:
            poly = raster.polygon()
        else:
            problems.append(f"{where}: needs a polygon (or a raster, whose extent it then is)")
            continue
        if not any(z.get(k) is not None for k in ("class", "wetness", "water_depth")):
            problems.append(f"{where}: sets none of class, wetness, water_depth")
        model = SOIL_MODELS.get(z.get("soil", False))
        if z.get("soil") not in SOIL_MODELS:
            problems.append(f"{where}: soil {z.get('soil')!r} is not one of true, grid, "
                            f"closed_form, false")
        elif model and not z.get("class"):
            problems.append(f"{where}: soil needs the zone's own class")
        elif model and z["class"] in table.classes and table[z["class"]].soil is None:
            have = ", ".join(sorted(n for n, c in table.classes.items() if c.soil)) or "none"
            problems.append(f"{where}: class {z['class']!r} has no soil parameters (soil "
                            f"classes: {have})")
        elif model and z["class"] in table.classes and table[z["class"]].unvalidated \
                and z["class"] not in allow:
            problems.append(f"{where}: class {z['class']!r} has unvalidated soil parameters "
                            f"({table[z['class']].unvalidated}): list it in allow_unvalidated "
                            f"to use them")
        ground_z = z.get("ground_z")
        if ground_z is not None:
            if not isinstance(ground_z, (int, float)) or not math.isfinite(ground_z):
                problems.append(f"{where}: ground_z is a height in metres, not {ground_z!r}")
            elif model != "grid":
                problems.append(f"{where}: ground_z (the ruts' ground) needs soil: grid")
        zones.append(Zone(zid, poly, z.get("class"), z.get("wetness"),
                          depth if isinstance(depth, (float, Raster)) else None, model,
                          float(ground_z) if isinstance(ground_z, (int, float)) else None))
    if problems:
        raise SurfaceError("\n".join(problems))
    return SurfaceMap(str(path), table, defaults, zones, tuple(allow))


def load(map_or_file):
    import yaml

    path = find(map_or_file)
    with open(path) as f:
        return parse(yaml.safe_load(f), str(path))


# --- Friction ------------------------------------------------------------------------------


@dataclass
class RoadCondition:
    """The world's road condition (OpenSCENARIO's RoadCondition, 047 S2): applied where no
    zone sets its own wetness."""

    friction_scale: float = 1.0
    wetness: str = "dry"
    ruts: str = None  # `scensim.ruts`: a saved set's name, RUTS_FRESH, or None (as they are)
    surface_map: str = None  # `scensim.surface_map`: the named map, or None (as loaded)


def road_condition(surface_map, friction_scale=1.0, wetness="dry", properties=None,
                   named=None):
    """A scenario's road condition on `surface_map`: `(RoadCondition, the map with the zone
    fields and label classes the properties set)`. The whole road state: zones it does not
    name are as the map has them. `scensim.surface_map` names another map, which `named(name)`
    returns (a SurfaceMap, or SurfaceError naming the maps there; None: this world has no
    named maps); the rest of the condition is then checked against that map. SurfaceError
    lists every problem; then nothing applies."""
    problems = []
    properties = dict(properties or {})
    map_name = properties.pop(SURFACE_MAP_PROPERTY, None)
    if map_name is not None:
        if not RUTS_NAME.match(str(map_name)):
            problems.append(f"{SURFACE_MAP_PROPERTY}: {map_name!r} is not a map name "
                            f"([a-z0-9_-]+, its {SURFACES_DIR}/<name>.yaml)")
        elif named is None:
            problems.append(f"{SURFACE_MAP_PROPERTY}: {map_name}: this world has no map "
                            f"directory to hold {SURFACES_DIR}/{map_name}.yaml")
        else:
            try:
                surface_map = named(str(map_name))
            except SurfaceError as e:
                problems.append(f"{SURFACE_MAP_PROPERTY}: {e}")
        if problems:  # nothing else can be checked against a map that is not there
            raise SurfaceError("\n".join(problems))
    defaults = {}
    wetness = wetness or "dry"
    if wetness not in WETNESS:
        problems.append(f"wetness {wetness!r} is not one of {', '.join(WETNESS)}")
    try:
        friction_scale = float(friction_scale)
    except (TypeError, ValueError):
        friction_scale = float("nan")
    if not (math.isfinite(friction_scale) and friction_scale >= 0):
        problems.append(f"frictionScaleFactor {friction_scale} is not in [0, inf)")
    zones = {z.id: z for z in surface_map.zones}
    overrides = {}
    ruts = None
    for name, value in sorted((properties or {}).items()):
        if not name.startswith(PROPERTY_PREFIX):
            continue
        if name == RUTS_PROPERTY:
            have = available_ruts(surface_map)
            if not any(z.soil == "grid" for z in surface_map.zones):
                problems.append(f"{name}: {surface_map.path} has no soil zone on the grid "
                                f"(soil: grid) to hold ruts")
            elif value != RUTS_FRESH and not RUTS_NAME.match(str(value)):
                problems.append(f"{name}: {value!r} is not a set name ([a-z0-9_-]+) or "
                                f"{RUTS_FRESH!r}")
            elif value != RUTS_FRESH and value not in have:
                where = " or ".join(map(str, ruts_dirs(surface_map)))
                problems.append(f"{name}: no {value}.npz in {where} (sets: "
                                f"{', '.join(have) or 'none'}; {RUTS_FRESH})")
            else:
                ruts = value
            continue
        d = DEFAULT_PROPERTY.match(name)
        if d:
            label = d.group(1)
            if label not in (*LABELS, UNLABELLED):
                problems.append(f"{name}: no ground label {label!r} (one of "
                                f"{', '.join((*LABELS, UNLABELLED))})")
            elif value not in surface_map.table.classes:
                problems.append(f"{name}: unknown class {value!r} (the table has "
                                f"{', '.join(sorted(surface_map.table.classes))})")
            else:
                defaults[label] = value  # a soil class: its friction (no soil under a label)
            continue
        m = ZONE_PROPERTY.match(name)
        if not m:
            problems.append(f"{name}: unknown property (scensim.zone.<id>.<field>, field one of "
                            f"{', '.join(ZONE_FIELDS)}; {RUTS_PROPERTY}; {SURFACE_MAP_PROPERTY};"
                            f" scensim.default.<Label>)")
            continue
        zid, key = m.groups()
        if zid not in zones:
            have = ", ".join(sorted(zones)) or "none"
            problems.append(f"{name}: no zone {zid!r} in {surface_map.path} (zones: {have})")
            continue
        if key not in ZONE_FIELDS:
            problems.append(f"{name}: a zone has no field {key!r} (one of "
                            f"{', '.join(ZONE_FIELDS)})")
            continue
        if key == "class":
            if value not in surface_map.table.classes:
                problems.append(f"{name}: unknown class {value!r} (the table has "
                                f"{', '.join(sorted(surface_map.table.classes))})")
                continue
            if zones[zid].soil:
                problems.append(f"{name}: zone {zid!r} is soil; its class is its soil's")
                continue
        elif key == "wetness":
            if value not in WETNESS:
                problems.append(f"{name}: {value!r} is not one of {', '.join(WETNESS)}")
                continue
        else:
            try:
                value = float(value)
            except ValueError:
                value = float("nan")
            if not (math.isfinite(value) and value >= 0):
                problems.append(f"{name}: water depth is metres >= 0, not {value}")
                continue
        overrides.setdefault(zid, {})[ZONE_FIELDS[key]] = value
    if problems:
        raise SurfaceError("\n".join(problems))
    changed = [dataclasses.replace(z, **overrides[z.id]) if z.id in overrides else z
               for z in surface_map.zones]
    return (RoadCondition(friction_scale, wetness, ruts, map_name),
            dataclasses.replace(surface_map, zones=changed,
                                defaults={**surface_map.defaults, **defaults}))


def named_maps(dirs):
    """The surface maps a scenario may name: `{name: file}` over `dirs` (`<state>/surfaces/`,
    then `<map dir>/surfaces/`), the first dir's file winning."""
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.yaml")) if Path(d).is_dir() else []:
            if RUTS_NAME.match(p.stem):
                out.setdefault(p.stem, p)
    return out


def load_named(dirs, name, ruts_roots=()):
    """Surface map `name` from `dirs` (`named_maps`); its rut sets are in `<root>/surfaces/
    <name>/ruts/` per root of `ruts_roots` (the state's, then the map directory). SurfaceError
    names the maps there when there is none, or lists the map's problems."""
    have = named_maps(dirs)
    if name not in have:
        where = " or ".join(str(Path(d) / f"{name}.yaml") for d in dirs) or "nowhere"
        raise SurfaceError(f"{name}: no {where} (maps: {', '.join(sorted(have)) or 'none'})")
    smap = load(have[name])
    return dataclasses.replace(smap, ruts_dirs=tuple(
        str(Path(r) / SURFACES_DIR / name / RUTS_DIR) for r in ruts_roots))


def ruts_dirs(surface_map):
    """Where a map's rut sets are, in lookup order: its `ruts_dirs`, else `<map dir>/ruts/`."""
    if surface_map.ruts_dirs:
        return [Path(d) for d in dict.fromkeys(surface_map.ruts_dirs)]
    return [Path(surface_map.path).parent / RUTS_DIR]


def ruts_dir(surface_map):
    """Where the server saves a map's ruts (`<dir>/runs/`) and where a set is copied to be
    named: the first of `ruts_dirs` (`<state>/ruts/`, or `<map dir>/ruts/`)."""
    return ruts_dirs(surface_map)[0]


def rut_set(surface_map, name):
    """The file of rut set `name`: the first `ruts_dirs` entry holding `<name>.npz`, else
    where it would be in `ruts_dir`."""
    for d in ruts_dirs(surface_map):
        if (d / f"{name}.npz").exists():
            return d / f"{name}.npz"
    return ruts_dir(surface_map) / f"{name}.npz"


def available_ruts(surface_map):
    """The names of the rut sets saved for the map (its state's and its own)."""
    if surface_map.path == NO_MAP:
        return []
    return sorted({p.stem for d in ruts_dirs(surface_map) for p in d.glob("*.npz")})


def hydroplane_factor(hp: Hydroplaning, speed, depth):
    """Grip left at `speed` (m/s) on `depth` (m) of standing water, 0..1."""
    if depth <= hp.depth_min:
        return 1.0
    w = min(1.0, (depth - hp.depth_min) / (hp.depth_full - hp.depth_min))
    vp = hp.speed
    r = min(1.0, max(0.0, (abs(speed) - hp.onset * vp) / ((1.0 - hp.onset) * vp)))
    return 1.0 - (1.0 - hp.residual) * w * r * r


def peak_mu(table: ClassTable, s: Surface, condition: RoadCondition, speed):
    """Peak friction on surface `s`, relative to dry reference ground (1.0 there)."""
    c = table[s.cls]
    wetness = s.wetness or condition.wetness
    level = WETNESS.index(wetness)
    depth = s.water_depth if s.water_depth is not None else table.wetness_depth[wetness]
    if depth > table.wetness_depth[wetness]:
        # Standing water at least as wet as the deepest level it reaches.
        level = max(level, max(i for i, w in enumerate(WETNESS)
                               if table.wetness_depth[w] <= depth))
    mu = c.mu * c.wet[level]
    if c.hydroplaning:
        mu *= hydroplane_factor(table.hydroplaning, speed, depth)
    return condition.friction_scale * mu / table[table.reference].mu


def applied_condition(surface_map, condition, zone_ids=()):
    """`ROAD_CONDITION_APPLIED` (092 K9f-a): the road state `surface_map` and `condition` (a
    SurfaceLayer's `map` and `condition` after `set_condition`) apply, as the world protocol's
    reply has it: the world-wide scale and wetness, and for each of `zone_ids` its class, its
    wetness (the world's where it sets none), its water depth (None: none of its own, or a
    raster, which is no one number) and mu: the class's mu times its wet factor at that
    wetness times the scale (`ZoneApplied.mu`; at rest: hydroplaning's speed factor and the
    standing water's level are not in it). A zone with no class of its own reads back class ""
    and mu 0: the ground's labels decide its class there, so there is none to read back. An
    id the map does not have is a SurfaceError (set_condition refused it already)."""
    zones = {z.id: z for z in surface_map.zones}
    missing = [z for z in zone_ids if z not in zones]
    if missing:
        raise SurfaceError(f"zone(s) {', '.join(missing)} not in {surface_map.path}")
    out = {}
    for zid in zone_ids:
        z = zones[zid]
        wetness = z.wetness or condition.wetness
        depth = z.water_depth if isinstance(z.water_depth, (int, float)) else None
        if z.cls is None:
            cls, mu = "", 0.0
        else:
            c = surface_map.table[z.cls]
            cls, mu = z.cls, c.mu * c.wet[WETNESS.index(wetness)] * condition.friction_scale
        out[zid] = {"surface_class": cls, "wetness": wetness, "water_depth": depth, "mu": mu}
    return {"friction_scale_factor": condition.friction_scale, "wetness": condition.wetness,
            "zones": out}


def graph_at(graph, slip):
    """PhysX's friction-vs-slip graph at |slip|: linear between its points, flat past them."""
    s = abs(slip)
    (x0, y0), (x1, y1), (x2, y2) = graph
    if s <= x0:
        return y0
    if s <= x1:
        return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    if s <= x2:
        return y1 + (y2 - y1) * (s - x1) / (x2 - x1)
    return y2


def sliding_share(c: SurfaceClass, sliding_speed):
    """Class `c`'s friction when sliding, as a share of its peak, at `sliding_speed` (m/s)."""
    if c.stribeck is None:
        return c.sliding
    fast = c.stribeck.sliding
    return fast + (c.sliding - fast) * math.exp(-(sliding_speed / c.stribeck.speed) ** 2)


def past_peak(rho, sliding):
    """The friction at combined slip `rho` (1 at the peak) as a share of the peak's."""
    return 1.0 if rho <= 1.0 else sliding + (1.0 - sliding) / rho


# Below this speed (m/s) the drop past the peak is not lagged (its time constant, the
# relaxation length over the speed, would hold it for ever at rest).
LAG_MIN_SPEED = 1.0


class WheelSurfaces:
    """One vehicle's wheels on a surface map: what each wheel's friction value and stiffness
    should be for the next physics step. `tire_friction` and `vehicle_graph` are the rig's;
    `rig_peaks` its tire's (longitudinal slip, tan slip angle) at the peak at friction 1.0
    (`surface_layer.rig_peaks`). A car whose axles have their own tyres (scensim roadmap 091
    W24) gives `tire_friction` and `rig_peaks` per wheel, as sequences of `wheels` items."""

    def __init__(self, surface_map: SurfaceMap, tire_friction, vehicle_graph, rig_peaks,
                 wheels=4):
        self.map = surface_map
        self.n = wheels
        self.tire_frictions = (list(tire_friction) if isinstance(tire_friction, (list, tuple))
                               else [tire_friction] * wheels)
        per_wheel = isinstance(rig_peaks[0], (list, tuple))
        self.wheel_peaks = ([tuple(p) for p in rig_peaks] if per_wheel
                            else [tuple(rig_peaks)] * wheels)
        if len(self.tire_frictions) != wheels or len(self.wheel_peaks) != wheels:
            raise ValueError(f"tire_friction and rig_peaks: one or {wheels} (one per wheel)")
        self.vehicle_graph = tuple(tuple(p) for p in vehicle_graph)
        self.reset()

    def reset(self):
        """Forget the filters (a teleport): the next update snaps to the ground."""
        self.mu = [None] * self.n  # the blended peak, relative to dry reference ground
        self.peaks = [None] * self.n  # the blended (kappa, tan alpha) at the peak
        self.g = [1.0] * self.n  # the lagged share of the peak past it
        self.stiffness = [(1.0, 1.0)] * self.n  # (lateral, longitudinal), shares of the rig's
        self.surfaces = [None] * self.n

    def class_peaks(self, c: SurfaceClass, i=0):
        """Class `c`'s (kappa, tan alpha) at the peak: its own, or wheel `i`'s rig tire's at
        its friction."""
        k_rig, t_rig = (p * self.tire_frictions[i] for p in self.wheel_peaks[i])
        return (c.peak_slip if c.peak_slip is not None else k_rig,
                math.tan(c.peak_slip_angle) if c.peak_slip_angle is not None else t_rig)

    def update(self, i, surface: Surface, condition: RoadCondition, speed, slip, dt,
               target=None, slip_angle=0.0):
        """Wheel `i` on `surface` (None: in the air, kept as it was) at forward `speed`
        (m/s) with longitudinal `slip` and `slip_angle` (rad) after a step of `dt` s: its
        friction value; `stiffness[i]` its tire's. `target`, relative to dry reference
        ground, replaces the surface's peak friction (the soil layer's, in a soil zone)."""
        if surface is None:
            surface = self.surfaces[i]
            if surface is None:
                return self.tire_frictions[i]
        c = self.map.table[surface.cls]
        if target is None and surface.mu is not None:
            # an authored material (091 W13): an absolute peak as a class's `mu` is, so realised
            # as written on a rig that realises the reference class's (the table's contract);
            # only the scenario's factor scales it, no wetness (an indoor floor)
            target = (condition.friction_scale * surface.mu
                      / self.map.table[self.map.table.reference].mu)
        elif target is None:
            target = peak_mu(self.map.table, surface, condition, speed)
        peaks = self.class_peaks(c, i)
        if self.mu[i] is None:
            self.mu[i], self.peaks[i] = target, peaks
        else:
            k = min(1.0, abs(speed) * dt / max(self.map.table.blend_length, 1e-6))
            self.mu[i] += (target - self.mu[i]) * k
            self.peaks[i] = tuple(a + (b - a) * k for a, b in zip(self.peaks[i], peaks))
        self.surfaces[i] = surface
        mu_peak = self.tire_frictions[i] * self.mu[i]
        kp, tp = self.peaks[i]
        # PhysX's peak is where 2.3 mu F_z / C meets the class's: C scaled by mu over the peak
        rk, rt = self.wheel_peaks[i]
        self.stiffness[i] = (rt * mu_peak / tp, rk * mu_peak / kp)
        t = math.tan(slip_angle)
        rho = math.hypot(slip / kp, t / tp)
        g = past_peak(rho, sliding_share(c, abs(speed) * math.hypot(slip, t)))
        if c.relaxation > 0 and abs(speed) >= LAG_MIN_SPEED:
            g = self.g[i] + (g - self.g[i]) * min(1.0, abs(speed) * dt / c.relaxation)
        self.g[i] = g
        return mu_peak * g / max(graph_at(self.vehicle_graph, slip), 1e-6)


# --- Check and plot ------------------------------------------------------------------------


def lanelet_ways(osm):
    """[(n, 2) array] per way of a Lanelet2 map, from its nodes' local_x / local_y."""
    import xml.etree.ElementTree as ET

    nodes, ways = {}, []
    for ev, el in ET.iterparse(osm, events=("end",)):
        if el.tag == "node":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if "local_x" in tags and "local_y" in tags:
                nodes[el.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
            el.clear()
        elif el.tag == "way":
            pts = [nodes[nd.get("ref")] for nd in el.findall("nd") if nd.get("ref") in nodes]
            if len(pts) >= 2:
                ways.append(np.array(pts))
            el.clear()
    return ways


def warnings(surface_map, ways=None):
    """Things a check reports without refusing the map."""
    out = []
    zs = surface_map.zones
    for i, a in enumerate(zs):
        for b in zs[i + 1:]:
            ax0, ax1, ay0, ay1 = a.bbox
            bx0, bx1, by0, by1 = b.bbox
            both = [f for f in ("cls", "wetness", "water_depth")
                    if getattr(a, f) is not None and getattr(b, f) is not None]
            if both and ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                names = ", ".join("class" if f == "cls" else f for f in both)
                out.append(f"zones {a.id} and {b.id} may overlap, both setting {names}: "
                           f"{b.id}'s wins")
    if ways:
        pts = np.concatenate(ways)
        lo, hi = pts.min(axis=0) - 50.0, pts.max(axis=0) + 50.0
        for z in zs:
            x0, x1, y0, y1 = z.bbox
            if x1 < lo[0] or x0 > hi[0] or y1 < lo[1] or y0 > hi[1]:
                out.append(f"zone {z.id} lies outside the Lanelet2 map's extent")
    return out


def plot(surface_map, out, ways=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    names = sorted(surface_map.table.classes)
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    colour = {n: cmap(i % cmap.N) for i, n in enumerate(names)}
    fig, ax = plt.subplots(figsize=(12, 12))
    for w in ways:
        ax.plot(w[:, 0], w[:, 1], color="0.6", lw=0.4)
    for z in surface_map.zones:
        if isinstance(z.water_depth, Raster):
            r = z.water_depth
            ax.imshow(r.values * 1000.0, extent=r.extent, origin="upper", cmap="Blues",
                      alpha=0.7, zorder=2)
        fc = colour[z.cls] if z.cls else (0.3, 0.55, 0.9, 0.25)  # water only: light blue
        ax.add_patch(Polygon(z.polygon, closed=True, fc=fc, ec="k", alpha=0.45 if z.cls else None,
                             hatch="//" if z.soil else None, zorder=3))
        cx, cy = z.polygon.mean(axis=0)
        extra = [z.cls or "", z.wetness or "",
                 "" if z.water_depth is None else
                 ("depth raster" if isinstance(z.water_depth, Raster)
                  else f"{z.water_depth * 1000:.1f} mm")]
        ax.text(cx, cy, "\n".join([z.id] + [e for e in extra if e]), ha="center", va="center",
                fontsize=7, zorder=4)
    handles = [Polygon([[0, 0]], fc=colour[n], alpha=0.45, label=n) for n in names
               if any(z.cls == n for z in surface_map.zones)]
    defaults = ", ".join(f"{k} {v}" for k, v in surface_map.defaults.items())
    ax.set_title(f"{surface_map.path}\nlabel defaults: {defaults}", fontsize=8)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (m, map frame)")
    ax.set_ylabel("y (m)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check or plot a map's surfaces.yaml (047) and, "
                                 "for a map directory, its named maps (surfaces/*.yaml, 056)")
    ap.add_argument("command", choices=("check", "plot"))
    ap.add_argument("map", help="a map directory (its surfaces.yaml and surfaces/*.yaml) or a "
                                "surfaces.yaml")
    ap.add_argument("-o", "--output", default="",
                    help="plot: the image (default <map>/surfaces.png)")
    args = ap.parse_args(argv)
    path = find(args.map)
    named = named_maps([Path(args.map) / SURFACES_DIR]) if Path(args.map).is_dir() else {}
    if not path.exists() and not named:
        sys.exit(f"{path}: no surface map")
    refused = []
    for name, f in sorted(named.items()):
        try:
            m = load(f)
            print(f"{f}: named map {name}: {len(m.zones)} zone(s), "
                  f"{sum(bool(z.soil) for z in m.zones)} with soil")
        except SurfaceError as e:
            refused.append(f"{f}: refused\n{e}")
    if not path.exists():
        if refused:
            sys.exit("\n".join(refused))
        return
    try:
        smap = load(path)
    except SurfaceError as e:
        refused.append(f"{path}: refused\n{e}")
    if refused:
        sys.exit("\n".join(refused))
    osm = path.parent / "lanelet2_map.osm"
    ways = lanelet_ways(osm) if osm.exists() else []
    for w in warnings(smap, ways):
        print(f"warning: {w}")
    for zid, cls, why in smap.unvalidated:
        print(f"warning: zone {zid}: soil on unvalidated {cls} parameters: {why}")
    print(f"{path}: {len(smap.zones)} zone(s), {sum(bool(z.soil) for z in smap.zones)} with soil; "
          f"{len(smap.table.classes)} classes")
    if args.command == "plot":
        out = args.output or str(path.parent / "surfaces.png")
        plot(smap, out, ways)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
