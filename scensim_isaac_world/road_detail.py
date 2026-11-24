"""A road's small-scale surface: one seeded height function per road corridor (roadmap 066 R3).
Pure numpy.

The world's own ground is the base road, smooth below a few metres, which PhysX's wheel query
sees. This adds what lies on it, as a height above the base, m:

- the cross-section, as a function of the distance v from the centreline: a straight "A"
  crown (FHWA: 4 %, never over 6 %) or a level section (Japanese forest roads, 林業専用道), the
  shoulder break, a 1V:3H foreslope, a flat-bottom ditch and a 1V:2H backslope;
- an isotropic 2D roughness field with ISO 8608's spectrum, Gd(n) = Gd(n0) (n / n0)^-w with
  n0 = 0.1 cycles/m, broken at 2 cycles/m as measured gravel is (Zuraulis et al. 2021);
- two ruts at half the track width with a windrow between them;
- potholes, Poisson-placed along the road, weighted to sags;
- washboard in zones (braking, acceleration, climbs): a sinusoid of wavelength about the
  modal speed in km/h / 100 m (TRH 20), slightly asymmetric;
- water bars, oblique ridges across the road, spaced by grade.

Everything is drawn from the corridor's seed per tile or per segment, so a tile generated again
(streamed back in) is identical. The detail is sampled into world-aligned tiles of TILE m at
SPACING m; `height` reads them bilinearly, and the render mesh and the lidar are built from
the same samples (question 3: one road).

The centreline is a polyline in the world frame (x, y); u is the distance along it, v the
signed distance to its left.
"""
from dataclasses import dataclass, field
import math

import numpy as np

SPACING = 0.05  # m between samples: six or more on a 0.3 m washboard wave, ISO 8608's 10 c/m
TILE = 16.0  # m, a tile's side; the roughness field's longest wave is 2 TILE
N0 = 0.1  # ISO 8608's reference spatial frequency, cycles/m
# ISO 8608 classes: the geometric mean of Gd(n0) per class, m^3
ISO_CLASS = {c: 16e-6 * 4 ** i for i, c in enumerate("ABCDEFGH")}
N_MAX = 10.0  # cycles/m: the shortest wave a tire feels (ISO 8608's upper limit)


@dataclass
class Section:
    """The cross-section, m. `crown`: the cross slope each way from the centreline (0: level).
    The carriageway is 2 `half_width` wide; then a shoulder, a foreslope down to the ditch,
    its flat bottom and the backslope up to the base level again."""

    half_width: float = 2.5
    crown: float = 0.04
    shoulder: float = 0.5
    foreslope: float = 1.0 / 3.0  # V:H
    ditch_depth: float = 0.30  # below the shoulder's edge
    ditch_bottom: float = 0.6
    backslope: float = 0.5

    def height(self, v):
        a = np.abs(v)
        w = self.half_width
        h = -self.crown * np.minimum(a, w + self.shoulder)
        edge = -self.crown * (w + self.shoulder)
        x = a - (w + self.shoulder)
        fore = self.ditch_depth / self.foreslope
        bottom = edge - self.ditch_depth
        back = (0.0 - bottom) / self.backslope  # up to the base (0) again
        h = np.where(x > 0, edge - self.foreslope * x, h)
        h = np.where(x > fore, bottom, h)
        x2 = x - fore - self.ditch_bottom
        h = np.where(x2 > 0, np.minimum(bottom + self.backslope * x2, 0.0), h)
        return np.where(x2 > back, 0.0, h)

    def extent(self):
        """The half width beyond which the section is the base's own, m."""
        edge = self.crown * (self.half_width + self.shoulder) + self.ditch_depth
        return (self.half_width + self.shoulder + self.ditch_depth / self.foreslope
                + self.ditch_bottom + edge / self.backslope)

    # the interface StationSection shares: the same section at every u
    def height_uv(self, u, v):
        """The section's height at (u, v), m above the base."""
        return self.height(v)

    def carriageway(self, u):
        """The carriageway's half width at u, m."""
        return np.full(np.shape(u), float(self.half_width))

    def edge(self, u, v):
        """Where the shoulder ends on v's side at u, m from the centreline."""
        return np.full(np.shape(u), float(self.half_width + self.shoulder))

    def narrowest(self):
        """The carriageway's least half width, m."""
        return float(self.half_width)

    def describe(self):
        return f"half width {self.half_width:g} m, crown {self.crown:g}"


class StationSection:
    """The cross-section per station of a road record (road_record.Stations; roadmap 086 C5),
    each field interpolated linearly in u between stations and held past the ends. On top of
    Section's shape, per station: the bank (the formation rolled about the centreline, positive
    leaning left), the crown line `crown_offset` m left of the centreline, each side's own crown,
    shoulder, foreslope, ditch and backslope, and the lip: a drop of `lip_h` from the
    carriageway's edge to the shoulder over a face of `lip_angle` (90: a step, which the tiles'
    SPACING samples as one cell's ramp). Past the ditch the backslope runs back to the base.
    With one constant, symmetric, unbanked section without a lip it is Section's, to rounding."""

    COMMON = ("half_width", "bank", "crown_offset")
    SIDE = ("crown", "shoulder", "foreslope", "ditch_depth", "ditch_bottom", "backslope",
            "lip_h", "lip_angle")

    def __init__(self, stations):
        st = stations
        self.u = np.asarray(st.u, dtype=float)
        self.common = {k: np.asarray(st.common[k], dtype=float) for k in self.COMMON}
        self.side = {k: np.asarray(st.side[k], dtype=float) for k in self.SIDE}
        self.n = len(self.u)

    def _at(self, a, u):
        return a[0] + 0.0 * u if self.n == 1 else np.interp(u, self.u, a)

    def _fields(self, u, v):
        """Every field at u, a side's taken on v's side (left: v >= 0)."""
        left = v >= 0
        f = {k: self._at(a, u) for k, a in self.common.items()}
        for k, a in self.side.items():
            f[k] = np.where(left, self._at(a[0], u), self._at(a[1], u))
        return f

    def height_uv(self, u, v):
        """The section's height at (u, v), m above the base."""
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        f = self._fields(u, v)
        a = np.abs(v)
        w, sh = f["half_width"], f["shoulder"]
        off = f["crown_offset"]
        # the crown: down and away from the crown line, each side's own slope (the side of
        # the crown line a point is on), across the carriageway and the shoulder
        cw = v - off
        crown = np.where(cw >= 0, self._at(self.side["crown"][0], u),
                         self._at(self.side["crown"][1], u))
        h = -crown * np.abs(np.where(a <= w + sh, v, np.sign(v) * (w + sh)) - off)
        # the lip: the shoulder lies lip_h below the carriageway's edge, the face lip_h /
        # tan(lip_angle) wide
        lip = f["lip_h"]
        face = np.where(f["lip_angle"] >= 90.0, 0.0,
                        lip / np.tan(np.radians(np.minimum(f["lip_angle"], 89.999))))
        x_lip = a - w
        drop = np.where(x_lip <= 0, 0.0,
                        np.where(face > 0, lip * np.clip(x_lip / np.where(face > 0, face, 1.0),
                                                         0.0, 1.0), lip))
        h = h - drop
        edge = h  # at the shoulder's end (a > w + sh: the clamped crown, less the lip)
        # past the shoulder: the foreslope, the ditch's bottom, then the backslope
        x = a - (w + sh)
        fore = f["ditch_depth"] / f["foreslope"]
        bottom = edge - f["ditch_depth"]
        h = np.where(x > 0, edge - f["foreslope"] * x, h)
        h = np.where(x > fore, bottom, h)
        # the bank rolls the formation (to the ditch's far side) about the centreline
        formation_end = w + sh + fore + f["ditch_bottom"]
        tb = np.tan(f["bank"])
        h = h - tb * np.where(a <= formation_end, v, np.sign(v) * formation_end)
        start = bottom - tb * np.sign(v) * formation_end  # the backslope's foot, banked
        x2 = a - formation_end
        up = np.where(start <= 0, np.minimum(start + f["backslope"] * x2, 0.0),
                      np.maximum(start - f["backslope"] * x2, 0.0))
        return np.where(x2 > 0, up, h)

    def carriageway(self, u):
        return self._at(self.common["half_width"], np.asarray(u, dtype=float))

    def edge(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        f = self._fields(u, v)
        return f["half_width"] + f["shoulder"]

    def narrowest(self):
        return float(np.min(self.common["half_width"]))

    def extent(self):
        """The half width beyond which every station's section is the base's own, m."""
        w = self.common["half_width"][None, :]
        sd = self.side
        tb = np.abs(np.tan(self.common["bank"]))[None, :]
        reach = w + sd["shoulder"] + sd["ditch_depth"] / sd["foreslope"] + sd["ditch_bottom"]
        far = (np.abs(sd["crown"]) * (w + sd["shoulder"] + np.abs(self.common["crown_offset"]))
               + sd["lip_h"] + sd["ditch_depth"] + tb * reach)
        return float(np.max(reach + far / sd["backslope"]))

    def describe(self):
        hw = self.common["half_width"]
        return (f"half width {hw.min():g} to {hw.max():g} m, bank up to "
                f"{np.degrees(np.abs(self.common['bank']).max()):.1f} deg, per station")


@dataclass
class Roughness:
    """ISO 8608 class `cls` (or `gd0`, m^3), waviness `w` below the break at `n_break`
    cycles/m and `w_high` above it. Zuraulis et al. measured the break on gravel but not the
    slope beyond it; w_high = 1.5 (more short-wave energy, from corrugation and stones) is
    assumed."""

    cls: str = "D"
    gd0: float = 0.0
    w: float = 2.0
    n_break: float = 2.0
    w_high: float = 1.5

    def psd(self, n):
        """The displacement PSD along a line, m^3 (m^2 per cycle/m)."""
        g0 = self.gd0 or ISO_CLASS[self.cls]
        n = np.asarray(n, dtype=float)
        low = g0 * (np.maximum(n, 1e-9) / N0) ** -self.w
        gb = g0 * (self.n_break / N0) ** -self.w
        return np.where(n <= self.n_break, low, gb * (np.maximum(n, 1e-9) / self.n_break)
                        ** -self.w_high)

    def psd2(self, kappa):
        """The isotropic 2D PSD at radial frequency kappa (cycles/m) whose line sections have
        `psd`: for G(n) ~ n^-w, G2 = G Gamma((w+1)/2) / (2 sqrt(pi) Gamma(w/2)) / kappa (G is
        one-sided, G2 over the whole plane), applied per band."""
        k = np.maximum(np.asarray(kappa, dtype=float), 1e-9)
        c = lambda x: math.gamma((x + 1) / 2) / (2 * math.sqrt(math.pi) * math.gamma(x / 2))
        return self.psd(k) * np.where(k <= self.n_break, c(self.w), c(self.w_high)) / k


@dataclass
class Ruts:
    """Two ruts at +-`offset` of the centreline (half a track) and a windrow between them."""

    depth: float = 0.03
    width: float = 0.30  # 2 sigma of each rut's Gaussian
    offset: float = 0.80
    windrow: float = 0.02  # the centre windrow's height
    windrow_width: float = 0.5

    def height(self, v):
        s = self.width / 2
        h = -self.depth * (np.exp(-0.5 * ((v - self.offset) / s) ** 2)
                           + np.exp(-0.5 * ((v + self.offset) / s) ** 2))
        return h + self.windrow * np.exp(-0.5 * (v / (self.windrow_width / 2)) ** 2)


@dataclass
class Potholes:
    """Poisson along the road at `per_100m`, `sag_weight` times as many in a sag (the base's
    grade changing from down to up); depth and diameter uniform in the ranges (USACE TM 5-626:
    1.5-10 cm deep, 0.3-1 m across), within the carriageway; edges 3 cm wide."""

    per_100m: float = 2.0
    depth: tuple = (0.015, 0.08)
    diameter: tuple = (0.3, 1.0)
    edge: float = 0.03
    sag_weight: float = 3.0
    # fixed holes in place of the Poisson ones (086 C6a, a fixture's): [u, v, depth, length,
    # width] each, m; rectangular, vertical edges
    at: tuple = ()


@dataclass
class Cleats:
    """Fixed cleats across the carriageway (086 C6a, R0's): [u, height, length] each, m, a box
    from u to u + length with vertical edges."""

    at: tuple = ()


@dataclass
class Washboard:
    """A corrugation zone from u0 to u1 m along the road: peak-to-peak `height`, wavelength
    `pitch` (m; TRH 20: cm about the modal speed in km/h), `skew` > 0 steepening each ridge's
    far face. Ramps in and out over two waves."""

    u0: float
    u1: float
    height: float = 0.02
    pitch: float = 0.7
    skew: float = 0.3

    def h(self, u):
        ph = 2 * math.pi * (u - self.u0) / self.pitch
        wave = np.cos(ph - self.skew * np.sin(ph))  # skewed: the ridges lean
        ramp = np.clip(np.minimum(u - self.u0, self.u1 - u) / (2 * self.pitch), 0.0, 1.0)
        return np.where((u >= self.u0) & (u <= self.u1), self.height / 2 * (1 - wave) * ramp,
                        0.0)


@dataclass
class WaterBars:
    """Oblique ridges across the carriageway at `angle` deg from square, `height` and
    `width` (a rubber bar's: 92 % of a Tokyo survey's cross drains; their height is not
    published, assumed). Spaced by the base's grade: `spacing(grade %)` m."""

    height: float = 0.06
    width: float = 0.25
    angle: float = 35.0
    at: tuple = ()  # u positions; empty: from the grade (`spacing`)

    @staticmethod
    def spacing(grade):
        """USFS-style cross-drain spacing on erodible soil, m: closer on steeper grades
        (assumed fit to the guides' tables: 120 m at 2 %, 40 m at 10 %)."""
        return float(np.clip(240.0 / max(grade, 1.0) ** 0.68, 20.0, 150.0))


@dataclass
class Corridor:
    """One road's detail: a centreline polyline (k, 2) world m, or (k, 3) with the base road's
    height along it (the world's road surface is built from the same line; `base_z`), and its
    features. `grade`:
    the base road's grade along u, a function u -> % (positive uphill), for the potholes'
    sags and the water bars' spacing; None: level. `taper` (m, 0: none): the detail fades in
    over the first `taper` m and out over the last, so that a road made of sections (one
    corridor each, the next starting `taper` m before this one ends) blends from one into the
    next: in the overlap the two weights sum to one, so a cross-section both share stays
    itself, and nothing steps where one ends."""

    centreline: np.ndarray
    seed: int = 0
    section: Section = field(default_factory=Section)  # or StationSection (086 C5)
    roughness: Roughness = field(default_factory=Roughness)
    ruts: Ruts = None
    potholes: Potholes = None
    washboard: tuple = ()
    water_bars: WaterBars = None
    cleats: Cleats = None  # 086 C6a: a fixture's box cleats
    grade: object = None
    rocks: object = None  # rocks.Zones: the rock cover by zone (R4); None: no rocks
    taper: float = 0.0
    stations: object = None  # road_record.Stations: the section per station (086 C1)
    section_in_base: bool = False  # the base road carries the section (086 C5b): features only

    def __post_init__(self):
        line = np.asarray(self.centreline, dtype=float)
        self.z = line[:, 2] if line.shape[1] > 2 else np.zeros(len(line))
        self.centreline = line[:, :2]
        d = np.diff(self.centreline, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.seg_dir = d / self.seg_len[:, None]
        self.seg_u = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.seg_u[-1])
        self.reach = self.section.extent() + 1.0
        self._holes = self._place_potholes()
        self._bars = self._place_bars()

    def base_z(self, u):
        """The base road's height at u along the centreline, m (level across it)."""
        return np.interp(u, self.seg_u, self.z)

    def surface_base(self, u, v):
        """What the true surface stands on at (u, v), m: the base road's height, and on a
        `section_in_base` corridor the record's section too, which the base mesh carries and
        `features` leaves out (issue 120). The true surface is this plus the detail."""
        z = self.base_z(u)
        if self.section_in_base and self.stations is not None:
            if getattr(self, "_record_section", None) is None:
                self._record_section = StationSection(self.stations)
            z = z + self._record_section.height_uv(u, v)
        return z

    UV_BLOCK = 2.0  # m: the side of a block of points culled on its own (issue 142)
    UV_BLOCKED = 4096  # points: at least this many are culled block by block

    def uv(self, x, y):
        """Each point's (u, v) on the centreline: its nearest segment's (P,), and whether it lies
        within the corridor's reach. Many points (a tile's) are culled in blocks of UV_BLOCK
        m, each keeping the segments that can be nearest to its own points (issue 142: a tile
        culled whole kept a hundred segments, each a pass over all its points); the nearest
        segment of every point survives its block's cull, so the result is the same."""
        p = np.stack([np.ravel(x), np.ravel(y)], axis=1)
        if len(p) < self.UV_BLOCKED or len(self.seg_len) < 64:
            return self._uv(p)
        b = np.floor((p - p.min(axis=0)) / self.UV_BLOCK).astype(np.int64)
        block = b[:, 0] * (int(b[:, 1].max()) + 1) + b[:, 1]
        order = np.argsort(block, kind="stable")
        cuts = np.nonzero(np.diff(block[order]))[0] + 1
        u, v, near = np.empty(len(p)), np.empty(len(p)), np.empty(len(p), dtype=bool)
        whole = self._candidates(p)  # a block's nearest segments are among the tile's
        for idx in np.split(order, cuts):
            u[idx], v[idx], near[idx] = self._uv_block(p[idx], self._candidates(p[idx], whole))
        return u, v, near

    def _uv_block(self, p, segs):
        """`_uv` over the segments `segs` at once: each point's first nearest in their order,
        as the loop's strict `<` keeps it; the same arithmetic per point and segment."""
        if not len(segs):
            return self._uv(p, segs)
        a, dr, ln = self.centreline[segs, :2], self.seg_dir[segs], self.seg_len[segs]
        rel = p[:, None, :] - a[None, :, :]  # (points, segments, 2)
        t = np.clip(rel[..., 0] * dr[:, 0] + rel[..., 1] * dr[:, 1], 0.0, ln)
        fx = a[:, 0] + t * dr[:, 0]
        fy = a[:, 1] + t * dr[:, 1]
        d = np.hypot(p[:, 0, None] - fx, p[:, 1, None] - fy)
        k = np.argmin(d, axis=1)
        r = np.arange(len(p))
        best = d[r, k]
        u = self.seg_u[segs][k] + t[r, k]
        cross = dr[k, 0] * rel[r, k, 1] - dr[k, 1] * rel[r, k, 0]
        return u, cross, best <= self.reach

    def _uv(self, p, among=None):
        a = self.centreline[:-1]
        best_d = np.full(len(p), np.inf)
        u = np.zeros(len(p))
        v = np.zeros(len(p))
        for s in self._candidates(p, among):  # in order: ties resolve as over every segment
            rel = p - a[s]
            t = np.clip(rel @ self.seg_dir[s], 0.0, self.seg_len[s])
            foot = a[s] + t[:, None] * self.seg_dir[s]
            d = np.hypot(*(p - foot).T)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            u = np.where(better, self.seg_u[s] + t, u)
            cross = self.seg_dir[s, 0] * rel[:, 1] - self.seg_dir[s, 1] * rel[:, 0]
            v = np.where(better, cross, v)
        return u, v, best_d <= self.reach

    GRID = 16.0  # m: the segment grid's cells (issue 142)

    def _segs_in(self, lo, hi):
        """The segments whose bounding boxes meet the cells covering [lo, hi], in order; None
        when that is more than 400 cells (then every segment)."""
        g = getattr(self, "_grid_index", None)
        if g is None:
            a, b = self.centreline[:-1], self.centreline[1:]
            c0 = np.floor(np.minimum(a, b) / self.GRID).astype(np.int64)
            c1 = np.floor(np.maximum(a, b) / self.GRID).astype(np.int64)
            cells = {}
            for k in range(len(a)):
                for i in range(c0[k, 0], c1[k, 0] + 1):
                    for j in range(c0[k, 1], c1[k, 1] + 1):
                        cells.setdefault((i, j), []).append(k)
            g = self._grid_index = {key: np.array(v) for key, v in cells.items()}
        i0, j0 = np.floor(np.asarray(lo) / self.GRID).astype(np.int64)
        i1, j1 = np.floor(np.asarray(hi) / self.GRID).astype(np.int64)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > 400:
            return None
        parts = [g[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if (i, j) in g]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def _near_segments(self, p):
        """A superset of the segments that can be nearest to some point of `p`, from the grid
        (issue 142: rocks' placement projects hundreds of small sets of points a tile, and each
        took a pass over every segment): a first bound from the segments around the points,
        then every segment within it. The cull after it is exact over any such superset."""
        lo, hi = p.min(axis=0), p.max(axis=0)
        r, seed = self.GRID, None
        while True:
            seed = self._segs_in(lo - r, hi + r)
            if seed is None:
                return np.arange(len(self.seg_len))
            if len(seed):
                break
            r *= 4.0  # far from the road: look wider
        a, dr, ln = self.centreline[seed], self.seg_dir[seed], self.seg_len[seed]
        far = np.zeros(len(seed))
        for c in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])):
            rel = np.asarray(c) - a
            t = np.clip(np.einsum("ij,ij->i", rel, dr), 0.0, ln)
            far = np.maximum(far, np.hypot(*(np.asarray(c) - (a + t[:, None] * dr)).T))
        bound = float(far.min()) + 1e-6
        out = self._segs_in(lo - bound, hi + bound)
        return np.arange(len(self.seg_len)) if out is None else out

    def _candidates(self, p, among=None):
        """The segments that can be nearest to some point of `p` (issue 128): D, the least over
        the segments of the farthest corner of p's bounding box from the segment, bounds every
        point's nearest distance, so a segment whose bounding box lies farther than D from p's
        cannot be nearest to any. Exact; a 3000-station record's tile keeps a few dozen."""
        if among is not None:  # only these (in order): those a larger set of points kept
            among = np.asarray(among)
        elif not len(p) or len(self.seg_len) < 64:
            return range(len(self.seg_len))
        else:
            among = self._near_segments(p)
        if not len(p):
            return among
        a, b = self.centreline[among, :2], self.centreline[among + 1, :2]
        seg_dir, seg_len = self.seg_dir[among], self.seg_len[among]
        lo, hi = p.min(axis=0), p.max(axis=0)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        far = np.zeros(len(a))
        for c in corners:
            rel = c - a
            t = np.clip(np.einsum("ij,ij->i", rel, seg_dir), 0.0, seg_len)
            foot = a + t[:, None] * seg_dir
            far = np.maximum(far, np.hypot(*(c - foot).T))
        bound = float(far.min()) + 1e-6
        smin, smax = np.minimum(a, b), np.maximum(a, b)
        gap = np.hypot(np.maximum(0.0, np.maximum(smin[:, 0] - hi[0], lo[0] - smax[:, 0])),
                       np.maximum(0.0, np.maximum(smin[:, 1] - hi[1], lo[1] - smax[:, 1])))
        return among[gap <= bound]

    def weight(self, u):
        """The share of this corridor's detail at u along it: 1, or falling linearly to 0 over
        `taper` m at either end."""
        if self.taper <= 0:
            return np.ones(np.shape(u))
        return np.clip(np.minimum(u, self.length - u) / self.taper, 0.0, 1.0)

    def _rng(self, *key):
        return np.random.default_rng([self.seed & 0xFFFFFFFF, *[int(k) & 0xFFFFFFFF
                                                                for k in key]])

    def _place_potholes(self):
        """(u, v, depth, radius) per pothole, per 100 m segment of the road; none when the
        potholes are fixed (`Potholes.at`, `features`' rectangles)."""
        if self.potholes is None or self.potholes.at:
            return np.zeros((0, 4))
        ph = self.potholes
        out = []
        for seg in range(int(math.ceil(self.length / 100.0))):
            u0 = seg * 100.0
            rate = ph.per_100m
            if self.grade is not None:  # a sag: the grade rises through it
                g0, g1 = self.grade(u0), self.grade(min(u0 + 100.0, self.length))
                if g0 < 0 < g1 or (g1 - g0) > 2.0:
                    rate *= ph.sag_weight
            rng = self._rng(1, seg)
            for _ in range(rng.poisson(rate)):
                r = rng.uniform(*ph.diameter) / 2
                out.append((u0 + rng.uniform(0, 100.0),
                            rng.uniform(-self.section.narrowest() + r,
                                        self.section.narrowest() - r),
                            rng.uniform(*ph.depth), r))
        return np.array(out).reshape(-1, 4)

    def _place_bars(self):
        if self.water_bars is None:
            return np.zeros(0)
        if self.water_bars.at:
            return np.array(self.water_bars.at, dtype=float)
        out, u = [], 0.0
        while True:
            g = abs(self.grade(u)) if self.grade is not None else 0.0
            u += WaterBars.spacing(g)
            if u >= self.length:
                return np.array(out)
            out.append(u)

    def features(self, u, v):
        """Everything but the roughness at road coordinates (u, v), m."""
        h = (np.zeros(np.broadcast(u, v).shape) if self.section_in_base
             else self.section.height_uv(u, v))
        inside = np.abs(v) <= self.section.carriageway(u)
        if self.ruts is not None:
            h = h + np.where(inside, self.ruts.height(v), 0.0)
        for (pu, pv, depth, r) in self._holes:
            d = np.hypot(u - pu, v - pv)
            e = self.potholes.edge
            h = h - depth * np.clip((r - d) / e, 0.0, 1.0)
        if self.potholes is not None:
            for (pu, pv, depth, length, width) in self.potholes.at:  # fixed, vertical edges
                h = h - np.where((u >= pu) & (u <= pu + length) & (np.abs(v - pv) <= width / 2),
                                 depth, 0.0)
        if self.cleats is not None:
            for (cu, height, length) in self.cleats.at:  # boxes, vertical edges
                h = h + np.where(inside & (u >= cu) & (u <= cu + length), height, 0.0)
        for zone in self.washboard:
            h = h + np.where(inside, zone.h(u), 0.0)
        if len(self._bars):
            wb = self.water_bars
            t = math.tan(math.radians(wb.angle))
            s = u - v * t  # along the road, the bar's oblique line through v
            for ub in self._bars:
                d = (s - ub) * math.cos(math.radians(wb.angle))
                bar = np.where(np.abs(d) < wb.width / 2,
                               wb.height * np.cos(math.pi * d / wb.width) ** 2, 0.0)
                h = h + np.where(inside, bar, 0.0)
        return h


class Detail:
    """The corridors' detail as tiles: `height(x, y)` m above the base road, any array shape.
    Tiles are made on first use and kept (`max_tiles` most recently used). With `workers`,
    `prefetch` makes the tiles about points in a worker thread, so that a tile takes no frame's
    time when the car reaches it (one takes about 40 ms)."""

    def __init__(self, corridors, spacing=SPACING, tile=TILE, max_tiles=64, workers=0,
                 processes=False, extras=None):
        """`extras`: "module:function", called as function(detail, i, j) after each tile is
        made, where the tile is made (the worker), its result kept per tile (`extra`); rocks'
        placement (`rocks:place`) is one."""
        import threading
        self.lock = threading.RLock()
        self.corridors = list(corridors)
        self.spacing, self.tile = spacing, tile
        self.extras_name, self._extras_fn = extras, None
        self.extras = {}  # (i, j) -> extras' result, kept for the session
        self.pool = None
        if workers and processes:
            self.pool = TileProcess(self.corridors, spacing, tile, extras)
        elif workers:
            import concurrent.futures
            self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.pending = {}  # tile -> its worker's future
        self.n = int(round(tile / spacing))
        self.tiles = {}  # (i, j) -> (n + 1, n + 1) heights, rows along y
        # (i, j) -> (base, zones): `surface_tile`'s base road and `zones_tile`, made with the tile
        # from the same uv (issue 134), evicted with it: float64 and int8, 0.93 MB a tile at n 320
        self.looks = {}
        self.look_fallbacks = 0  # a tile's look made on the caller's thread (issue 134)
        self.max_tiles = max_tiles
        self.fields = {}  # (corridor, i, j) -> its roughness field on 2 tile
        self.amps = {}  # corridor -> its spectrum's amplitudes on the field's grid
        self.touched = {}  # (i, j) -> whether a corridor reaches the tile
        self.made_ms = []  # each tile's generation time
        self.stall_ms = []  # tile_of's waits for a tile not made ahead (the caller's thread)

    def _field(self, ci, i, j):
        """The roughness of corridor `ci` over the window of lattice cell (i, j), which spans
        2 tiles from (i, j) tile's origin: a periodic FFT field with the 2D spectrum."""
        key = (ci, i, j)
        with self.lock:
            f = self.fields.get(key)
        if f is not None:
            return f
        m = 2 * self.n
        amp = self.amps.get(ci)
        if amp is None:  # the spectrum's amplitudes, the same for every cell of a corridor
            k = np.fft.fftfreq(m, d=self.spacing)
            kx, ky = np.meshgrid(k, k)
            kappa = np.hypot(kx, ky)
            band = (kappa >= 1.0 / (2 * self.tile)) & (kappa <= N_MAX)
            dk = 1.0 / (m * self.spacing)
            psd2 = self.corridors[ci].roughness.psd2(kappa)
            amp = self.amps[ci] = np.where(band, np.sqrt(psd2 * dk * dk), 0.0)
        rng = self.corridors[ci]._rng(2, i, j)
        z = (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))) / math.sqrt(2)
        # the real part of the inverse transform of a complex Gaussian spectrum: each mode's
        # variance amp^2, the 2D PSD's integral over the band
        f = np.real(np.fft.ifft2(amp * z)) * m * m * math.sqrt(2)
        with self.lock:
            if len(self.fields) > 2 * self.max_tiles:
                self.fields.pop(next(iter(self.fields)))
            self.fields[key] = f
        return f

    def _grid(self, i, j):
        n, d = self.n, self.spacing
        return np.meshgrid(i * self.tile + np.arange(n + 1) * d,
                           j * self.tile + np.arange(n + 1) * d)

    def _reaches(self, c, i, j):
        """Whether corridor `c` can reach tile (i, j): its centre's |v| within the corridor's
        reach and a tile. A point's distance to the centreline changes by at most its own
        displacement, and every sample lies within half a diagonal (under a tile) of the
        centre, so a corridor that fails this has no sample within reach."""
        n = self.n
        cx = i * self.tile + (n // 2) * self.spacing
        cy = j * self.tile + (n // 2) * self.spacing
        _, v, _ = c.uv(np.array([cx]), np.array([cy]))
        return not (v.size and abs(v[0]) > c.reach + self.tile)

    def _uvs(self, i, j):
        """Each corridor's (u, v, near) over tile (i, j)'s samples: one projection a tile, shared
        by its heights, base road and zones (issue 134). A corridor that cannot reach the tile
        is not projected (None): no sample is near it (issue 142: projecting every tile on every
        corridor made the worker's tiles off the road ten times slower, 44 to 424 ms)."""
        X, Y = self._grid(i, j)
        return [c.uv(X, Y) if self._reaches(c, i, j) else None for c in self.corridors]

    def _look(self, i, j, uvs):
        """The tile's base road (`surface_tile` less the detail) and zones, from `uvs`."""
        X, _ = self._grid(i, j)
        base = np.zeros(X.shape)
        zones = np.zeros(X.shape, dtype=np.int8)
        for c, uvn in zip(self.corridors, uvs):
            if uvn is None:  # not near anywhere: its base adds nothing, its zones are OUTSIDE
                continue
            u, v, near = uvn
            if c.z.any() or c.section_in_base:
                base = np.where(near.reshape(X.shape), c.surface_base(u, v).reshape(X.shape),
                                base)
            U, V, N = u.reshape(X.shape), v.reshape(X.shape), near.reshape(X.shape)
            road = N & (np.abs(V) <= c.section.edge(U, V))
            zones = np.maximum(zones, np.where(road, self.ROAD, np.where(N, self.VERGE,
                                                                         self.OUTSIDE)))
        return base, zones

    def _make(self, i, j, uvs=None):
        n, d = self.n, self.spacing
        xs = i * self.tile + np.arange(n + 1) * d
        ys = j * self.tile + np.arange(n + 1) * d
        X, Y = np.meshgrid(xs, ys)
        out = np.zeros(X.shape)
        for ci, c in enumerate(self.corridors):
            # a corridor that reaches nowhere near this tile adds nothing
            if uvs is not None:
                if uvs[ci] is None:
                    continue
                u, v, near = uvs[ci]
            elif not self._reaches(c, i, j):
                continue
            else:
                u, v, near = c.uv(X, Y)
            if not near.any():
                continue
            u, v, near = u.reshape(X.shape), v.reshape(X.shape), near.reshape(X.shape)
            h = c.features(u, v)
            # the roughness: four overlapping windows (cells (i-1..i, j-1..j)), sin^2 + cos^2
            # = 1 in each axis so the variance is the field's everywhere
            r = np.zeros(X.shape)
            for di in (-1, 0):
                for dj in (-1, 0):
                    f = self._field(ci, i + di, j + dj)
                    ox = (np.arange(n + 1) - di * n)  # this tile's samples in the window
                    oy = (np.arange(n + 1) - dj * n)
                    wx = np.sin(math.pi * ox / (2 * n))
                    wy = np.sin(math.pi * oy / (2 * n))
                    r += f[np.ix_(oy % (2 * n), ox % (2 * n))] * np.outer(wy, wx)
            # the roughness is the carriageway's and its shoulders', fading out over the ditch
            fade = np.clip((c.section.edge(u, v) + 1.0 - np.abs(v)), 0, 1)
            out += np.where(near, (h + r * fade) * c.weight(u), 0.0)
        return out

    def _keep(self, key, t, look=None):
        with self.lock:
            self.tiles[key] = t
            if look is not None:
                self.looks[key] = look
            while len(self.tiles) > self.max_tiles:
                old = next(iter(self.tiles))
                self.tiles.pop(old)
                self.looks.pop(old, None)

    def look(self, i, j):
        """(base, zones) of tile (i, j): the worker's, made with the tile; made here (and
        counted, `look_fallbacks`) when the tile came without one."""
        self.tile_of(i, j)
        with self.lock:
            look = self.looks.get((i, j))
        if look is None:
            self.look_fallbacks += 1
            look = self._look(i, j, self._uvs(i, j))
            with self.lock:
                if (i, j) in self.tiles:
                    self.looks[(i, j)] = look
        return look

    def surface_tile(self, i, j):
        """The tile's samples with the base road under them: what the render mesh shows and the
        collision tiles are (the corridor's surface_base where it reaches, its section too on a
        section_in_base corridor, issue 120; 0 elsewhere, the world's own ground there)."""
        base, _ = self.look(i, j)
        return base + self.tile_of(i, j)

    OUTSIDE, VERGE, ROAD = 0, 1, 2

    def zones_tile(self, i, j):
        """What each of the tile's samples is, as `surface_tile`'s grid: ROAD on a corridor's
        carriageway or shoulder, VERGE over the rest of its section (foreslope, ditch,
        backslope), OUTSIDE beyond every corridor's reach, where the world's own ground is. The
        render mesh covers only the corridors, and dresses the road and the verge apart."""
        return self.look(i, j)[1]

    def tile_of(self, i, j):
        with self.lock:
            t = self.tiles.pop((i, j), None)
            if t is not None:  # most recently used last
                self.tiles[(i, j)] = t
                return t
            job = self.pending.get((i, j))
        t0 = _now()
        t = None
        if job is not None:
            try:
                t, _, extra, look = job.result()  # being made: wait for it
            except Exception:  # the worker failed or is gone: make it here
                t = None
        if t is None:
            t, ms, extra, look = self._timed(i, j)
            self.made_ms.append(ms)
        self._keep((i, j), t, look)
        if self.extras_name:
            with self.lock:
                self.extras.setdefault((i, j), extra)
        self.stall_ms.append((_now() - t0) * 1e3)
        return t

    def tiles_near(self, x, y, reach):
        """The tiles within `reach` m of the points (x, y), as keys."""
        x, y = np.ravel(x), np.ravel(y)
        keys = set()
        for i0, i1, j0, j1 in zip(np.floor((x - reach) / self.tile).astype(int),
                                  np.floor((x + reach) / self.tile).astype(int),
                                  np.floor((y - reach) / self.tile).astype(int),
                                  np.floor((y + reach) / self.tile).astype(int)):
            keys.update((int(i), int(j)) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
        return keys

    def touches(self, i, j):
        """Whether a corridor reaches the tile: one that does not is all zeros."""
        hit = self.touched.get((i, j))
        if hit is None:
            cx, cy = (i + 0.5) * self.tile, (j + 0.5) * self.tile
            hit = False
            for c in self.corridors:
                _, v, _ = c.uv(np.array([cx]), np.array([cy]))
                if not v.size or abs(v[0]) <= c.reach + self.tile:
                    hit = True
                    break
            self.touched[(i, j)] = hit  # a bool per tile: kept for the session
        return hit

    def ready(self, key):
        with self.lock:
            return key in self.tiles

    def prefetch(self, keys):
        """Make these tiles in the worker (at once, without one)."""
        for key in keys:
            with self.lock:
                if key in self.tiles or key in self.pending:
                    continue
                if self.pool is None:
                    job = None
                else:
                    job = self.pending[key] = self.pool.submit(self._timed, *key)
                    job.add_done_callback(lambda f, key=key: self._made(key, f))
            if job is None:
                self.tile_of(*key)

    def _timed(self, i, j):
        t0 = _now()
        uvs = self._uvs(i, j)
        t = self._make(i, j, uvs)
        look = self._look(i, j, uvs)
        extra = None
        if self.extras_name:
            if self._extras_fn is None:
                import importlib

                mod, fn = self.extras_name.split(":")
                self._extras_fn = getattr(importlib.import_module(mod), fn)
            self._keep((i, j), t, look)  # the extras read the tile's heights
            extra = self._extras_fn(self, i, j)
        return t, (_now() - t0) * 1e3, extra, look

    def extra(self, i, j):
        """The tile's extras, once made (None before)."""
        with self.lock:
            return self.extras.get((i, j))

    def _made(self, key, job):
        try:
            if not job.cancelled() and job.exception() is None:
                t, ms, extra, look = job.result()
                self.made_ms.append(ms)
                self._keep(key, t, look)
                if self.extras_name:
                    with self.lock:
                        self.extras.setdefault(key, extra)
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def height(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        fx, fy = x / self.spacing, y / self.spacing
        gx, gy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ti, tj = gx // self.n, gy // self.n
        out = np.empty(x.shape)
        ti, tj, gx, gy = ti.ravel(), tj.ravel(), gx.ravel(), gy.ravel()
        fx, fy = fx.ravel(), fy.ravel()
        flat_out = out.reshape(-1)
        if not ti.size:
            return out
        # a car's samples lie on one to four neighbouring tiles: loop over their box, no sort
        for i in range(int(ti.min()), int(ti.max()) + 1):
            on_i = ti == i
            for j in range(int(tj.min()), int(tj.max()) + 1):
                sel = on_i & (tj == j)
                if not sel.any():
                    continue
                t = self.tile_of(i, j)
                lx, ly = gx[sel] - i * self.n, gy[sel] - j * self.n
                ax, ay = fx[sel] - gx[sel], fy[sel] - gy[sel]
                h00, h10 = t[ly, lx], t[ly, lx + 1]
                h01, h11 = t[ly + 1, lx], t[ly + 1, lx + 1]
                flat_out[sel] = ((h00 * (1 - ax) + h10 * ax) * (1 - ay)
                                 + (h01 * (1 - ax) + h11 * ax) * ay)
        return out


def _now():
    import time
    return time.perf_counter()


MAX_IN_FLIGHT = 2  # keys written to the tile process and not yet answered (issue 139)
MAX_QUEUED = 64  # keys waiting to be written; past it the oldest is dropped as stale


class TileProcess:
    """Makes tiles in a child process: a thread in the caller competes for the GIL with the
    simulation loop (in Isaac it made 2 of the 3 tiles a second 80 km/h needs). The child is
    `python -m scensim_isaac_world.road_detail --worker`, numpy only, on pipes, not
    multiprocessing: spawn would import the caller's __main__ again (a second SimulationApp),
    fork a process holding CUDA. The executor's `submit` and `shutdown`, for `Detail`.

    Issue 139: `submit` only queues the key, under the lock, and never blocks; a writer thread
    writes the keys, outside the lock, at most MAX_IN_FLIGHT unanswered, and the reader takes
    the lock only to pop a finished job. Before, `submit` wrote under the lock the reader
    needed, and a burst of keys that filled the worker's stdin, its 1.7 MB results filling its
    stdout, hung all three. Past MAX_QUEUED waiting keys the oldest is dropped: its future
    fails ("dropped"), and the caller makes that tile itself if it still wants it."""

    def __init__(self, corridors, spacing, tile, extras=None):
        import collections
        import os
        import pickle
        import subprocess
        import sys
        import threading

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        self.proc = subprocess.Popen([sys.executable, "-m", "scensim_isaac_world.road_detail",
                                      "--worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env, cwd=root)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jobs = {}  # (i, j) -> future, waiting or in flight
        self.queue = collections.deque()  # keys waiting to be written, oldest first
        self.in_flight = 0
        self.dropped = []  # futures to fail, by the writer, outside every lock
        self.closed = False
        self.dropped_n = 0
        pickle.dump((corridors, spacing, tile, extras), self.proc.stdin)
        self.proc.stdin.flush()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def submit(self, fn, i, j):
        import concurrent.futures

        key = (i, j)
        job = concurrent.futures.Future()
        job.set_running_or_notify_cancel()
        with self.lock:
            had = self.jobs.get(key)
            if had is not None:  # waiting or in flight: the same tile
                return had
            closed = self.closed
            if not closed:
                self.jobs[key] = job
                self.queue.append(key)
                while len(self.queue) > MAX_QUEUED:
                    old = self.queue.popleft()
                    self.dropped.append((self.jobs.pop(old), "dropped: stale (issue 139)"))
                    self.dropped_n += 1
                self.cond.notify_all()
        if closed:  # outside the lock, as every completion
            job.set_exception(RuntimeError("road tile process closed"))
        return job

    def _write(self):
        import pickle

        while True:
            with self.cond:
                while not (self.dropped or self.closed
                           or (self.queue and self.in_flight < MAX_IN_FLIGHT)):
                    self.cond.wait()
                drops, self.dropped = self.dropped, []
                key = None
                if not self.closed and self.queue and self.in_flight < MAX_IN_FLIGHT:
                    key = self.queue.popleft()
                    self.in_flight += 1
                closed = self.closed
            for job, why in drops:
                job.set_exception(RuntimeError(why))
            if key is not None:
                try:
                    pickle.dump(key, self.proc.stdin)
                    self.proc.stdin.flush()
                except (BrokenPipeError, ValueError, OSError) as e:
                    self._fail_all(f"road tile process: {e!r}")
                    return
            elif closed:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
                return

    def _fail_all(self, why):
        with self.cond:
            self.closed = True
            jobs = list(self.jobs.values()) + [j for j, _ in self.dropped]
            self.jobs, self.dropped = {}, []
            self.queue.clear()
            self.cond.notify_all()
        for job in jobs:
            if not job.done():
                job.set_exception(RuntimeError(why))

    def _read(self):
        import pickle

        while True:
            try:
                key, t, ms, extra, look = pickle.load(self.proc.stdout)
            except Exception as e:  # the child is gone: every waiting tile fails
                self._fail_all(f"road tile process ended: {e!r}")
                return
            with self.cond:
                job = self.jobs.pop(key, None)
                self.in_flight -= 1
                self.cond.notify_all()
            if job is not None:
                job.set_result((t, ms, extra, look))

    def shutdown(self, wait=False, cancel_futures=True):
        with self.cond:
            self.closed = True
            if cancel_futures:
                self.dropped += [(self.jobs.pop(k), "road tile process closed")
                                 for k in list(self.queue)]
                self.queue.clear()
            self.cond.notify_all()
        if wait:
            self.writer.join()
            self.proc.wait()
        else:
            self.proc.terminate()


def _worker():
    """The child of TileProcess: (corridors, spacing, tile, extras), then (i, j) per tile, on
    stdin; (key, heights, ms, extra, look) per tile on stdout. Ends at stdin's end."""
    import os
    import pickle
    import sys

    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed goes to stderr, not into the tiles
    inp = sys.stdin.buffer
    corridors, spacing, tile, extras = pickle.load(inp)
    d = Detail(corridors, spacing, tile, max_tiles=16, extras=extras)
    while True:
        try:
            key = pickle.load(inp)
        except EOFError:
            return
        t, ms, extra, look = d._timed(*key)
        pickle.dump((key, t, ms, extra, look), out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


# --- measures (R3's acceptance) ---------------------------------------------------------------

def line_psd(h, dx, segment=4096):
    """Welch's displacement PSD of a profile h sampled every dx m: (n cycles/m, G m^3)."""
    seg = min(segment, len(h))
    win = np.hanning(seg)
    step = seg // 2
    acc, k = 0.0, 0
    for s in range(0, len(h) - seg + 1, step):
        x = h[s:s + seg] - h[s:s + seg].mean()
        acc = acc + np.abs(np.fft.rfft(x * win)) ** 2
        k += 1
    n = np.fft.rfftfreq(seg, d=dx)
    return n, 2.0 * acc / k * dx / (win ** 2).sum()


def iri(h, dx, speed=80.0 / 3.6):
    """The International Roughness Index of profile h (m, every dx m), m/km: the golden quarter
    car (k1 653, k2 63.3, c 6.0, mu 0.15 per unit sprung mass) at 80 km/h on the profile's 250 mm
    moving average, its suspension's mean absolute rate over the speed (Sayers 1995)."""
    n = max(1, int(round(0.25 / dx)))
    p = np.convolve(h, np.ones(n) / n, mode="valid")
    k1, k2, c, mu = 653.0, 63.3, 6.0, 0.15
    a = np.array([[0, 1, 0, 0], [-k2, -c, k2, c], [0, 0, 0, 1],
                  [k2 / mu, c / mu, -(k1 + k2) / mu, -c / mu]])
    b = np.array([0, 0, 0, k1 / mu])
    dt = dx / speed
    # exact discretisation (zero-order hold on the profile)
    from scipy.linalg import expm
    m = np.zeros((5, 5))
    m[:4, :4], m[:4, 4] = a * dt, b * dt
    e = expm(m)
    ad, bd = e[:4, :4], e[:4, 4]
    # start at rest on the first 11 m's mean slope (the standard's initialisation)
    j = min(len(p) - 1, int(round(11.0 / dx)))
    slope = (p[j] - p[0]) / (j * dx) if j else 0.0
    x = np.array([p[0], slope * speed, p[0], slope * speed])
    acc = 0.0
    for z in p:
        x = ad @ x + bd * z
        acc += abs(x[1] - x[3])
    return acc * dt / (len(p) * dx) * 1000.0  # (m/s * s) / m -> m/m -> m/km


def iso2631_wk(acc, rate):
    """ISO 2631-1's Wk weighting (vertical, seated) applied to an acceleration signal (m/s^2)
    sampled at `rate` Hz, in the frequency domain; returns the weighted signal."""
    f = np.fft.rfftfreq(len(acc), d=1.0 / rate)
    s = 2j * math.pi * np.maximum(f, 1e-9)
    w = lambda hz: 2 * math.pi * hz
    f1, f2, f3, f4, q4, f5, q5, f6, q6 = 0.4, 100.0, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91
    hh = s * s / (s * s + w(f1) / (1 / math.sqrt(2)) * s + w(f1) ** 2)
    hl = w(f2) ** 2 / (s * s + w(f2) / (1 / math.sqrt(2)) * s + w(f2) ** 2)
    ht = (1 + s / w(f3)) / (1 + s / (q4 * w(f4)) + (s / w(f4)) ** 2)
    hs = ((1 + s / (q5 * w(f5)) + (s / w(f5)) ** 2) / (1 + s / (q6 * w(f6)) + (s / w(f6)) ** 2)
          * (w(f5) / w(f6)) ** 2)
    hw = hh * hl * ht * hs
    hw[0] = 0.0
    return np.fft.irfft(np.fft.rfft(acc) * hw, n=len(acc))


if __name__ == "__main__" and "--worker" in __import__("sys").argv:
    _worker()
