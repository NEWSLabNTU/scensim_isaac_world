"""The road record: a corridor's cross-section per station (roadmap 086 C1). Pure numpy.

A corridor of `road_detail.yaml` gives its section either as constants (`section:`, as
before) or per station along its centreline (`stations: <file>.npz`, beside the YAML). Both
become one `Stations`: arrays over the stations, one value per station and, for the fields of
one side, one per side. The fields are 069's section with 066 R3's:

- each station: `half_width`, `bank`, `crown_offset`, `kappa_v` (`COMMON`);
- each side, left and right: `crown`, `shoulder`, `foreslope`, `ditch_depth`, `ditch_bottom`,
  `backslope`, `lip_h`, `lip_angle`, `runoff_len`, `barrier_offset` (`SIDE`), and the
  surfaces and the hazard class (`SIDE_TEXT`).

In the YAML a side's field is a number for both sides or `{left: .., right: ..}`; in the .npz a
side's field is two arrays, `<field>_left` and `<field>_right`, and `u` (m along the
centreline, from 0, increasing) gives the stations. The .npz may hold the centreline too, `x`,
`y` and optionally `z` at each station (map frame, m; a generated course's record, 086 C3),
in place of the YAML's `centreline`; `u` is then the distance along it. A field left out takes its default
everywhere. Every problem found is listed (`RoadRecordError`), with where it is.

Who reads what: the Isaac road layer takes the section's shape (`detail_section`, else
road_detail.StationSection per station, 086 C5); the rails,
the surface strips and the planner read the rest (086's later steps).
"""
from dataclasses import dataclass
import math

import numpy as np

SCHEMA = 1

# field: (default, unit and meaning, check(value array) -> bad mask)
COMMON = {
    "half_width": (2.5, "m, half the carriageway", lambda a: ~(a > 0)),
    "bank": (0.0, "rad, the road's roll, positive leaning left", lambda a: ~(np.abs(a) < 0.35)),
    "crown_offset": (0.0, "m, the crown line's offset to the left of the centreline",
                     lambda a: ~np.isfinite(a)),
    "kappa_v": (0.0, "1/m, vertical curvature, positive in a sag", lambda a: ~np.isfinite(a)),
}
SIDE = {
    "crown": (0.04, "the cross slope down and away from the crown line (0: level)",
              lambda a: ~(np.abs(a) <= 0.15)),
    "shoulder": (0.5, "m, the shoulder's width", lambda a: ~(a >= 0)),
    "foreslope": (1.0 / 3.0, "V:H, down from the shoulder's edge to the ditch",
                  lambda a: ~(a > 0)),
    "ditch_depth": (0.30, "m, the ditch below the shoulder's edge", lambda a: ~(a >= 0)),
    "ditch_bottom": (0.6, "m, the ditch's flat bottom", lambda a: ~(a >= 0)),
    "backslope": (0.5, "V:H, up from the ditch to the ground", lambda a: ~(a > 0)),
    "lip_h": (0.0, "m, the drop from the carriageway's edge to the shoulder",
              lambda a: ~(a >= 0)),
    "lip_angle": (90.0, "deg, the lip's face (90 vertical, 30 tapered)",
                  lambda a: ~((a > 0) & (a <= 90))),
    "runoff_len": (0.0, "m of run-off area past the shoulder", lambda a: ~(a >= 0)),
    "barrier_offset": (math.nan, "m from the centreline to a guardrail or wall (nan: none)",
                       lambda a: ~(np.isnan(a) | (a > 0))),
}
HAZARDS = ("shoulder", "runoff", "lip", "ditch", "barrier", "drop")
SIDE_TEXT = {
    "shoulder_surface": ("", "the shoulder's surface class (empty: the surface map's)"),
    "runoff_surface": ("", "the run-off's surface class (empty: the surface map's)"),
    "hazard": ("", f"what lies past the edge: {', '.join(HAZARDS)} (empty: not classified)"),
}
SIDES = ("left", "right")
# the fields the Isaac road layer's section (road_detail.Section) shapes the road with today
DETAIL = ("half_width", "crown", "shoulder", "foreslope", "ditch_depth", "ditch_bottom",
          "backslope")


class RoadRecordError(ValueError):
    pass


@dataclass
class Stations:
    """The section at each station: `u` (n,), `common[field]` (n,), `side[field]` (2, n),
    left then right, `text[field]` (2, n) of str; `xyz` (n, 3), the centreline at each
    station when the record carries it, else None."""

    u: np.ndarray
    common: dict
    side: dict
    text: dict
    xyz: np.ndarray = None

    def __len__(self):
        return len(self.u)

    def constant(self, field):
        """Whether `field` is the same at every station (and, for a side's, on both sides)."""
        a = self.common.get(field, self.side.get(field))
        if a is None:
            a = self.text[field]
        flat = np.ravel(a)
        if flat.dtype.kind == "f":
            return bool(np.all(flat == flat[0]) or np.all(np.isnan(flat)))
        return bool(np.all(flat == flat[0]))


def keys():
    """Every field: `{name: (default, meaning)}`, a side's named `<field>.left|right`."""
    out = {k: (d, m) for k, (d, m, _) in COMMON.items()}
    out.update({f"{k}.left|right": (d, m) for k, (d, m, _) in SIDE.items()})
    out.update({f"{k}.left|right": (d, m) for k, (d, m) in SIDE_TEXT.items()})
    return out


def _check_values(name, a, check, at, problems):
    bad = check(np.asarray(a, dtype=float))
    if np.any(bad):
        i = int(np.flatnonzero(np.ravel(bad))[0])
        # a Python float: numpy 2's repr of its scalar reads "np.float64(-1.0)" (issue 090)
        problems.append(f"{at}: {name} {float(np.ravel(a)[i])!r} out of range ({_meaning(name)})"
                        + (f", at station {i % max(1, np.shape(a)[-1])}" if np.size(a) > 1
                           else ""))


def _meaning(name):
    for table in (COMMON, SIDE):
        if name in table:
            return table[name][1]
    return SIDE_TEXT[name][1]


def _text_check(name, a, at, problems):
    if name == "hazard":
        bad = sorted({str(v) for v in np.ravel(a)} - set(HAZARDS) - {""})  # Python strs (090)
        if bad:
            problems.append(f"{at}: hazard {bad[0]!r} is not one of {', '.join(HAZARDS)}")


def from_constants(given, at="section"):
    """One station's section from a `section:` mapping (a side's field: a number, or
    `{left, right}`); returns `(Stations of one station at u 0, problems)`."""
    problems = []
    given = given or {}
    if not isinstance(given, dict):
        return None, [f"{at}: a mapping of {', '.join(_all())}"]
    unknown = sorted(set(given) - set(_all()))
    if unknown:
        problems.append(f"{at}: unknown {unknown} ({', '.join(_all())})")
    common, side, text = {}, {}, {}
    for k, (d, _, check) in COMMON.items():
        v = given.get(k, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            problems.append(f"{at}.{k}: a number, not {v!r}")
            v = d
        common[k] = np.array([float(v)])
        _check_values(k, common[k], check, f"{at}", problems)
    for table, store, kind in ((SIDE, side, "number"), (SIDE_TEXT, text, "text")):
        for k, spec in table.items():
            d = spec[0]
            v = given.get(k, d)
            pair = _pair(v, d, kind, f"{at}.{k}", problems)
            store[k] = np.array([[pair[0]], [pair[1]]],
                                dtype=float if kind == "number" else str)
            if kind == "number":
                _check_values(k, store[k], spec[2], f"{at}", problems)
            else:
                _text_check(k, store[k], f"{at}", problems)
    return Stations(np.array([0.0]), common, side, text), problems


def _pair(v, d, kind, at, problems):
    def ok(x):
        if kind == "number":
            return not isinstance(x, bool) and isinstance(x, (int, float))
        return isinstance(x, str)
    if isinstance(v, dict):
        bad = sorted(set(v) - set(SIDES))
        if bad:
            problems.append(f"{at}: unknown {bad} (left, right)")
        out = [v.get(s, d) for s in SIDES]
    else:
        out = [v, v]
    for i, x in enumerate(out):
        if not ok(x):
            problems.append(f"{at}: a {kind} or {{left, right}}, not {x!r}")
            out[i] = d
    return [float(x) for x in out] if kind == "number" else out


def _all():
    return list(COMMON) + list(SIDE) + list(SIDE_TEXT)


def load_stations(path, length=None, at=None):
    """A corridor's stations from its .npz: `(Stations, problems)`. `length`: the
    centreline's, m; the last station must lie on it."""
    at = at or str(path)
    problems = []
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError) as e:
        return None, [f"{at}: cannot read: {e}"]
    names = set(COMMON) | {f"{k}_{s}" for k in list(SIDE) + list(SIDE_TEXT) for s in SIDES}
    unknown = sorted(set(arrays) - names - {"u", "x", "y", "z"})
    if unknown:
        problems.append(f"{at}: unknown array(s) {unknown} (u, x, y, z, "
                        f"{', '.join(sorted(names))})")
    u = arrays.get("u")
    if u is None or u.ndim != 1 or len(u) < 2 or u.dtype.kind not in "fi":
        return None, problems + [f"{at}: u must be two or more numbers, m along the "
                                 "centreline"]
    u = u.astype(float)
    if u[0] != 0 or np.any(np.diff(u) <= 0):
        problems.append(f"{at}: u must start at 0 and increase")
    if length is not None and u[-1] > length + 0.5:
        problems.append(f"{at}: the last station, u {u[-1]:g}, lies past the centreline's "
                        f"end ({length:g} m)")
    n = len(u)

    def get(name, d, kind):
        a = arrays.get(name)
        if a is None:
            return np.full(n, d, dtype=float if kind == "number" else object).astype(
                float if kind == "number" else str)
        if a.shape != (n,):
            problems.append(f"{at}: {name} has shape {a.shape}, u has {n} stations")
            return np.full(n, d).astype(float if kind == "number" else str)
        if kind == "number" and a.dtype.kind not in "fi":
            problems.append(f"{at}: {name} must be numbers")
            return np.full(n, d, dtype=float)
        if kind == "text" and a.dtype.kind != "U":
            problems.append(f"{at}: {name} must be text")
            return np.full(n, d).astype(str)
        return a.astype(float) if kind == "number" else a

    common = {k: get(k, d, "number") for k, (d, _, _) in COMMON.items()}
    for k, (_, _, check) in COMMON.items():
        _check_values(k, common[k], check, at, problems)
    side = {k: np.stack([get(f"{k}_{s}", d, "number") for s in SIDES])
            for k, (d, _, _) in SIDE.items()}
    for k, (_, _, check) in SIDE.items():
        _check_values(k, side[k], check, at, problems)
    text = {k: np.stack([get(f"{k}_{s}", d, "text") for s in SIDES])
            for k, (d, _) in SIDE_TEXT.items()}
    for k in SIDE_TEXT:
        _text_check(k, text[k], at, problems)
    xyz = None
    if "x" in arrays or "y" in arrays:
        xyz, why = _centreline(arrays, u)
        if why:
            problems.append(f"{at}: {why}")
            xyz = None
    elif "z" in arrays:
        problems.append(f"{at}: z without x and y")
    return Stations(u, common, side, text, xyz), problems


def _centreline(arrays, u):
    """The record's own centreline (n, 3) and why it is refused (or "")."""
    n = len(u)
    cols = []
    for k in ("x", "y", "z"):
        a = arrays.get(k)
        if a is None:
            if k == "z":
                cols.append(np.zeros(n))
                continue
            return None, "x and y go together"
        if a.shape != (n,) or a.dtype.kind not in "fi" or not np.all(np.isfinite(a)):
            return None, f"{k} must be {n} finite numbers, one per station"
        cols.append(a.astype(float))
    xyz = np.stack(cols, axis=1)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xyz[:, :2], axis=0).T))])
    off = float(np.max(np.abs(along - u)))
    if off > 0.05 + 1e-3 * along[-1]:
        return None, (f"u is not the distance along x, y (off by up to {off:.3f} m; "
                      f"allowed {0.05 + 1e-3 * along[-1]:.3f})")
    return xyz, ""


def save_stations(path, st):
    """Write `st` as a corridor's .npz (every field, both sides; its centreline, if any)."""
    arrays = {"u": np.asarray(st.u, dtype=float)}
    if st.xyz is not None:
        arrays.update({k: np.asarray(st.xyz[:, i], dtype=float) for i, k in enumerate("xyz")})
    arrays.update({k: np.asarray(v, dtype=float) for k, v in st.common.items()})
    for k, v in st.side.items():
        for i, s in enumerate(SIDES):
            arrays[f"{k}_{s}"] = np.asarray(v[i], dtype=float)
    for k, v in st.text.items():
        for i, s in enumerate(SIDES):
            arrays[f"{k}_{s}"] = np.asarray(v[i]).astype(str)
    np.savez(path, **arrays)


def detail_section(st, cls):
    """The one section (`cls`, road_detail.Section) stations `st` reduce to, or why they do
    not: `(Section or None, reason)`. A record that does not (banked, an offset crown, a lip,
    or a shape that changes along the road or across it) is shaped per station by
    road_detail.StationSection (086 C5)."""
    why = []
    for k in DETAIL:
        if not st.constant(k):
            why.append(f"{k} changes along the road or between the sides")
    for k, zero in (("bank", "banked"), ("crown_offset", "an offset crown")):
        if np.any(st.common[k] != 0):
            why.append(f"it is {zero}")
    if np.any(st.side["lip_h"] != 0):
        why.append("it has a lip")
    if why:
        return None, "; ".join(why)
    first = {k: float(st.common[k][0]) if k in st.common else float(st.side[k][0, 0])
             for k in DETAIL}
    return cls(**first), ""
