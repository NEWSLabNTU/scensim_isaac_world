"""086 C5b: a fixture course, its centreline and its road record from the course's `fixture:`
block instead of OSM and a DEM (`courses/fixture_c5.yaml`).

The plan is a list of straights, clothoids (curvature linear in s) and arcs, integrated from the
origin heading east; the profile one grade. The section is a table of rows in s, linear between
them, and becomes the record (`road_record.Stations`) at every centreline station: the record is
the spec, which the road mesh then carries (`geometry.road`'s `section`). The ground is flat
across the road, `terrain_below` m under its centreline, cut to the road as a real course's is
(`geometry.terrain`), and the photo plain grey.
"""
import math
import os

import numpy as np

from .. import road_record as rr
from .centreline import STEP

FINE = 0.01  # m: the plan's integration step
GREY = (128, 128, 128)
# the section table's columns: (record field, side or None, slope as tan -> angle)
COLUMNS = {
    "bank": ("bank", None, True),
    "crown_left": ("crown", 0, False),
    "crown_right": ("crown", 1, False),
    "crown_offset": ("crown_offset", None, False),
    "lip_h_left": ("lip_h", 0, False),
    "lip_h_right": ("lip_h", 1, False),
}


def is_fixture(course):
    return "fixture" in course


def _curvature(plan):
    """The plan as (s ends, curvature at each end) for np.interp: piecewise linear in s."""
    s, k = [0.0], [0.0]
    for seg in plan:
        (kind, length), = [(n, float(v)) for n, v in seg.items() if n != "to"]
        if length <= 0:
            raise ValueError(f"fixture plan: {seg}: a length > 0")
        if kind == "straight":
            s += [s[-1], s[-1] + length]
            k += [0.0, 0.0]
        elif kind == "arc":
            s.append(s[-1] + length)
            k.append(k[-1])
        elif kind == "clothoid":
            s.append(s[-1] + length)
            k.append(float(seg["to"]))
        else:
            raise ValueError(f"fixture plan: {seg}: straight, clothoid or arc")
    return np.array(s), np.array(k)


def centreline(course):
    """centreline.build's dict for the fixture: s, x, y, z, z_dem, heading, curvature, grade
    and no hairpins (HAIRPIN_TURN is not reached: the plan says what turns)."""
    fx = course["fixture"]
    se, ke = _curvature(fx["plan"])
    length = float(se[-1])
    sf = np.linspace(0.0, length, int(round(length / FINE)) + 1)
    kf = np.interp(sf, se, ke)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (kf[1:] + kf[:-1]) * np.diff(sf))])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(h[1:]) + np.cos(h[:-1])) * np.diff(sf))])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(h[1:]) + np.sin(h[:-1])) * np.diff(sf))])
    s = np.arange(0.0, length + 1e-9, STEP)
    grade = float(fx["grade"])
    z = grade * s
    return {"s": s, "x": np.interp(s, sf, x), "y": np.interp(s, sf, y), "z": z, "z_dem": z.copy(),
            "heading": np.interp(s, sf, h), "curvature": np.interp(s, sf, kf),
            "grade": np.full(len(s), grade), "hairpins": []}


def stations(course, c):
    """The fixture's record at every station of `c`: road_record.Stations."""
    from .record import vertical_curvature

    fx, r = course["fixture"], course["road"]
    st, problems = rr.from_constants({
        "half_width": r["lanes"] * r["lane_width"] / 2, "shoulder": float(r["shoulder"]),
        "crown": 0.0, "ditch_depth": 0.0, "ditch_bottom": 0.0,
        "lip_angle": {"left": float(fx.get("lip_angle_left", 90.0)),
                      "right": float(fx.get("lip_angle_right", 90.0))}})
    assert not problems, problems
    u = np.asarray(c["s"], float) - float(c["s"][0])
    n = len(u)
    common = {k: np.repeat(v, n).astype(float) for k, v in st.common.items()}
    common["kappa_v"] = vertical_curvature(c)
    side = {k: np.repeat(v, n, axis=1).astype(float) for k, v in st.side.items()}
    text = {k: np.repeat(v, n, axis=1) for k, v in st.text.items()}
    cols = fx["section"]["columns"]
    rows = np.asarray(fx["section"]["rows"], float)
    if cols[0] != "s" or rows.shape[1] != len(cols) or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("fixture section: columns start with s, rows increase in s")
    for j, name in enumerate(cols[1:], 1):
        if name not in COLUMNS:
            raise ValueError(f"fixture section: column {name!r} ({', '.join(COLUMNS)})")
        field, sd, angle = COLUMNS[name]
        val = np.interp(np.asarray(c["s"], float), rows[:, 0], rows[:, j])  # the table's s
        if angle:
            val = np.arctan(val)
        if sd is None:
            common[field] = val
        else:
            side[field][sd] = val
    xyz = np.stack([c["x"], c["y"], c["z"]], axis=1)
    return rr.Stations(u, common, side, text, xyz)


class Dem:
    """The fixture's ground: flat across the road, `terrain_below` m under its nearest
    centreline point (data.Dem's `at`)."""

    def __init__(self, course, c, frame):
        from scipy.spatial import cKDTree

        self.frame = frame
        self.c = c
        self.below = float(course["fixture"].get("terrain_below", 0.3))
        self.tree = cKDTree(np.stack([c["x"], c["y"]], axis=1))

    def at(self, lat, lon):
        x, y = self.frame.xy(lat, lon)
        _, j = self.tree.query((x, y))
        return float(self.c["z"][j]) - self.below + self.frame.alt0


def photo(out, size=64):
    """A plain grey photo (the terrain's texture)."""
    from PIL import Image

    Image.new("RGB", (size, size), GREY).save(out, quality=90)
    return out


def box(c, margin):
    return (float(c["x"].min() - margin), float(c["x"].max() + margin),
            float(c["y"].min() - margin), float(c["y"].max() + margin))


def summary(c):
    turn = math.degrees(float(c["heading"][-1] - c["heading"][0]))
    k = np.abs(c["curvature"]).max()
    return {"length_m": round(float(c["s"][-1]), 1), "turn_deg": round(turn, 1),
            "min_radius_m": round(1.0 / k, 1) if k > 0 else None,
            "z_from_m": round(float(c["z"][0]), 2), "z_to_m": round(float(c["z"][-1]), 2)}


def write_centreline(work, c):
    import json

    np.savez(os.path.join(work, "centreline.npz"), **{k: v for k, v in c.items() if k != "hairpins"})
    s = summary(c)
    json.dump(dict(s, hairpins=[], fixture=True), open(os.path.join(work, "centreline.json"), "w"),
              indent=1)
    return s
