"""The course's road record (scensim roadmap 086 C3, 069 E0): its section at every centreline
station, and what lies past each edge, written to `<work>/road_detail.yaml` and
`<work>/section.npz` (`road_record`'s format, the centreline carried in the .npz).

It is the course's record, in `course/`, for the rails (`props.guardrails`), the planner and
the steps after this one: the world server reads a map's road detail from `<map
dir>/road_detail.yaml` only, so this file switches nothing on in Isaac.

The section is the builder's own: level across (bank and crown 0, `geometry.ribbon`), the
carriageway `lanes * lane_width / 2` each side of the centreline and the course's shoulder,
no lip and no ditch (the terrain is cut and filled to the road and blends back to the DEM,
`geometry.terrain`). `kappa_v` is the vertical curvature of the smoothed profile.

What lies past each edge comes from one classifier, `classify`, which keeps the guardrail rule
`props` had (ground more than `props.DROP` m below the road `props.DROP_AT` m past the shoulder,
or the outside of a bend tighter than `props.RAIL_RADIUS` m, gets a rail at `props.RAIL_OFFSET`
past the shoulder) and writes the answer down: `barrier` there; elsewhere, by the ground's
slope over those `DROP_AT` m, `runoff` at 1V:4H or flatter (069's recoverable class) and
`ditch` steeper, down or up (a cut face stops a car as a ditch does).
"""
import numpy as np

from .. import road_record as rr
from . import props as pr
from .geometry import _normals, is_closed

YAML = "road_detail.yaml"
NPZ = "section.npz"
RUNOFF_SLOPE = 0.25  # V:H, 1V:4H: flatter is a recoverable run-off (AASHTO; 069)


def vertical_curvature(c):
    """kappa_v (1/m) of the profile, positive in a sag."""
    s, z = np.asarray(c["s"], float), np.asarray(c["z"], float)
    if is_closed(c):
        h = float(s[1] - s[0])
        dz = (np.roll(z, -1) - np.roll(z, 1)) / (2 * h)
        ddz = (np.roll(z, -1) - 2 * z + np.roll(z, 1)) / (h * h)
    else:
        dz = np.gradient(z, s)
        ddz = np.gradient(dz, s)
    return ddz / (1.0 + dz * dz) ** 1.5


def classify(c, m, half):
    """Per side (left, right) and station: the ground's height `props.DROP_AT` m past `half`
    (the paved half width), whether a rail goes there (props' rule), and the hazard class."""
    tz = pr._terrain_z(m)
    x, y, z, k = c["x"], c["y"], c["z"], c["curvature"]
    nrm = _normals(x, y, is_closed(c))
    n = len(x)
    rail = np.zeros((2, n), bool)
    hazard = np.empty((2, n), dtype="<U8")
    for i, side in enumerate((1, -1)):  # 1: left of travel (the normals' side)
        probe = side * (half + pr.DROP_AT)
        gz = np.asarray(tz(x + probe * nrm[:, 0], y + probe * nrm[:, 1]), float)
        outer = (np.abs(k) > 1.0 / pr.RAIL_RADIUS) & (np.sign(k) == -side)
        rail[i] = (gz < z - pr.DROP) | outer
        slope = np.abs(z - gz) / pr.DROP_AT
        hazard[i] = np.where(rail[i], "barrier",
                             np.where(slope <= RUNOFF_SLOPE, "runoff", "ditch"))
    return rail, hazard


def build(c, course, m):
    """The course's record: road_record.Stations over the centreline's stations."""
    r = course["road"]
    carriage = r["lanes"] * r["lane_width"] / 2
    shoulder = float(r["shoulder"])
    half = float(m["half_width"])  # geometry.road's: carriageway and shoulder
    if abs(half - (carriage + shoulder)) > 1e-6:
        raise ValueError(f"meshes.npz's half width {half} is not the course's "
                         f"{carriage} + {shoulder} m (rebuild the geometry step)")
    st, problems = rr.from_constants({"half_width": carriage, "crown": 0.0,
                                      "shoulder": shoulder, "ditch_depth": 0.0,
                                      "ditch_bottom": 0.0})
    assert not problems, problems
    x, y, z = (np.asarray(c[k], float) for k in ("x", "y", "z"))
    u = np.asarray(c["s"], float) - float(c["s"][0])
    n = len(u)
    rail, hazard = classify(c, m, half)
    common = {k: np.repeat(v, n) for k, v in st.common.items()}
    common["kappa_v"] = vertical_curvature(c)
    side = {k: np.repeat(v, n, axis=1) for k, v in st.side.items()}
    side["barrier_offset"] = np.where(rail, half + pr.RAIL_OFFSET, np.nan)
    text = {k: np.repeat(v, n, axis=1) for k, v in st.text.items()}
    text["hazard"] = hazard
    xyz = np.stack([x, y, z], axis=1)
    if is_closed(c):  # the loop closed: its first station again, a lap on
        step = float(u[1] - u[0])
        u = np.append(u, u[-1] + float(np.hypot(*(xyz[0, :2] - xyz[-1, :2]))))
        assert abs(u[-1] - u[-2] - step) < 0.05 * step
        xyz = np.vstack([xyz, xyz[:1]])
        common = {kk: np.append(v, v[0]) for kk, v in common.items()}
        side = {kk: np.concatenate([v, v[:, :1]], axis=1) for kk, v in side.items()}
        text = {kk: np.concatenate([v, v[:, :1]], axis=1) for kk, v in text.items()}
    return rr.Stations(u, common, side, text, xyz)


def write(work, st, in_mesh=False, features=None):
    """`<work>/road_detail.yaml` and `<work>/section.npz`; returns a short summary. `in_mesh`:
    the course's road mesh carries the section (a fixture's, 086 C5b), which the YAML says, so
    the road layer given it adds the features only. `features`: the corridor's road-detail
    features (a fixture's `features:`, 086 C6a: cleats, potholes, washboard, ...), as
    road_layer reads them."""
    import os

    rr.save_stations(os.path.join(work, NPZ), st)
    with open(os.path.join(work, YAML), "w") as f:
        f.write("# The course's road record (scensim roadmap 086 C3), written by the course\n"
                "# builder's record step: the section per station and the centreline in\n"
                f"# {NPZ}. In course/, so it is not the map's road detail (the world server\n"
                "# reads <map dir>/road_detail.yaml).\n"
                f"schema: {rr.SCHEMA}\ncorridors:\n  - stations: {NPZ}\n"
                + ("    section_in_base: true   # the road mesh carries the section (086 C5b)\n"
                   if in_mesh else ""))
        if features:
            import yaml

            f.write("    # the fixture's features (086 C6a)\n")
            for line in yaml.safe_dump(features, default_flow_style=None,
                                       sort_keys=False).splitlines():
                f.write(f"    {line}\n")
    out = {"stations": len(st), "length_m": round(float(st.u[-1]), 1)}
    for i, s in enumerate(rr.SIDES):
        h = st.text["hazard"][i]
        out[f"hazard_{s}"] = {k: int(np.sum(h == k)) for k in rr.HAZARDS if np.any(h == k)}
    return out


def load(work):
    """The course's record from `<work>`: road_record.Stations; refused, naming the fix, when
    absent or bad."""
    import os

    path = os.path.join(work, NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no road record (run the course builder's record "
                                "step: python3 -m scensim_isaac_world.course <course.yaml> "
                                "<map dir> record)")
    st, problems = rr.load_stations(path, at=path)
    if problems:
        raise ValueError("\n".join(problems))
    return st


MAP_MARK = "# scensim course record step: course "   # the map file's own line (scensim issue 177)


def write_map(map_dir, course_name, features):
    """scensim issue 177: `<map dir>/road_detail.yaml`, the map's road detail, for a fixture
    with features: the course record's corridor (its stations in course/, the section in the
    base) and the features, which the world server reads (it reads the map's file only; the
    record in course/ stays the course's). A map file this step wrote for this course is
    written again; any other is refused (SystemExit naming it), never overwritten. None
    without features; else the path written."""
    import os

    if not features:
        return None
    path = os.path.join(map_dir, YAML)
    mark = f"{MAP_MARK}{course_name}\n"
    if os.path.exists(path):
        with open(path) as f:
            head = [ln for ln in f.readlines()[:8] if ln.startswith(MAP_MARK)]
        if head != [mark]:
            owner = head[0][len(MAP_MARK):].strip() if head else None
            raise SystemExit(
                f"{path}: not written by this record step for course {course_name} "
                + (f"(it names course {owner})" if owner else "(no record-step header)")
                + "; refusing to overwrite it (scensim issue 177): move it away or merge "
                  f"by hand from {os.path.join('course', YAML)}")
    import yaml

    entry = {"stations": f"course/{NPZ}", "section_in_base": True}
    entry.update(features)
    with open(path, "w") as f:
        f.write("# The map's road detail (scensim issue 177), which the world server reads:\n"
                "# the course record's corridor (course/road_detail.yaml, the record, as it\n"
                "# was) and the fixture's features. Written again by the record step; do not\n"
                "# edit (edit the course's features). Its base_drop (issue 168): python3 -m\n"
                "# scensim_isaac_world.base_drop <map dir> --write (numpy, seconds)\n"
                + mark +
                f"schema: {rr.SCHEMA}\ncorridors:\n")
        for i, line in enumerate(yaml.safe_dump(entry, default_flow_style=None,
                                                sort_keys=False).splitlines()):
            f.write(("  - " if i == 0 else "    ") + line + "\n")
    return path
