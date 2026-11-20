"""What an authored stage is, for a run's record and for the map check (roadmap 091 W15, W16;
design roboracer-indoor-twin.md 3.4), read with pxr alone (no Isaac app state), as `authored`.

`record` gives `provenance.stage`: every layer the stage composes (its layer stack, references,
payloads) and every asset it resolves, each by the sha256 of its content, one sha256 over them
all, the overrides of the run and the hashes of the map's records. An unresolved dependency
raises `Unresolved`, naming it. The world server computes it once per load of an authored world
and writes it beside the map's state (`write`), where the run reads it.

`check` gives what the map check refuses in an authored world: units other than metres, an up
axis other than Z, a PhysX vehicle prim, an unresolved dependency, and what `authored.problems`
finds (an unknown class token, a friction out of range).
"""
import hashlib
import json
import os
import tempfile
import time

from pxr import Sdf, Usd, UsdGeom, UsdUtils

from . import authored

FILE = "stage.json"  # in the map's state directory
CACHE = "stage-hashes.json"  # beside it: {path: [size, mtime_ns, sha256]}
CHUNK = 1 << 20


def _umask_mode():
    """0666 less the umask: what open() would give, since mkstemp makes its file 0600."""
    m = os.umask(0)
    os.umask(m)
    return 0o666 & ~m


class Unresolved(ValueError):
    """A dependency of the stage that does not resolve to a file."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class HashCache:
    """File hashes kept by (path, size, mtime): a scanned field can be gigabytes, and only a
    changed file is hashed again. `path`: the cache file (None: nothing kept between loads).
    `hashed` counts the files read this time."""

    def __init__(self, path=None):
        self.path = path
        self.table = {}
        self.hashed = 0
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    self.table = {k: tuple(v) for k, v in json.load(f).items()}
            except (OSError, ValueError, TypeError):
                self.table = {}  # a torn or foreign file: hash everything again

    def __call__(self, path):
        st = os.stat(path)
        hit = self.table.get(path)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2]
        sha = file_sha256(path)
        self.hashed += 1
        self.table[path] = (st.st_size, st.st_mtime_ns, sha)
        return sha

    def save(self):
        if not self.path:
            return
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".hashes.", dir=d)
        with os.fdopen(fd, "w") as f:
            json.dump({k: list(v) for k, v in sorted(self.table.items())}, f)
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, self.path)


def _local(path):
    """The file behind a layer's or asset's resolved path, or None (an in-memory layer, a
    package member)."""
    path = str(path or "")
    if not path or path.startswith("anon:") or "[" in path:
        return None
    return path if os.path.isfile(path) else None


def _name(path, base):
    """`path` relative to `base` when under it, else absolute: what the record names a file by."""
    rel = os.path.relpath(path, base)
    return path if rel.startswith("..") else rel


def dependencies(root, sha=file_sha256):
    """`(layers, assets)` of the stage file `root`, each `{file: sha256}` keyed by the path
    relative to the root's directory (absolute outside it); `sha` hashes a file (a
    `HashCache`). Raises `Unresolved` naming every dependency that does not resolve."""
    root = os.path.realpath(root)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(root))
    bad = sorted(str(u) for u in unresolved)
    files = []
    for layer in layers:
        path = _local(layer.realPath)
        if path is None:
            if layer.anonymous:
                continue
            bad.append(layer.identifier)
        else:
            files.append(("layers", path))
    for asset in assets:
        path = _local(asset)
        if path is None:
            bad.append(str(asset))
        else:
            files.append(("assets", path))
    if bad:
        raise Unresolved(f"{root}: unresolved: " + ", ".join(sorted(set(bad))))
    base = os.path.dirname(root)
    out = {"layers": {}, "assets": {}}
    for kind, path in files:
        out[kind][_name(path, base)] = sha(path)
    return out["layers"], out["assets"]


def digest(layers, assets):
    """One sha256 over the sorted (name, sha256) pairs of the layers and the assets."""
    h = hashlib.sha256()
    for kind, table in (("layer", layers), ("asset", assets)):
        for name in sorted(table):
            h.update(f"{kind}\0{name}\0{table[name]}\n".encode())
    return h.hexdigest()


def record(root, overrides=None, records=None, cache=None):
    """`provenance.stage` for the stage file `root`: `{root, path, sha256, layers, assets,
    overrides, records, hashing}`. `overrides`: `{attribute: {"authored", "used"}}` the run
    replaced (the world's `stage_overrides`); `records`: `{name: path}` of the map's Track C
    records, each hashed, or None where the file does not exist; `cache`: a `HashCache` (None:
    every file hashed). `hashing`: the seconds it took and the files read anew."""
    root = os.path.realpath(root)
    cache = cache if cache is not None else HashCache()
    t0 = time.monotonic()
    layers, assets = dependencies(root, cache)
    cache.save()
    hashing = {"seconds": round(time.monotonic() - t0, 3), "files_hashed": cache.hashed,
               "files": len(layers) + len(assets)}
    return {"root": os.path.basename(root),
            "path": root,
            "sha256": digest(layers, assets),
            "layers": layers,
            "assets": assets,
            "overrides": dict(overrides or {}),
            "records": {name: file_sha256(p) if p and os.path.isfile(p) else None
                        for name, p in sorted((records or {}).items())},
            "hashing": hashing}


def write(directory, rec):
    """`rec` into `<directory>/stage.json`, atomically (a run reading it never sees half)."""
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".stage.", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, os.path.join(directory, FILE))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return os.path.join(directory, FILE)


# A PhysX vehicle prim in the field: the car comes from the spec, never from the map (3.1).
VEHICLE_PREFIX = "PhysxVehicle"


def check(root, classes):
    """What the map check refuses in the authored stage file `root`, each a line naming its
    prim or file: `[]` when it passes. `classes`: the surface class names a token may name."""
    root = os.path.realpath(root)
    try:
        stage = Usd.Stage.Open(root)
    except Exception as e:  # pxr raises Tf errors as plain exceptions
        return [f"{root}: does not open: {e}"]
    out = []
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(mpu - 1.0) > 1e-9:
        out.append(f"{root}: metersPerUnit is {mpu:g}, not 1 (the field is in metres)")
    up = UsdGeom.GetStageUpAxis(stage)
    if up != UsdGeom.Tokens.z:
        out.append(f"{root}: upAxis is {up}, not Z")
    for prim in stage.Traverse():
        # the authored list, not GetAppliedSchemas: that drops schemas pxr has no plugin for
        # (usd-core has none of PhysX's)
        listop = prim.GetMetadata("apiSchemas")
        names = list(listop.GetAddedOrExplicitItems()) if listop else []
        apis = [str(a) for a in names if str(a).startswith(VEHICLE_PREFIX)]
        if apis:
            out.append(f"{prim.GetPath()}: {', '.join(apis)}: a PhysX vehicle in the field "
                       "(the car comes from its spec, not the map)")
    out += authored.problems(stage, classes)
    try:
        dependencies(root)
    except Unresolved as e:
        out.append(str(e))
    return out


def classes_of(surfaces_file=None):
    """The class names a `scensim:surface` token may name for a map: its `surfaces.yaml`'s
    table when it has one, else the shared class table."""
    import os.path

    from . import surfaces
    if surfaces_file and os.path.isfile(surfaces_file):
        return surfaces.load(surfaces_file).table.classes
    return surfaces.load_classes().classes


def check_map(world_file, state_dir=None):
    """The map check of a map's `isaac/world.yaml`: `{"map", "authored", "stage",
    "problems"}`; a map that is not `authored: true` is not checked (`problems` empty). Its
    `surfaces.yaml` is the state directory's over the map directory's, as the server's."""
    import yaml

    from . import names

    with open(world_file) as f:
        w = yaml.safe_load(f) or {}
    map_dir = names.map_dir_of(world_file)
    out = {"map": names.map_name(map_dir), "authored": w.get("authored", False) is True,
           "stage": None, "problems": []}
    if not out["authored"]:
        return out
    try:
        usd = names.world_usd_of(world_file, w)
    except names.UnresolvedName as e:
        out["problems"].append(f"{world_file}: {e}")
        return out
    out["stage"] = usd
    if not usd or not os.path.isfile(usd):
        out["problems"].append(f"{world_file}: usd: {usd or 'none'}: no such file")
        return out
    surfaces_file = names.find_file("surfaces.yaml", map_dir,
                                    names.state_for(state_dir, map_dir))
    out["problems"] = check(usd, classes_of(surfaces_file))
    return out


def main(argv=None):
    """`python -m scensim_isaac_world.stage_record map <world.yaml> [--state-dir <dir>]`: the
    map check of a map (`check_map`) as one JSON line. `check <stage> [--surfaces
    <surfaces.yaml>]`: a stage's problems, `{"problems": [...]}`. Both exit 0 whatever they
    find. `record <stage> [--surfaces ...] [--road ...]`: `provenance.stage` as JSON (exit 2 on
    an unresolved dependency). Any Python with pxr runs them (Isaac's, or usd-core with numpy
    and PyYAML)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="stage_record")
    ap.add_argument("what", choices=("map", "check", "record"))
    ap.add_argument("stage")
    ap.add_argument("--surfaces")
    ap.add_argument("--road")
    ap.add_argument("--state-dir")
    a = ap.parse_args(argv)
    if a.what == "map":
        print(json.dumps(check_map(a.stage, a.state_dir)))
        return 0
    if a.what == "check":
        print(json.dumps({"problems": check(a.stage, classes_of(a.surfaces))}))
        return 0
    records = {"surfaces.yaml": a.surfaces, "road_detail.yaml": a.road}
    try:
        print(json.dumps(record(a.stage, records=records), indent=2, sort_keys=True))
    except Unresolved as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
