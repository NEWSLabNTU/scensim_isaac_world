"""The ground an authored stage gives a collider (roadmap 091 W13; design note
roboracer-indoor-twin.md 3.3), read with pxr alone: no Isaac app state, so a tool outside the
world server (a map check, a training stage clone) reads a stage the same way.

A map whose `isaac/world.yaml` says `authored: true` keeps what was authored in its stage. For
the ground under a wheel the order is, highest first:

1. a Track C zone (`surfaces.yaml`) over the point: `surfaces.SurfaceMap.at`, not here;
2. `scensim:surface` (a token naming a class of the class table) on the collider or its
   nearest ancestor that has one;
3. the physics material bound to the collider (purpose `physics`, else all-purpose, as
   `UsdShade.MaterialBindingAPI.ComputeBoundMaterial` resolves it): its `dynamicFriction`,
   taken as our tires' peak friction on that ground on the class table's scale (the wheel
   layer divides it by the reference class's `mu`, as it does a class's); contacts other
   than a tire on the ground use the material as authored, by PhysX alone;
4. nothing authored: the label's default class (the reference class unless a surface map or
   the scenario says otherwise), source `no_map`.

`GroundReader` gives sources 2 to 4 for a collider path; `problems` lists what an authored
stage holds that the reader would refuse (an unknown class token, a friction out of range), for
the world's load and for the map check.
"""
from dataclasses import dataclass
from typing import Optional

from pxr import Usd, UsdPhysics, UsdShade

SURFACE_ATTR = "scensim:surface"
# The material purpose Isaac binds physics materials with (`material:binding:physics`)
PHYSICS_PURPOSE = "physics"
# Which of the four sources decided a wheel's ground (`surfaces.Surface.source`); a map that
# is not authored gives `label` (the ground's SimReady label or none) instead of 2 to 4.
SOURCES = ("zone", "token", "material", "no_map")
# A tire's peak friction on an indoor floor: anything outside is a typo (a percentage, a sign).
MU_RANGE = (0.05, 3.0)


@dataclass(frozen=True)
class Ground:
    """What a collider's authoring says of its ground: `source` (`token`, `material` or
    `no_map`), the class (`token`), the peak friction (`material`), and the prim it came from
    (the collider, an ancestor, or the material)."""

    source: str
    cls: Optional[str] = None
    mu: Optional[float] = None
    prim: Optional[str] = None


NOTHING = Ground("no_map")


def surface_token(prim):
    """`(token, prim path)` of the nearest `scensim:surface` on `prim` or an ancestor, or None."""
    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        attr = prim.GetAttribute(SURFACE_ATTR)
        if attr and attr.HasAuthoredValue():
            return str(attr.Get()), str(prim.GetPath())
        prim = prim.GetParent()
    return None


def physics_friction(prim):
    """`(dynamicFriction, material path)` of the physics material bound to `prim`, or None
    when none is bound or the bound one is not a physics material."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        PHYSICS_PURPOSE)
    if not material:
        return None
    mprim = material.GetPrim()
    if not mprim.HasAPI(UsdPhysics.MaterialAPI):
        return None
    attr = UsdPhysics.MaterialAPI(mprim).GetDynamicFrictionAttr()
    if not attr or not attr.HasAuthoredValue():
        return None
    return round(float(attr.Get()), 6), str(mprim.GetPath())  # float32 as authored


class GroundReader:
    """The authored ground of a collider path on `stage` (`__call__`), cached per path: a
    LoadWorld makes a new reader, so an edit in Isaac's editor counts from the next load.
    `classes`: the class names a token may name."""

    def __init__(self, stage, classes):
        self.stage = stage
        self.classes = frozenset(classes)
        self.cache = {}

    def __call__(self, path):
        g = self.cache.get(path)
        if g is None:
            g = self.cache[path] = self._read(path)
        return g

    def _read(self, path):
        prim = self.stage.GetPrimAtPath(path) if path else None
        if not prim or not prim.IsValid():
            return NOTHING
        tok = surface_token(prim)
        if tok is not None:
            name, where = tok
            if name not in self.classes:
                # `problems` refuses the stage at load; one edited since is refused here
                raise ValueError(f"{where}: {SURFACE_ATTR} {name!r} is not a surface class "
                                 f"(the table has {', '.join(sorted(self.classes))})")
            return Ground("token", cls=name, prim=where)
        mat = physics_friction(prim)
        if mat is not None:
            mu, where = mat
            lo, hi = MU_RANGE
            if not lo <= mu <= hi:
                raise ValueError(f"{where}: dynamicFriction {mu:g} out of range "
                                 f"({lo:g} to {hi:g}, the tire's peak friction on this ground)")
            return Ground("material", mu=mu, prim=where)
        return NOTHING


def problems(stage, classes, root=None):
    """Every problem the reader would refuse on `stage` (under `root`, a path; None: all):
    `scensim:surface` tokens not in `classes`, physics materials whose `dynamicFriction` is
    negative or above `MU_RANGE`'s top. Each once, naming its prim."""
    out = []
    classes = frozenset(classes)
    top = stage.GetPrimAtPath(root) if root else stage.GetPseudoRoot()
    if not top or not top.IsValid():
        return [f"{root}: no such prim"]
    seen = set()
    for prim in Usd.PrimRange(top):
        attr = prim.GetAttribute(SURFACE_ATTR)
        if attr and attr.HasAuthoredValue():
            name = str(attr.Get())
            if name not in classes:
                out.append(f"{prim.GetPath()}: {SURFACE_ATTR} {name!r} is not a surface class "
                           f"(the table has {', '.join(sorted(classes))})")
        if prim.HasAPI(UsdPhysics.MaterialAPI) and str(prim.GetPath()) not in seen:
            seen.add(str(prim.GetPath()))
            a = UsdPhysics.MaterialAPI(prim).GetDynamicFrictionAttr()
            if a and a.HasAuthoredValue():
                # any material, a wall's or a cone's too (0 allowed there); the floor's lower
                # bound is the reader's, for the ground a wheel is on
                mu = float(a.Get())
                if not 0.0 <= mu <= MU_RANGE[1]:
                    out.append(f"{prim.GetPath()}: dynamicFriction {mu:g} out of range "
                               f"(0 to {MU_RANGE[1]:g})")
    return out


def authored_scenes(stage, root):
    """The physics scenes authored under `root` (a path): `[(path, gravity direction, gravity
    magnitude, timeStepsPerSecond or None)]`, each value as authored (None where it is not).
    The world server runs its own scene (scensim owns time); an authored one is turned off for
    the run and its gravity applied to the server's (design 3.2)."""
    top = stage.GetPrimAtPath(root)
    out = []
    if not top or not top.IsValid():
        return out
    for prim in Usd.PrimRange(top):
        if not prim.IsA(UsdPhysics.Scene):
            continue
        scene = UsdPhysics.Scene(prim)
        d, m = scene.GetGravityDirectionAttr(), scene.GetGravityMagnitudeAttr()
        tsps = prim.GetAttribute("physxScene:timeStepsPerSecond")
        out.append((str(prim.GetPath()),
                    tuple(d.Get()) if d and d.HasAuthoredValue() else None,
                    round(float(m.Get()), 6) if m and m.HasAuthoredValue() else None,
                    int(tsps.Get()) if tsps and tsps.HasAuthoredValue() else None))
    return out


def edited_layers(stage):
    """The identifiers of the stage's file layers with unsaved edits, the session layer and
    anonymous layers aside. scensim writes into the server's own layers only, never into the
    user's (design 3.2); one listed here is a scensim bug, and the load refuses to go on."""
    session = stage.GetSessionLayer()
    return sorted(layer.identifier for layer in stage.GetUsedLayers()
                  if layer.dirty and not layer.anonymous and layer != session)
