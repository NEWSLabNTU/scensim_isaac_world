"""The world's contacts from PhysX's contact report, without Isaac (scensim issue 192): which
reporter touched which prim, and how hard. `IsaacWorld._collect_contacts` calls `rows` on the
frame's report. Pure Python, so it is tested with stub headers.

A contact is kept when one side is a reporter (the ego, a tracked body) and the other is not
ground: the world's `ground_paths`, the flat plane, the road detail's collision tiles, and,
once a soil zone is built, the soil zones' own ground (`SOIL_ROOT`: the cut copies of the
world's road, the flat box, the wheel plates; issue 192). Each reporting side has its
contact: two reporters touching are two (044).
"""
import math

SOIL_ROOT = "/World/soil"  # soil_tiles.ROOT; here so this module needs no pxr


def is_under(path, roots):
    """`path` is one of `roots` or below one."""
    return any(path == p or path.startswith(p + "/") for p in roots)


def rows(headers, data, path_of, by_path, reporters, is_ground, on_ground=None):
    """`[(handle, other, path, impulse)]`, one per reporting side of each contact pair that is
    not lost and not with ground; `impulse` the norm of each contact point's impulse, summed.
    `path_of(int)`: a header's actor as a prim path. `on_ground(header, mine, theirs)`: called
    for a reporter's contact with ground, which is dropped (the 0.0 pair's log, issue 192)."""
    out = []
    for header in headers:
        if str(header.type).endswith("CONTACT_LOST"):
            continue
        a0, a1 = path_of(header.actor0), path_of(header.actor1)
        for mine, theirs in ((a0, a1), (a1, a0)):
            handle = by_path.get(mine)
            if handle not in reporters:
                continue
            if is_ground(theirs):
                if on_ground is not None:
                    on_ground(header, mine, theirs)
                continue
            impulse = 0.0
            for i in range(header.contact_data_offset,
                           header.contact_data_offset + header.num_contact_data):
                v = data[i].impulse
                impulse += math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            out.append((handle, by_path.get(theirs, 0), theirs, impulse))
    return out


def separation(header, data):
    """The smallest separation among a pair's contact points (m; negative: penetrating), or
    None when it has none."""
    seps = [data[i].separation for i in range(header.contact_data_offset,
                                              header.contact_data_offset + header.num_contact_data)]
    return min(seps) if seps else None
