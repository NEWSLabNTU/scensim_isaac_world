"""The minimal Lanelet2 loop of an authored field (scensim roadmap 091 W17)."""
import math
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scensim_isaac_world import loop  # noqa: E402

try:  # the simple pilot's forward projector: what Autoware and SSv2 compute (scensim_agent)
    from scensim_agent.adapters import simple_route
except Exception:  # noqa: BLE001 - absent outside scensim's workspace
    simple_route = None
try:
    import lanelet2
    from lanelet2.projection import LocalCartesianProjector
except Exception:  # noqa: BLE001 - absent, or ROS's build for another Python (Isaac's venv)
    lanelet2 = None


def circle(r=5.0, n=40, cx=2.0, cy=-3.0, ccw=True):
    s = 1 if ccw else -1
    return [(cx + r * math.cos(s * 2 * math.pi * k / n), cy + r * math.sin(s * 2 * math.pi * k / n))
            for k in range(n)]


def osm_of(path):
    root = ET.parse(path).getroot()
    nodes = {}
    for n in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")),
                              float(tags["local_x"]), float(tags["local_y"]))
    ways = {w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.findall("way")}
    rels = []
    for r in root.findall("relation"):
        roles = {m.get("role"): m.get("ref") for m in r.findall("member")}
        rels.append((ways[roles["left"]], ways[roles["right"]]))
    return nodes, rels


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, pts, **kw):
        g = loop.build(pts, **{k: v for k, v in kw.items() if k in ("width", "direction", "chunk")})
        osm = loop.write(self.d, g, kw.get("width", 1.0), kw.get("direction", "forward"),
                         origin=kw.get("origin", (0.0, 0.0)))
        return g, osm

    def test_the_loop_closes(self):
        g, osm = self.make(circle())
        nodes, rels = osm_of(osm)
        self.assertGreaterEqual(len(rels), 5)
        for (l0, r0), (l1, r1) in zip(rels, rels[1:] + rels[:1]):
            self.assertEqual(l0[-1], l1[0])  # each lanelet ends where the next begins
            self.assertEqual(r0[-1], r1[0])
        self.assertAlmostEqual(g["length"], 2 * math.pi * 5.0, delta=0.2)

    def test_width_and_lanelet_length(self):
        g, osm = self.make(circle(r=6.0), width=1.4, chunk=4.0)
        nodes, rels = osm_of(osm)
        for left, right in rels:
            for a, b in zip(left, right):
                self.assertAlmostEqual(math.dist(nodes[a][2:], nodes[b][2:]), 1.4, delta=0.01)
            run = sum(math.dist(nodes[p][2:], nodes[q][2:]) for p, q in zip(left, left[1:]))
            self.assertTrue(2.0 < run < 9.0, run)

    def test_direction_puts_the_left_bound_on_the_left(self):
        for direction, outside in (("forward", "right"), ("reverse", "left")):
            g, osm = self.make(circle(ccw=True), direction=direction)
            nodes, rels = osm_of(osm)
            left, right = rels[0]
            rl = math.dist(nodes[left[0]][2:], (2.0, -3.0))
            rr = math.dist(nodes[right[0]][2:], (2.0, -3.0))
            # counter-clockwise forward: the centre is on the left, the right bound outside
            self.assertEqual("right" if rr > rl else "left", outside, direction)

    def test_the_header_says_the_direction_and_the_synthetic_origin(self):
        self.make(circle(), direction="reverse")
        text = (self.d / "lanelet2_map.osm").read_text()
        self.assertIn("direction reverse", text.split("<osm", 1)[0])
        self.assertIn("synthetic", text.split("<osm", 1)[0])
        info = (self.d / "map_projector_info.yaml").read_text()
        self.assertIn("projector_type: TransverseMercator", info)
        self.assertIn("scale_factor: 1.0", info)
        self.assertIn("synthetic", info)

    def test_lat_lon_round_trip(self):
        _, osm = self.make(circle(r=20.0, cx=30.0, cy=-40.0))
        nodes, _ = osm_of(osm)
        proj = loop.Projector(0.0, 0.0, 1.0)
        for lat, lon, x, y in nodes.values():
            px, py = proj.project(lat, lon)
            self.assertLess(math.hypot(px - x, py - y), 1e-3)

    @unittest.skipIf(simple_route is None, "scensim_agent not on the path")
    def test_the_simple_pilot_computes_the_same_positions(self):
        _, osm = self.make(circle(r=20.0, cx=30.0, cy=-40.0), origin=(0.0, 0.0))
        info = simple_route.read_projector_info(str(self.d))
        pos = simple_route.map_frame_positions(str(osm), info)
        nodes, _ = osm_of(osm)
        for nid, (lat, lon, x, y) in nodes.items():
            px, py = pos[int(nid)] if int(nid) in pos else pos[nid]
            self.assertLess(math.hypot(px - x, py - y), 1e-3, nid)

    @unittest.skipIf(lanelet2 is None, "no lanelet2 Python")
    def test_lanelet2_routes_round_the_loop(self):
        from lanelet2.core import GPSPoint
        from lanelet2.io import Origin
        from lanelet2 import routing, traffic_rules
        _, osm = self.make(circle(r=8.0))
        m, errors = lanelet2.io.loadRobust(str(osm), LocalCartesianProjector(Origin(0.0, 0.0)))
        self.assertEqual(errors, [])
        rules = traffic_rules.create(traffic_rules.Locations.Germany,
                                     traffic_rules.Participants.Vehicle)
        graph = routing.RoutingGraph(m, rules)
        lls = sorted(m.laneletLayer, key=lambda ll: ll.id)
        for ll in lls:
            self.assertEqual(len(graph.following(ll)), 1, ll.id)
        self.assertIsNotNone(graph.getRoute(lls[0], lls[-1]))
        del GPSPoint

    def test_a_point_cloud_for_ssv2s_loader(self):
        self.make(circle(r=6.0), width=2.0)
        text = (self.d / "pointcloud_map.pcd").read_text().splitlines()
        self.assertIn("not a localisation map", text[0])
        n = int(next(x for x in text if x.startswith("POINTS")).split()[1])
        rows = text[text.index("DATA ascii") + 1:]
        self.assertEqual(len(rows), n)
        for row in rows:
            x, y, z = map(float, row.split())
            self.assertAlmostEqual(z, 0.0)
            # within the half width of the circle (its 40-gon chords sit up to 2 cm inside it)
            self.assertLessEqual(abs(math.dist((x, y), (2.0, -3.0)) - 6.0), 1.0 + 0.05)

    def test_closing_point_and_duplicates_are_dropped(self):
        pts = circle(n=20)
        g1 = loop.build(pts + [pts[0]])
        g2 = loop.build([pts[0]] + pts)
        self.assertAlmostEqual(g1["length"], g2["length"], places=6)

    def test_refusals_name_the_cause(self):
        with self.assertRaisesRegex(loop.LoopError, "distinct point"):
            loop.build([(0, 0), (1, 0), (0, 0)])
        with self.assertRaisesRegex(loop.LoopError, "crosses itself"):
            loop.build([(0, 0), (10, 10), (10, 0), (0, 10)])
        with self.assertRaisesRegex(loop.LoopError, "tighter than the lane's half width"):
            loop.build(circle(r=0.8, n=40), width=2.0, chunk=1.0)
        with self.assertRaisesRegex(loop.LoopError, "two lanelets"):
            loop.build(circle(r=1.0), chunk=5.0)
        with self.assertRaisesRegex(loop.LoopError, "direction"):
            loop.build(circle(), direction="backwards")

    def test_csv_and_cli(self):
        csv = self.d / "track.csv"
        csv.write_text("x,y\n# a test\n" + "\n".join(f"{x},{y}" for x, y in circle()) + "\n")
        self.assertEqual(len(loop.read_csv(csv)), 40)
        self.assertEqual(loop.main([str(self.d), "--centreline", str(csv), "--width", "1.2"]), 0)
        self.assertTrue((self.d / "lanelet2_map.osm").exists())
        bad = self.d / "bad.csv"
        bad.write_text("0,0\n1,x\n")
        self.assertEqual(loop.main([str(self.d), "--centreline", str(bad)]), 2)
        self.assertIsNotNone(re.search(r"width 1\.2 m", (self.d / "lanelet2_map.osm").read_text()))


if __name__ == "__main__":
    unittest.main()
