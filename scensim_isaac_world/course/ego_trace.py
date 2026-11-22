"""The ego's truth per frame (scensim's truth stream), as CSV in the attachment's directory
(`ego.csv`): sim time, x, y, z, roll, pitch, yaw (map frame, rad), world velocity and angular
velocity. For 086 C5b's banked-arc check (`arc_check`). A testbench attachment:

    attachments:
      - name: ego_trace
        run: [python3, -m, scensim_isaac_world.course.ego_trace]
        required: true
"""
import json
import os
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from scensim_interfaces.msg import TruthFrame

out_dir = os.environ.get("SCENSIM_ATTACHMENT_DIR", ".")
topic = os.environ.get("SCENSIM_TRUTH_TOPIC", "/scensim/truth")
roles = json.loads(os.environ.get("SCENSIM_ROLES", "{}"))
ego = next((n for n, r in roles.items() if r == "ego"), "ego")
f = open(os.path.join(out_dir, "ego.csv"), "w", buffering=1)
f.write("t,frame,x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,measured\n")


class Trace(Node):
    def __init__(self):
        super().__init__("c5b_ego_trace")
        self.create_subscription(TruthFrame, topic, self.on, QoSProfile(
            depth=200, reliability=ReliabilityPolicy.RELIABLE))

    def on(self, m):
        if ego not in m.names:
            return
        k = list(m.names).index(ego)
        if not m.present[k]:
            return
        p = m.poses[6 * k:6 * k + 6]
        v = m.velocities[3 * k:3 * k + 3]
        w = m.angular_velocities[3 * k:3 * k + 3]
        f.write(",".join(f"{a:.9g}" for a in [m.sim_time_ns * 1e-9, m.sim_frame, *p, *v, *w])
                + f",{int(m.velocity_measured[k])}\n")


def stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)
rclpy.init()
node = Trace()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    f.close()
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
sys.exit(0)
