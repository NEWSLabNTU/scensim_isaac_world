"""Issue 139: a burst of tile requests no longer deadlocks the tile process. The worker is
replaced by a fake that answers each key with 1.5 MB (past a pipe's 64 KB) 5 ms apart; 3000
keys of about 30 bytes each (past the stdin pipe's 64 KB) go in at once. Before the fix, the
submitting thread blocked writing a key while holding the lock the result reader needed. No
Isaac."""
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scensim_isaac_world import road_detail as rd  # noqa: E402

FAKE = '''
import os, pickle, sys, time
out = os.fdopen(os.dup(1), "wb")
inp = sys.stdin.buffer
pickle.load(inp)
blob = b"x" * 1_500_000
while True:
    try:
        key = pickle.load(inp)
    except EOFError:
        break
    time.sleep(0.005)
    pickle.dump((key, blob, 1.0, None, None), out, protocol=pickle.HIGHEST_PROTOCOL)
    out.flush()
'''
BURST = 3000
TIMEOUT = 20.0


def keys():
    return [(i, 10 ** 12 + i) for i in range(BURST)]  # about 30 bytes pickled


class Burst(unittest.TestCase):
    def setUp(self):
        fd, self.fake = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(FAKE)
        real = subprocess.Popen

        def popen(args, **kw):
            return real([sys.executable, self.fake], **kw)

        self.patch = mock.patch("subprocess.Popen", side_effect=popen)
        self.patch.start()
        self.pools = []

    def tearDown(self):
        self.patch.stop()
        for p in self.pools:  # a hung burst ends when its worker does
            p.proc.kill()
            p.proc.wait()
        os.unlink(self.fake)

    def pool(self):
        p = rd.TileProcess([], 0.05, 16.0)
        self.pools.append(p)
        return p

    def run_burst(self, fn):
        out, done = {}, threading.Event()

        def body():
            out["futures"] = fn()
            for f in out["futures"]:
                f.exception()  # a result or a failure, either way done
            done.set()

        threading.Thread(target=body, daemon=True).start()
        self.assertTrue(done.wait(TIMEOUT), "the burst hung (issue 139)")
        return out["futures"]

    def check(self, futures):
        failed = [f for f in futures if f.exception() is not None]
        for f in failed:
            self.assertIn("dropped", str(f.exception()))
        newest = futures[-rd.MAX_QUEUED:]
        self.assertTrue(all(f.exception() is None for f in newest))
        self.assertEqual(newest[-1].result()[0], b"x" * 1_500_000)

    def test_a_burst_through_submit(self):
        p = self.pool()
        futures = self.run_burst(lambda: [p.submit(None, i, j) for i, j in keys()])
        self.check(futures)

    def test_a_burst_through_prefetch(self):
        """`Detail.prefetch` submits under its own lock, which the done-callback takes on the
        reader thread."""
        d = rd.Detail([], workers=0)
        d.pool = self.pool()
        ks = keys()
        self.run_burst(lambda: (d.prefetch(ks), [])[1])
        wait = threading.Event()
        for _ in range(int(TIMEOUT / 0.05)):
            with d.lock:
                if not d.pending:
                    break
            wait.wait(0.05)
        self.assertEqual(d.pending, {})
        self.assertTrue(all(d.ready(k) for k in ks[-min(rd.MAX_QUEUED, d.max_tiles):]))

    def test_a_key_asked_twice_shares_its_future(self):
        p = self.pool()
        a, b = p.submit(None, 1, 2), p.submit(None, 1, 2)
        self.assertIs(a, b)
        self.assertIsNone(a.exception(timeout=TIMEOUT))

    def test_after_shutdown_a_submission_fails_at_once(self):
        p = self.pool()
        p.shutdown(wait=True)
        self.assertIn("closed", str(p.submit(None, 3, 4).exception(timeout=1.0)))


if __name__ == "__main__":
    unittest.main()
