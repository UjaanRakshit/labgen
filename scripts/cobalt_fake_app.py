"""A scripted stand-in for the COBALT phone app, speaking its exact protocol.

    python scripts/cobalt_fake_app.py [--host 127.0.0.1] [--port 8080]

Connects like the app (``/ws?config=...``), waits for init and status ready,
then at 20 Hz, as the app does: holds still, enables, moves the phone 5 cm
down (phone -z), rolls it 10 degrees, toggles Grasp, then presses Reset and
completes the reset handshake the way the app's handleReset does. Use it to
check a bridge + sim setup with no phone. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labgen.cobalt_app import read_frame, send_frame  # noqa: E402


def connect(host, port):
    s = socket.create_connection((host, port), timeout=10)
    key = base64.b64encode(b"cobalt-fake-app!").decode()
    cfg = json.dumps({"sim": "Robosuite", "sim_type": "single", "arm": "left", "username": "fake"})
    from urllib.parse import quote
    s.sendall((f"GET /ws?config={quote(cfg)} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    # One byte at a time: the server's first frame can share a segment with the
    # 101, and a bigger recv would swallow it (the next read then hangs).
    buf = b""
    while not buf.endswith(b"\r\n\r\n"):
        c = s.recv(1)
        if not c:
            raise SystemExit("connection closed during the handshake")
        buf += c
    if b" 101 " not in buf.split(b"\r\n")[0]:
        raise SystemExit(f"handshake refused: {buf[:80]!r}")
    return s


AXES = [  # (camera-frame direction, seconds to travel 6 cm); each followed by the return
    ((0.0, 1.0, 0.0), 1.0),     # right
    ((-1.0, 0.0, 0.0), 1.0),    # forward (toward the phone's top = -x)
    ((0.0, 0.0, 1.0), 1.0),     # up
    ((0.0, 1.0, 0.0), 0.3),     # fast right
]


def axes(args) -> int:
    s = connect(args.host, args.port)
    for _ in range(2):
        m = json.loads(read_frame(s)[1])
        if m["type"] == "status" and not m["data"]["ready"]:
            raise SystemExit("server not ready")
    plan = [(0.0, 0.0, 0.0, args.lead_in)]             # hold, enabled, while the sim parks
    for d, secs in AXES:
        plan += [(*d, secs), (0.0, 0.0, 0.0, 1.0), (-d[0], -d[1], -d[2], secs), (0.0, 0.0, 0.0, 1.0)]
    n = 0
    for dx, dy, dz, secs in plan:
        steps = max(1, round(secs / 0.05))
        step = [0.06 * c / steps for c in (dx, dy, dz)] if (dx or dy or dz) else [1e-6, 0, 0]
        if not (dx or dy or dz):
            secs_steps = round(secs / 0.05)
            steps = secs_steps
        for _ in range(steps):
            payload = {"id": n, "enable": 1, "grasp": 0, "reset": 0, "completion": 0, "timeout": 0,
                       "valid": 1, "keep_demo_decision": 1, "demo_decision_indicator": 1,
                       "dpos": step, "timestamp": time.time(), "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1.0]}
            send_frame(s, json.dumps({"type": "device data", "data": payload}).encode(), mask=True)
            read_frame(s)
            n += 1
            time.sleep(0.05)
    print(f"[fake app] axes pattern done, {n} samples", flush=True)
    s.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pattern", default="demo", choices=("demo", "axes"),
                    help="axes: 6 cm right, back, forward, back, up, back (1 s each), then a fast "
                         "6 cm right in 0.3 s and back -- in the camera frame the app reports "
                         "(x = toward the phone's bottom, y = right, z = up, phone flat, top forward)")
    ap.add_argument("--lead-in", type=float, default=8.0,
                    help="axes: seconds to hold still first (the sim parks its grippers after connecting)")
    args = ap.parse_args()
    if args.pattern == "axes":
        return axes(args)
    s = connect(args.host, args.port)
    for _ in range(2):
        op, data = read_frame(s)
        m = json.loads(data)
        print(f"[fake app] <- {m['type']}: {m['data']}", flush=True)
        if m["type"] == "status" and not m["data"]["ready"]:
            raise SystemExit("server not ready (is the sim connected?)")

    state = dict(enable=0, grasp=0, reset=0, completion=0, timeout=0, valid=1,
                 keep_demo_decision=1, demo_decision_indicator=1)
    last_reset_bit, n = 0, 0
    t0 = time.time()
    while True:
        t = time.time() - t0
        dpos = [0.0, 0.0, 0.0]
        roll = 0.0
        state["enable"] = 1 if 1.0 < t < 9.0 else 0
        if 2.0 < t < 5.0:
            dpos = [0.0, 0.0, -0.05 / 60]          # 5 cm down over 3 s at 20 Hz
        if t > 5.0:
            roll = math.radians(10) * min(1.0, (t - 5.0) / 2.0)
        if 7.5 < t < 7.6:
            state["grasp"] = 1
        if t > 10.0 and state["reset"] == 0 and last_reset_bit == 0 and n > 0:
            state["reset"] = 1                     # the operator presses Reset
        c, sn = math.cos(roll), math.sin(roll)
        rot = [c, -sn, 0.0, sn, c, 0.0, 0.0, 0.0, 1.0]
        if state["enable"] and dpos == [0.0, 0.0, 0.0] and roll == 0.0:
            dpos = [1e-6, 0.0, 0.0]                # a real phone never reads exactly zero
        payload = {"id": n, **state, "dpos": dpos, "timestamp": time.time(), "rotation": rot}
        send_frame(s, json.dumps({"type": "device data", "data": payload}).encode(), mask=True)
        op, data = read_frame(s)
        resp = json.loads(json.loads(data)["data"])
        if resp["reset"] != last_reset_bit:        # the app's handleReset
            last_reset_bit = resp["reset"]
            state.update(enable=0, grasp=0, keep_demo_decision=0, reset=0)
            state["demo_decision_indicator"] ^= 1
            print(f"[fake app] reset acknowledged at t={t:.1f}s", flush=True)
            time.sleep(0.5)
            break
        n += 1
        time.sleep(0.05)
    print(f"[fake app] sent {n} samples", flush=True)
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
