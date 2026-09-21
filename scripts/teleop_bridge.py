"""Phone -> sim bridge. Runs on WINDOWS, in its own venv.

    .venv-teleop/Scripts/python scripts/teleop_bridge.py

Serves the WebXR page to the phone and re-publishes each pose sample as one
line of JSON to any TCP client that connects. The sim connects to it.

WHY THIS PROCESS EXISTS AT ALL, twice over:

1. CLAUDE.md rule 5. The WebXR frontend needs FastAPI, uvicorn and
   transforms3d. The Isaac Lab environment is a pinned beta that must not be
   installed into casually, so the web half lives in a separate venv and the
   sim half (labgen.devices) needs nothing but the standard library.

2. WSL2 networking. Under the default NAT mode a phone on the LAN cannot reach
   a server running inside WSL -- WSL sits on its own private network
   (172.x.x.x) behind the Windows host. The usual fixes are mirrored networking
   mode or a netsh portproxy, both of which need admin and a `wsl --shutdown`.
   Neither is necessary if the direction is inverted: WSL can always reach the
   Windows host at its default gateway. So the phone talks to THIS process, on
   Windows, directly on the LAN; and the sim inside WSL connects OUT to it.
   Measured working with no configuration at all.

WHAT IS FORWARDED, and what is deliberately not:

The `teleop` package accumulates a single end-effector pose inside its server.
That is not what goes on the wire here. This forwards the RAW device pose,
converted from WebXR's Right-Up-Back frame to robotics Forward-Left-Up using
the package's own TF_RUB2FLU, and leaves the anchoring to
`labgen.devices.RelativeRetargeter`. Two reasons: the retargeting is then pure
arithmetic that can be tested with no phone, and a bimanual rig needs one
anchor per hand, which a single accumulated pose cannot express.

The phone will warn about the certificate -- the one shipped with `teleop` is
self-signed. WebXR requires a secure context, so plain http will silently fail
to give the page any pose at all. Accept the warning.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import numpy as np
from teleop import TF_RUB2FLU, Teleop, get_local_ip

DEFAULT_BRIDGE_PORT = 9871
DEFAULT_WEB_PORT = 4443

R_RUB2FLU = np.asarray(TF_RUB2FLU, float)[:3, :3]


class Fanout:
    """Holds the connected sim clients and drops the ones that go away.

    Non-blocking sends. A stalled consumer must never back-pressure the phone
    stream: the freshest pose is the only one that matters, and an operator
    whose arm lags by a growing buffer has no idea why.
    """

    def __init__(self) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self.sent = 0
        self.dropped = 0
        self.stalled = 0
        self.stalled = 0

    def add(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Blocking with a short timeout, NOT non-blocking. sendall on a
        # non-blocking socket raises BlockingIOError the moment the kernel
        # buffer is momentarily full, and BlockingIOError IS an OSError --
        # so the 'drop dead clients' branch deleted the live sim on the
        # first hiccup. It showed up as the client count going 1 -> 0 and
        # the arm quietly ignoring the phone.
        sock.settimeout(0.25)
        with self._lock:
            self._clients.append(sock)

    def broadcast(self, line: bytes) -> None:
        with self._lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(line)
                    self.sent += 1
                except (BlockingIOError, OSError):
                    dead.append(c)
            for c in dead:
                self.dropped += 1
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._clients)


def rub_to_flu(position, quat_wxyz):
    """WebXR Right-Up-Back -> robotics Forward-Left-Up.

    Uses the `teleop` package's own constant rather than a convention invented
    here. A frame convention nobody can check without the hardware in hand is
    exactly the kind of thing this project has got wrong before by assuming it.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    p_flu = R_RUB2FLU @ np.asarray(position, float)
    R_flu = R_RUB2FLU @ R @ R_RUB2FLU.T

    t = np.trace(R_flu)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = (0.25 / s, (R_flu[2, 1] - R_flu[1, 2]) * s,
             (R_flu[0, 2] - R_flu[2, 0]) * s, (R_flu[1, 0] - R_flu[0, 1]) * s)
    else:
        i = int(np.argmax(np.diag(R_flu)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R_flu[i, i] - R_flu[j, j] - R_flu[k, k])
        qi = 0.25 * s
        qw = (R_flu[k, j] - R_flu[j, k]) / s
        qj = (R_flu[j, i] + R_flu[i, j]) / s
        qk = (R_flu[k, i] + R_flu[i, k]) / s
        q = [qw, 0.0, 0.0, 0.0]
        q[1 + i], q[1 + j], q[1 + k] = qi, qj, qk
        q = tuple(q)
    n = np.sqrt(sum(c * c for c in q)) or 1.0
    return p_flu, tuple(c / n for c in q)


def serve_clients(fan: Fanout, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(8)
    print(f"[bridge] sim clients connect here: 0.0.0.0:{port}", flush=True)
    while True:
        sock, addr = srv.accept()
        print(f"[bridge] sim connected from {addr}", flush=True)
        fan.add(sock)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    ap.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    ap.add_argument("--hand", default="right",
                    help="which hand this device drives; bimanual runs two bridges")
    args = ap.parse_args()

    fan = Fanout()
    threading.Thread(target=serve_clients, args=(fan, args.bridge_port),
                     daemon=True).start()

    seq = 0
    last_report = time.time()

    def on_pose(_accumulated, message: dict) -> None:
        nonlocal seq, last_report
        pos, ori = message.get("position"), message.get("orientation")
        if pos is None or ori is None:
            return
        p, q = rub_to_flu((pos["x"], pos["y"], pos["z"]),
                          (ori["w"], ori["x"], ori["y"], ori["z"]))
        seq += 1
        fan.broadcast((json.dumps({
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "orientation": {"w": q[0], "x": q[1], "y": q[2], "z": q[3]},
            "move": bool(message.get("move", False)),
            "gripper": float(message.get("gripper", 0.0)),
            "scale": float(message.get("scale", 1.0)),
            "seq": seq, "t": time.time(),
            "device": "phone", "hand": args.hand,
        }) + "\n").encode("utf-8"))

        now = time.time()
        if now - last_report > 2.0:
            print(f"[bridge] {seq} samples, {fan.count} sim client(s), "
                  f"{fan.dropped} dropped", flush=True)
            last_report = now

    teleop = Teleop(port=args.web_port)
    teleop.subscribe(on_pose)

    ip = get_local_ip()
    print(f"[bridge] open this on the phone:  https://{ip}:{args.web_port}")
    print("[bridge] the certificate is self-signed -- accept the warning.")
    print("[bridge] WebXR needs a secure context; plain http yields no pose at all.")
    print(f"[bridge] from WSL the sim reaches this host at its default gateway, "
          f"port {args.bridge_port}")
    teleop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
