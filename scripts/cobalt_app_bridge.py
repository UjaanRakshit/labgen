"""Be the server the COBALT phone app connects to, and drive the labgen sim from it.

    python scripts/cobalt_app_bridge.py [--port 8080]

Install the COBALT app (github.com/pairlab/cobalt-mobile-app) UNCHANGED. In its
Settings, set the IP to this machine's LAN address (printed below), the port to
8080, the task to `test`. Bimanual: turn it on in both phones' settings and
give the second phone the Session ID the first one shows. Then:

  phone            this bridge                         the sim (teleop_task.py --input cobalt)
  COBALT app  <->  WebSocket, COBALT's protocol  ->    pose events on TCP 9871 (as teleop_touch)
                   (labgen.cobalt_app)           <-    {"event": "success"} when the task succeeds
                   MJPEG view http://<ip>:8000   <-    JPEG frames on TCP 9872 (--stream)

What it reproduces from COBALT, exactly: the handshake (init, status ready,
response after every message), its DeviceState (reset and completion
handshakes, engaged = enable and not mid-reset, the first-sample guard), and its
YAM phone-to-robot mapping (diag(-1,-1,1), positions / 1.5). Gripper: the app's
Grasp/Release toggle. Reset: discards the episode and resets the sim (COBALT
discards on a user reset). Completion: when the sim's success check fires, the
app is told the task is complete, exactly as COBALT's server tells it.

Devices map to arms in connection order, as COBALT indexes them: the first phone
drives the right arm (robot0), the second the left (robot1).

Stdlib only -- runs on Windows (WSL2 setups, where phones cannot reach WSL) or
on the Linux sim machine itself.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from labgen.cobalt_app import (MAPPINGS, DeviceState, read_frame, rotation_to_wxyz,  # noqa: E402
                               send_frame, server_handshake, to_robot)
from teleop_touch import Fanout, FrameStore, local_ips, serve_frame_producer  # noqa: E402

HANDS = ("right", "left")        # connection order -> arm, as COBALT indexes devices
SKIP_AFTER_ENGAGE = 2             # samples ignored after each Enable (see on_message)
MAX_SAMPLE_M = 0.03               # one 50 ms sample moving more than this is a tracking glitch


class Session:
    """One COBALT session: up to two devices, the sim clients, the episode bits."""

    def __init__(self, fan: Fanout, scale: float = 1.0, mapping: str = "cobalt"):
        self.fan = fan
        self.mapping = mapping
        self.phone_to_robot = MAPPINGS[mapping]
        self.scale = scale      # on top of COBALT's 1/1.5; 1.0 = COBALT exactly
        self.lock = threading.Lock()
        self.session_id = uuid.uuid4().hex[:8]
        self.devices: dict[str, dict] = {}        # device_id -> {state, hand, pos, seq}
        self.cmd = ""
        self.cmd_seq = 0
        self.sims = 0

    def join(self) -> tuple[str, str]:
        with self.lock:
            if len(self.devices) >= len(HANDS):
                raise ConnectionError("session already has two devices")
            device_id = uuid.uuid4().hex[:8]
            hand = HANDS[len(self.devices)]
            self.devices[device_id] = {"state": DeviceState(device_id), "hand": hand,
                                       "pos": np.zeros(3), "rot": np.eye(3),
                                       "phone_travel": 0.0, "moving": False, "skip": 0}
            return device_id, hand

    def leave(self, device_id: str) -> None:
        with self.lock:
            self.devices.pop(device_id, None)

    def on_success(self) -> None:
        with self.lock:
            for d in self.devices.values():
                d["state"].signal_completion()
        print("[cobalt] sim reports SUCCESS -> 'complete' sent to the app", flush=True)

    def on_message(self, device_id: str, msg: dict) -> str:
        """Apply one 'device data' message; publish a pose event; return the response."""
        with self.lock:
            d = self.devices[device_id]
            s = d["state"].parse(msg)
            if s.user_reset:
                self.cmd, self.cmd_seq = "discard", self.cmd_seq + 1
                d["pos"] = np.zeros(3)
                print(f"[cobalt] {d['hand']} phone pressed Reset -> discard and reset the sim", flush=True)
            moving = s.engaged and s.valid
            if moving and not d["moving"]:
                # The app resets its AR reference on Enable but not its last
                # position, so its first deltas after each re-engage are a jump of
                # up to 4 cm (more trips its own too-fast check). Skip them.
                d["skip"] = SKIP_AFTER_ENGAGE
            spike = float(np.linalg.norm(s.dpos)) > MAX_SAMPLE_M
            if moving and (d["skip"] > 0 or spike):
                d["skip"] = max(0, d["skip"] - 1)
                d["dropped"] = d.get("dropped", 0) + 1
            elif moving:
                dp, R = to_robot(s.dpos, s.rotation, phone_to_robot=self.phone_to_robot)
                d["pos"] = d["pos"] + dp * self.scale
                d["rot"] = R
                d["phone_travel"] += float(np.linalg.norm(s.dpos))
            d["moving"] = moving
            w, x, y, z = rotation_to_wxyz(d["rot"])
            line = json.dumps({
                "position": {"x": float(d["pos"][0]), "y": float(d["pos"][1]), "z": float(d["pos"][2])},
                "orientation": {"w": w, "x": x, "y": y, "z": z},
                "move": moving, "gripper": 1.0 if s.grasp else 0.0, "scale": 1.0,
                "seq": s.id, "t": time.time(), "device": "cobalt-app", "hand": d["hand"],
                "cmd": self.cmd, "cmd_seq": self.cmd_seq,
            }) + "\n"
            response = d["state"].response()
        self.fan.broadcast(line.encode("utf-8"))
        return response


def report(session: Session) -> None:
    """Every 2 s: how far each phone has travelled, and where that puts the arm."""
    while True:
        time.sleep(2.0)
        with session.lock:
            for d in session.devices.values():
                p = d["pos"] * 100
                print(f"[cobalt] {d['hand']:5s} {'ENGAGED' if d['moving'] else 'idle   '} "
                      f"phone travelled {d['phone_travel']*100:6.1f} cm total; arm offset "
                      f"x {p[0]:+6.1f} y {p[1]:+6.1f} z {p[2]:+6.1f} cm (scale {session.scale:g} x 1/1.5)",
                      flush=True)


def serve_app(conn: socket.socket, addr, session: Session, public_ip: str) -> None:
    device_id = None
    try:
        path = server_handshake(conn)
        q = parse_qs(urlparse(path).query)
        cfg = json.loads(q.get("config", ["{}"])[0] or "{}")
        device_id, hand = session.join()
        send_frame(conn, json.dumps({"type": "init", "data": {
            "session_id": session.session_id, "device_id": device_id, "server_ip": public_ip}}).encode())
        ready = session.sims > 0
        send_frame(conn, json.dumps({"type": "status", "data": {"ready": ready}}).encode())
        print(f"[cobalt] app {addr[0]} joined as {hand} arm (config {cfg}); ready={ready}", flush=True)
        if not ready:
            print("[cobalt] no sim connected -- start teleop_task.py --input cobalt first", flush=True)
            return
        while True:
            op, payload = read_frame(conn)
            if op == 0x8:
                return
            if op == 0x9:
                send_frame(conn, payload, opcode=0xA)
                continue
            if op not in (0x1, 0x2):
                continue
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("type") != "device data":
                continue
            send_frame(conn, json.dumps({"type": "response",
                                         "data": session.on_message(device_id, msg["data"])}).encode())
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"[cobalt] app {addr[0]} disconnected: {exc}", flush=True)
    finally:
        if device_id:
            session.leave(device_id)
        try:
            conn.close()
        except OSError:
            pass


def serve_sims(session: Session, port: int) -> None:
    """The sim connects here (as to teleop_touch), and reports success back on the same socket."""
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    print(f"[cobalt] sim connects here: 0.0.0.0:{port}", flush=True)
    while True:
        sock, addr = srv.accept()
        session.fan.add(sock)
        session.sims += 1
        print(f"[cobalt] sim connected from {addr[0]}", flush=True)
        threading.Thread(target=read_sim_events, args=(sock, session), daemon=True).start()


def read_sim_events(sock: socket.socket, session: Session) -> None:
    buf = b""
    try:
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    if json.loads(line).get("event") == "success":
                        session.on_success()
                except ValueError:
                    pass
    except OSError:
        pass
    session.sims = max(0, session.sims - 1)
    print("[cobalt] sim disconnected", flush=True)


def viewer(store: FrameStore):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        got = store.wait_for_next(seq)
                        if not got:
                            continue
                        jpeg, seq = got
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except OSError:
                    return
            body = (b"<!doctype html><title>labgen sim</title><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100vw;height:100vh;object-fit:contain'></body>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="the app's WebSocket port (app default 8080)")
    ap.add_argument("--sim-port", type=int, default=9871)
    ap.add_argument("--frame-port", type=int, default=9872)
    ap.add_argument("--view-port", type=int, default=8000)
    ap.add_argument("--mapping", default="cobalt", choices=sorted(MAPPINGS),
                    help="cobalt: COBALT's YAM matrix -- correct for an operator behind the arms, "
                         "phone flat, screen up, top toward the robots")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="extra position gain on top of COBALT's 1/1.5 (1.0 = COBALT exactly)")
    args = ap.parse_args()

    session = Session(Fanout(), scale=args.scale, mapping=args.mapping)
    print(f"[cobalt] phone mapping: {args.mapping}, position gain {args.scale:g} x 1/1.5", flush=True)
    threading.Thread(target=report, args=(session,), daemon=True).start()
    store = FrameStore()
    threading.Thread(target=serve_sims, args=(session, args.sim_port), daemon=True).start()
    threading.Thread(target=serve_frame_producer, args=(store, args.frame_port), daemon=True).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.view_port), viewer(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    ips = local_ips()
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port)); srv.listen(4)
    print(f"[cobalt] COBALT app: Settings -> IP one of {ips}, port {args.port}, task 'test'", flush=True)
    print(f"[cobalt] sim view: " + ", ".join(f"http://{ip}:{args.view_port}" for ip in ips), flush=True)
    public_ip = ips[0] if ips else "127.0.0.1"
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=serve_app, args=(conn, addr, session, public_ip), daemon=True).start()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
