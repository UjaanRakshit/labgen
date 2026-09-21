"""A synthetic pose device, so the whole chain can be tested with no phone.

Speaks the exact wire format scripts/teleop_bridge.py emits and listens on the
same port, so `teleop_sim.py` cannot tell the difference. Stdlib only; runs on
Windows under any Python.

    python scripts/teleop_fake_device.py --pattern circle

What this actually buys: the parts of phone teleop most likely to be wrong are
not the phone. They are the frame conventions, the engage/clutch behaviour, the
WSL-to-Windows socket direction, and whether the IK keeps up at 60 Hz. All four
are exercised here, on a motion whose right answer is known in advance -- so
when the phone does arrive, a failure is a phone problem and not a
five-way guess.

Patterns:
    circle   a 6 cm circle in the horizontal plane, engaged throughout
    square   axis-aligned steps, for checking each axis maps where it should
    clutch   moves, releases, jumps far away, re-engages -- the hand must NOT
             jump, which is the single most important behaviour to get right
    dropout  injects a 3 m teleport, which must be rejected rather than obeyed
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time

PORT = 9871


def quat_identity():
    return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def frames(pattern: str, hz: float):
    """Yield (position, engaged, grip) forever."""
    t = 0.0
    dt = 1.0 / hz
    while True:
        if pattern == "circle":
            a = 2.0 * math.pi * (t / 6.0)
            yield (0.03 * math.cos(a), 0.03 * math.sin(a), 0.0), True, 0.0
        elif pattern == "square":
            leg = int(t / 2.0) % 4
            u = (t % 2.0) / 2.0 * 0.04
            pos = [(u, 0.0, 0.0), (0.04, u, 0.0),
                   (0.04 - u, 0.04, 0.0), (0.0, 0.04 - u, 0.0)][leg]
            yield pos, True, 0.0
        elif pattern == "clutch":
            phase = t % 12.0
            if phase < 4.0:
                yield (0.04 * phase / 4.0, 0.0, 0.0), True, 0.0
            elif phase < 6.0:
                yield (0.04, 0.0, 0.0), False, 0.0          # released
            elif phase < 7.0:
                yield (-0.50, 0.30, 0.0), False, 0.0        # recentred, still off
            else:
                u = (phase - 7.0) / 5.0 * 0.04
                yield (-0.50 + u, 0.30, 0.0), True, 0.0     # re-engaged
        elif pattern == "dropout":
            if 3.0 < t < 3.05:
                yield (3.0, 0.0, 0.0), True, 0.0            # teleport
            else:
                yield (0.02 * math.sin(t), 0.0, 0.0), True, 0.0
        else:
            raise SystemExit(f"unknown pattern {pattern!r}")
        t += dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="circle",
                    choices=("circle", "square", "clutch", "dropout"))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 runs until interrupted")
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(4)
    print(f"[fake] pattern={args.pattern} listening on 0.0.0.0:{args.port}",
          flush=True)

    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[fake] sim connected from {addr}", flush=True)

    gen = frames(args.pattern, args.hz)
    seq = 0
    started = time.time()
    try:
        while True:
            (x, y, z), engaged, grip = next(gen)
            seq += 1
            line = json.dumps({
                "position": {"x": x, "y": y, "z": z},
                "orientation": quat_identity(),
                "move": engaged, "gripper": grip, "scale": 1.0,
                "seq": seq, "t": time.time(),
                "device": "fake", "hand": "right",
            }) + "\n"
            conn.sendall(line.encode("utf-8"))
            if args.seconds and time.time() - started > args.seconds:
                break
            time.sleep(1.0 / args.hz)
    except (BrokenPipeError, ConnectionResetError):
        print("[fake] sim disconnected", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        srv.close()
    print(f"[fake] sent {seq} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
