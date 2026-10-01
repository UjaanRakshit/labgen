"""The COBALT phone app's protocol, so labgen can be the server it talks to.

The app (github.com/pairlab/cobalt-mobile-app) is used UNCHANGED. It opens a
WebSocket to ``ws://<host>:<port>/ws?config=...`` (or ``/<task>/ws``) and:

  server -> app   {"type": "init",   "data": {session_id, device_id, server_ip}}
                  {"type": "status", "data": {"ready": true}}   -- it sends nothing before this
                  {"type": "response", "data": "<json {reset, complete, timeout}>"}  -- after every message
  app -> server   {"type": "device data", "data": {id, enable, dpos[3], grasp, reset,
                   completion, timeout, valid, keep_demo_decision,
                   demo_decision_indicator, timestamp, rotation[9]}}   every 50 ms

`DeviceState` is COBALT's own server-side state machine
(backend/teleop-server/server_utils/device_state.py), reproduced: the reset
and completion handshakes, and when the operator counts as engaged.
`to_robot` is COBALT's YAM mapping (robots/yam_robot.py): position deltas and
the rotation are taken into the robot base frame by diag(-1, -1, 1), and
positions are scaled by 1/1.5.

Stdlib only, including the minimal RFC 6455 WebSocket framing below: this runs
in whatever environment is to hand without installing anything.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import socket
import struct
from dataclasses import dataclass

import numpy as np

__all__ = ["DeviceState", "AppSample", "to_robot", "PHONE_TO_ROBOT", "POSITION_SCALE",
           "accept_key", "read_frame", "send_frame", "server_handshake", "rotation_to_wxyz"]

# COBALT robots/yam_robot.py: _phone_to_robot_rotation_global and the 1/1.5 in
# teleop_to_position_control. This is COBALT's YAM rig, operator stance included.
PHONE_TO_ROBOT = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
POSITION_SCALE = 1.0 / 1.5


@dataclass
class AppSample:
    """One 'device data' message, parsed the way COBALT's server parses it."""

    id: int
    engaged: bool            # enable AND not mid reset-handshake
    valid: bool
    dpos: np.ndarray         # phone frame, metres, change since the previous sample
    rotation: np.ndarray     # 3x3, phone relative to its pose at engage (app's EEF convention)
    grasp: bool
    user_reset: bool         # the operator pressed Reset (rising edge)


@dataclass
class DeviceState:
    """COBALT's DeviceState, reproduced: handshakes and engagement."""

    device_id: str
    reset_indicator: int = 0
    task_completion_flag: bool = False
    task_timeout_flag: bool = False
    _last_reset_state: bool = False
    _doing_reset_handshaking: bool = False
    _task_completion_indicator: int = 0
    _task_timeout_indicator: int = 0

    def response(self) -> str:
        """The 'response' payload. COBALT sends it as a JSON *string*; the app decodes it."""
        return json.dumps({"reset": int(self.reset_indicator),
                           "complete": int(self.task_completion_flag),
                           "timeout": int(self.task_timeout_flag)})

    def signal_completion(self) -> None:
        """The task succeeded: raise 'complete' until the app acknowledges."""
        self.task_completion_flag = True

    def parse(self, msg: dict) -> AppSample:
        reset_state = bool(msg.get("reset", 0))
        user_reset = False
        if not self._last_reset_state and reset_state:
            user_reset = True
            self.reset_indicator = (self.reset_indicator + 1) % 2
            self._doing_reset_handshaking = True
        elif self._last_reset_state and not reset_state:
            self._doing_reset_handshaking = False
        self._last_reset_state = reset_state

        ci = int(msg.get("completion", 0))
        if ci != self._task_completion_indicator:
            self.task_completion_flag = False          # the app has handled it
        self._task_completion_indicator = ci
        ti = int(msg.get("timeout", 0))
        if ti != self._task_timeout_indicator:
            self.task_timeout_flag = False
        self._task_timeout_indicator = ti

        dpos = np.asarray(msg["dpos"], float).reshape(3)
        rot = np.asarray(msg["rotation"], float).reshape(3, 3)
        engaged = bool(msg["enable"]) and not self._doing_reset_handshaking
        valid = bool(msg["valid"])
        # COBALT: the phone's first samples after engaging can carry an untouched
        # dpos (all zero) and identity rotation; they are not real readings.
        if engaged and valid and float(np.sum(dpos)) == 0.0 and float(np.sum(rot)) == 3.0:
            valid = False
        return AppSample(id=int(msg.get("id", 0)), engaged=engaged, valid=valid, dpos=dpos,
                         rotation=rot, grasp=bool(msg["grasp"]), user_reset=user_reset)


def to_robot(dpos, rotation, phone_to_robot=PHONE_TO_ROBOT, scale=POSITION_SCALE):
    """COBALT's YAM mapping: (robot-frame position delta, robot-frame rotation)."""
    return (phone_to_robot @ np.asarray(dpos, float)) * scale, phone_to_robot @ np.asarray(rotation, float)


def rotation_to_wxyz(R) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), robust for any angle."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


# ---- minimal RFC 6455 server side --------------------------------------------------

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + _GUID).encode()).digest()).decode()


def server_handshake(sock: socket.socket) -> str:
    """Read the HTTP upgrade request, answer 101, return the request path+query."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("closed during handshake")
        buf += chunk
        if len(buf) > 65536:
            raise ConnectionError("handshake too large")
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    path = head[0].split(" ")[1]
    headers = {k.strip().lower(): v.strip() for k, v in (h.split(":", 1) for h in head[1:] if ":" in h)}
    if headers.get("upgrade", "").lower() != "websocket" or "sec-websocket-key" not in headers:
        sock.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        raise ConnectionError("not a websocket upgrade")
    sock.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n").encode())
    return path


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("socket closed")
        out += chunk
    return out


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """One complete message: (opcode, payload). Reassembles fragments; unmasks."""
    opcode, data = None, b""
    while True:
        b0, b1 = _recv_exact(sock, 2)
        fin, op = b0 & 0x80, b0 & 0x0F
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _recv_exact(sock, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
        mask = _recv_exact(sock, 4) if b1 & 0x80 else None
        payload = _recv_exact(sock, n)
        if mask:
            payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        if op >= 0x8:                       # control frames are never fragmented
            return op, payload
        if op != 0x0:
            opcode = op
        data += payload
        if fin:
            return opcode, data


def send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x1, mask: bool = False) -> None:
    head = bytes([0x80 | opcode])
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 65536:
        head += bytes([mbit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack(">Q", n)
    if mask:
        key = b"\x11\x22\x33\x44"
        payload = bytes(c ^ key[i % 4] for i, c in enumerate(payload))
        head += key
    sock.sendall(head + payload)
