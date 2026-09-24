"""Pose-streaming input devices, and how a device's motion becomes a hand target.

One interface, several devices. A phone running WebXR and a Quest controller
both produce the same thing -- a 6-DoF pose plus a couple of buttons -- so
neither the retargeting nor the sim loop should know which one is on the other
end. `PoseSource` is that slot. Phone first because it is cheap to test; the
Quest plugs into the same slot without the sim loop changing.

Why the retargeting lives here rather than in the bridge process:

* It is pure arithmetic, so it can be tested against recorded input with no
  phone, no network and no GPU -- the same reason `control.py` takes its
  kinematics through a protocol.
* Bimanual needs one anchor per hand. A single accumulated pose in the bridge
  cannot express that, and retrofitting it later means rewriting the thing the
  operator has already got used to.

**Relative, not absolute.** The operator presses to engage, moves, and the hand
moves by the same delta; releasing freezes the hand and re-pressing re-anchors
without the hand jumping. Absolute mapping would require the phone and the
robot to share an origin, which nothing establishes, and would send the arm
across the workspace the instant tracking hiccuped.

Credit where due: the WebXR frontend, the websocket and the Right-Up-Back to
Forward-Left-Up frame conversion come from the `teleop` package
(https://github.com/SpesRobotics/teleop), which LeRobot's own phone teleop is
built on. This module deliberately does not reimplement them. What it does
reimplement is the anchoring, because the version there accumulates a single
pose inside the server and this project needs per-hand anchors it can test.
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass, field, replace
from typing import Iterator, Protocol

import numpy as np

__all__ = [
    "PoseEvent",
    "PoseSource",
    "HandTarget",
    "RelativeRetargeter",
    "ReplayPoseSource",
    "TcpPoseSource",
    "Workspace",
    "DEFAULT_PORT",
    "MAX_JUMP_M",
    "MAX_JUMP_DEG",
]

# The bridge listens here, on the Windows side. See scripts/teleop_bridge.py
# for why the direction is that way round rather than the obvious one.
DEFAULT_PORT = 9871

# Reject a single-frame device motion larger than this and re-anchor instead.
# Tracking dropouts on a phone show up as one enormous delta, and forwarding it
# commands the arm across the bench in one tick. Same thresholds the teleop
# package uses for its own jump protection.
MAX_JUMP_M = 0.05
MAX_JUMP_DEG = 35.0


def _quat_to_matrix_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two rotation matrices."""
    r = a @ b.T
    cos = (np.trace(r) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))))


@dataclass(frozen=True)
class PoseEvent:
    """One sample from a pose-streaming device.

    Position and orientation are in the device's own reference frame, already
    converted to Forward-Left-Up by the bridge. They are NOT in the robot base
    frame and are not meaningful in absolute terms -- only their change between
    samples is used.

    `hand` exists so a bimanual device can stream both controllers over one
    connection without the consumer having to demultiplex by convention.
    """

    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    engaged: bool = False
    grip: float = 0.0
    scale: float = 1.0
    seq: int = 0
    t_s: float = 0.0
    device: str = "phone"
    hand: str = "right"
    # Episode control ("save" / "discard"). Sent on EVERY sample with a counter
    # that increments per press, because the sim polls only the newest sample:
    # a one-shot command packet would be overwritten by the next pose and lost.
    # The consumer acts when `command_seq` changes, not when `command` is set.
    command: str = ""
    command_seq: int = 0

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.position_m, float)

    @property
    def rotation(self) -> np.ndarray:
        return _quat_to_matrix_wxyz(self.orientation_wxyz)

    @classmethod
    def from_json(cls, payload: dict) -> "PoseEvent":
        """Parse one line off the wire, rejecting anything malformed.

        Strict on purpose: a dropped field silently defaulting to zero is a
        hand that lunges to the origin, and the operator finds out by watching
        it happen.
        """
        try:
            pos = payload["position"]
            ori = payload["orientation"]
            position = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
            orientation = (float(ori["w"]), float(ori["x"]),
                           float(ori["y"]), float(ori["z"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed pose payload: {exc}") from exc

        norm = math.sqrt(sum(c * c for c in orientation))
        if not (0.9 < norm < 1.1):
            raise ValueError(
                f"orientation is not a unit quaternion (norm {norm:.4f}). "
                f"A non-unit quaternion here becomes a scaled rotation and the "
                f"hand drifts in a way that looks like tracking error.")
        orientation = tuple(c / norm for c in orientation)

        return cls(
            position_m=position,
            orientation_wxyz=orientation,
            engaged=bool(payload.get("move", False)),
            grip=float(payload.get("gripper", 0.0)),
            scale=float(payload.get("scale", 1.0)),
            seq=int(payload.get("seq", 0)),
            t_s=float(payload.get("t", 0.0)),
            device=str(payload.get("device", "phone")),
            hand=str(payload.get("hand", "right")),
            command=str(payload.get("cmd", "")),
            command_seq=int(payload.get("cmd_seq", 0)),
        )


@dataclass(frozen=True)
class HandTarget:
    """Where the grasp frame should be, and how open the jaws should be.

    `approach` is the axis the IK actually constrains. The full `rotation` is
    carried too, but note what that means: `control.solve_pose` solves position
    plus approach axis, so rotation ABOUT the approach axis is not commanded.
    Phone roll is therefore ignored today. That is a missing degree of freedom,
    not a design choice, and it is recorded here so nobody discovers it by
    rolling the phone and watching nothing happen.
    """

    position_m: np.ndarray
    rotation: np.ndarray
    grip: float
    engaged: bool
    hand: str = "right"

    @property
    def approach(self) -> np.ndarray:
        """The device's forward axis, which the gripper points along."""
        return self.rotation @ np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Workspace:
    """An axis-aligned box the commanded target is kept inside.

    Clamping rather than refusing: the operator is holding a phone and watching
    a screen, and an arm that stops responding at the edge reads as a bug. The
    clamp is reported on the target so the caller can show it.
    """

    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, np.asarray(self.lower_m, float),
                       np.asarray(self.upper_m, float))

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= np.asarray(self.lower_m, float))
                    and np.all(p <= np.asarray(self.upper_m, float)))


class PoseSource(Protocol):
    """Anything that yields PoseEvents. The phone and the Quest are both this."""

    def events(self) -> Iterator[PoseEvent]:
        ...

    def close(self) -> None:
        ...


@dataclass
class RelativeRetargeter:
    """Turns device motion into hand motion, anchored at each press.

    The whole contract in three lines:

        not engaged      -> no target; the hand holds station
        first engaged    -> anchor (device pose, hand pose); target = hand pose
        still engaged    -> target = hand_anchor + scale * (device - anchor)

    Releasing clears the anchor, so the operator can re-centre the phone and
    press again without the arm moving -- which is what makes a small phone
    able to drive a large workspace at all.
    """

    position_scale: float = 1.0
    use_orientation: bool = True
    max_jump_m: float = MAX_JUMP_M
    max_jump_deg: float = MAX_JUMP_DEG
    workspace: Workspace | None = None
    # Rotation from the device's axes into the robot's. For a phone held in
    # front of you this is identity; for a touch pad it is whatever makes
    # "drag right" move the arm right ON SCREEN, which depends on where the
    # camera is. Getting this wrong does not look like a bug, it looks like
    # the operator being bad at driving, so it is a parameter rather than a
    # convention baked into the page.
    frame: np.ndarray | None = None

    _device_anchor_p: np.ndarray | None = field(default=None, init=False)
    _device_anchor_r: np.ndarray | None = field(default=None, init=False)
    _hand_anchor_p: np.ndarray | None = field(default=None, init=False)
    _hand_anchor_r: np.ndarray | None = field(default=None, init=False)
    _last_device_p: np.ndarray | None = field(default=None, init=False)
    _last_device_r: np.ndarray | None = field(default=None, init=False)
    _clamped: bool = field(default=False, init=False)
    jumps_rejected: int = field(default=0, init=False)

    @property
    def engaged(self) -> bool:
        return self._device_anchor_p is not None

    @property
    def clamped(self) -> bool:
        """Whether the last target was pushed back into the workspace."""
        return self._clamped

    def release(self) -> None:
        self._device_anchor_p = self._device_anchor_r = None
        self._hand_anchor_p = self._hand_anchor_r = None
        self._last_device_p = self._last_device_r = None

    def update(self, event: PoseEvent, hand_position_m, hand_rotation=None):
        """One device sample plus where the hand is now -> a target, or None.

        `hand_position_m` is only read when anchoring, so the caller passing the
        live measured pose every tick is correct and cheap: between anchors the
        target is computed from the anchor, not from the current hand, which is
        what stops the loop chasing its own tail.
        """
        if not event.engaged:
            self.release()
            return None

        p = event.position
        r = event.rotation if self.use_orientation else np.eye(3)
        if self.frame is not None:
            p = self.frame @ p
            r = self.frame @ r @ self.frame.T

        if self._device_anchor_p is None:
            self._device_anchor_p = p
            self._device_anchor_r = r
            self._hand_anchor_p = np.asarray(hand_position_m, float).copy()
            self._hand_anchor_r = (np.eye(3) if hand_rotation is None
                                   else np.asarray(hand_rotation, float).copy())
            self._last_device_p, self._last_device_r = p, r
            self._clamped = False
            return HandTarget(self._hand_anchor_p.copy(), self._hand_anchor_r.copy(),
                              event.grip, True, event.hand)

        # Tracking dropouts arrive as one enormous delta. Re-anchor on the new
        # pose and emit nothing, rather than commanding the arm across the bench
        # in a single tick.
        if self._last_device_p is not None:
            moved = float(np.linalg.norm(p - self._last_device_p))
            turned = _angle_between(r, self._last_device_r)
            if moved > self.max_jump_m or turned > self.max_jump_deg:
                self.jumps_rejected += 1
                self._device_anchor_p = p
                self._device_anchor_r = r
                self._hand_anchor_p = np.asarray(hand_position_m, float).copy()
                if hand_rotation is not None:
                    self._hand_anchor_r = np.asarray(hand_rotation, float).copy()
                self._last_device_p, self._last_device_r = p, r
                return None

        self._last_device_p, self._last_device_r = p, r

        scale = self.position_scale * max(float(event.scale), 0.0)
        target_p = self._hand_anchor_p + scale * (p - self._device_anchor_p)

        if self.use_orientation:
            target_r = (r @ self._device_anchor_r.T) @ self._hand_anchor_r
        else:
            target_r = self._hand_anchor_r

        self._clamped = False
        if self.workspace is not None and not self.workspace.contains(target_p):
            target_p = self.workspace.clamp(target_p)
            self._clamped = True

        return HandTarget(target_p, target_r, event.grip, True, event.hand)


@dataclass
class ReplayPoseSource:
    """A recorded list of events, for tests and for replaying a session.

    The point of having this at all: the retargeting can then be exercised
    exactly, with no phone, no network and no GPU.
    """

    recorded: list[PoseEvent]

    def events(self) -> Iterator[PoseEvent]:
        yield from self.recorded

    def close(self) -> None:
        pass


class TcpPoseSource:
    """Reads newline-delimited JSON pose events from the bridge.

    Stdlib only, deliberately. This runs inside the Isaac Lab environment,
    which CLAUDE.md rule 5 says not to install into casually -- so the side that
    needs FastAPI, uvicorn and the WebXR frontend is a separate process in its
    own venv, and this side needs nothing.

    It CONNECTS rather than listens. Under WSL2's default NAT the phone cannot
    reach a server inside WSL, but WSL can always reach the Windows host at the
    default gateway, so inverting the direction removes the need for mirrored
    networking or a portproxy entirely.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 timeout_s: float = 0.0005, connect_timeout_s: float = 10.0):
        """`timeout_s` is the READ timeout and wants to be sub-millisecond so a
        control loop never blocks on it. `connect_timeout_s` is a different
        quantity entirely and wants to be seconds.

        They were one parameter at first, which gave TCP connect half a
        millisecond to complete. It worked on a warm local socket and failed
        the moment anything was slower, with a bare `raise exceptions[0]` out
        of socket.create_connection and no hint about which timeout was wrong.
        """
        self.host, self.port = host, port
        self._sock = socket.create_connection((host, port),
                                              timeout=connect_timeout_s)
        self._sock.settimeout(timeout_s)
        self._buf = b""
        self.malformed = 0
        self.received = 0
        self._closed = False

    @staticmethod
    def wsl_default_gateway() -> str:
        """The Windows host, as seen from inside WSL."""
        with open("/proc/net/route", encoding="ascii") as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if parts[1] == "00000000":
                    return ".".join(str(int(parts[2][i:i + 2], 16))
                                    for i in (6, 4, 2, 0))
        raise RuntimeError("no default route; cannot locate the Windows host")

    def events(self) -> Iterator[PoseEvent]:
        while True:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield PoseEvent.from_json(json.loads(line))
                except (ValueError, json.JSONDecodeError):
                    # Counted, not raised: one corrupt packet must not end a
                    # recording session, but a stream that is mostly corrupt
                    # has to be visible rather than silently sparse.
                    self.malformed += 1

    def poll_latest(self) -> PoseEvent | None:
        """The newest event available right now, or None. Never blocks.

        A control loop must not work through a backlog: the operator would be
        watching the arm replay where the phone used to be, with the lag
        growing every tick. Only the freshest sample means anything, so the
        rest of the buffer is parsed and discarded.

        The blocking `events()` iterator stays the right thing for recording
        and replay, where every sample matters and nothing is waiting on it.
        """
        newest = None
        while True:
            try:
                chunk = self._sock.recv(65536)
            except (socket.timeout, BlockingIOError):
                break
            except OSError:
                break
            if not chunk:
                self._closed = True
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    newest = PoseEvent.from_json(json.loads(line))
                    self.received += 1
                except (ValueError, json.JSONDecodeError):
                    self.malformed += 1
        return newest

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
