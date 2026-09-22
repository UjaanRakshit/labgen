"""Sim-to-real validation: one protocol, run on both, compared by number.

"The sim is accurate" is a claim until something measured on the real robot is
set against the same thing measured in the sim. This module is the part of that
which needs no robot and no GPU, so it can be tested: the excitation protocol,
the log format both sides write, and the comparison that turns two logs into a
report.

Joint angles are exchanged, never end-effector poses. The real rig runs i2rt
1.1.2 and the sim was built on 1.3.6; the arm is identical between them to
0.002 mm of forward kinematics, but the gripper link FRAME was redefined, so an
end-effector pose means different things on each side while a joint angle means
the same thing on both (scripts/probe_model_versions.py).

The protocol is deliberately gentle, because one side of it is real hardware:
small steps (0.10 rad) from the rig's own recorded reset pose, eased over a few
tens of milliseconds rather than jumped, one joint at a time, then a slow sine
on the shoulder. Every step is small enough that the first-instant PD torque is
well inside each motor's peak (0.10 rad x kp 80 = 8 N.m on a 28 N.m DM4340).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "RESET_POSE_RAD",
    "CONTROL_HZ",
    "Segment",
    "Protocol",
    "default_protocol",
    "Log",
    "StepMetrics",
    "step_metrics",
    "compare",
    "Comparison",
]

# The rig's recorded reset pose (pairlab/yam_teleop/deployment/config.yaml,
# citing balancing_act/yam_home.json). A pose the real arm is known to hold.
RESET_POSE_RAD = (0.0, 1.5886, 0.9016, 1.0007, 0.6594, 0.0)

# The real teleop loop's rate (deployment/config.yaml: control_hz 100).
CONTROL_HZ = 100.0


@dataclass(frozen=True)
class Segment:
    """One piece of the protocol, commanded as joint targets at CONTROL_HZ.

    kind:
      hold  stay at `base` for `seconds`
      step  ease from `base` to `base + amplitude * e_joint` over `ease_s`, then
            hold there for the rest of `seconds`
      sine  `base + amplitude * sin(2 pi freq t) * e_joint` for `seconds`
    """

    kind: str
    seconds: float
    joint: int = -1
    amplitude: float = 0.0
    freq_hz: float = 0.0
    ease_s: float = 0.05
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("hold", "step", "sine"):
            raise ValueError(f"unknown segment kind {self.kind!r}")
        if self.seconds <= 0:
            raise ValueError("a segment must last some time")
        if self.kind != "hold" and not (0 <= self.joint < 6):
            raise ValueError(f"{self.kind} segment needs a joint in 0..5")
        if abs(self.amplitude) > 0.25:
            raise ValueError(
                f"amplitude {self.amplitude} rad is too large for a protocol that "
                f"runs on real hardware; keep excitations small")


@dataclass
class Protocol:
    """A list of segments, all relative to one base pose."""

    base: tuple[float, ...] = RESET_POSE_RAD
    segments: list[Segment] = field(default_factory=list)
    hz: float = CONTROL_HZ

    def commands(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-tick joint targets and the segment index each tick belongs to.

        Steps return to the base afterwards through their own segment, so every
        step starts from rest at a known pose -- the condition step metrics
        assume.
        """
        base = np.asarray(self.base, float)
        rows, seg_ids = [], []
        dt = 1.0 / self.hz
        for i, s in enumerate(self.segments):
            n = max(int(round(s.seconds * self.hz)), 1)
            for k in range(n):
                t = k * dt
                q = base.copy()
                if s.kind == "step":
                    u = min(t / s.ease_s, 1.0) if s.ease_s > 0 else 1.0
                    ease = u * u * (3.0 - 2.0 * u)
                    q[s.joint] += s.amplitude * ease
                elif s.kind == "sine":
                    q[s.joint] += s.amplitude * math.sin(2 * math.pi * s.freq_hz * t)
                rows.append(q)
                seg_ids.append(i)
        return np.array(rows), np.array(seg_ids, dtype=int)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.segments)

    def to_json(self) -> str:
        return json.dumps({"base": list(self.base), "hz": self.hz,
                           "segments": [asdict(s) for s in self.segments]})

    @classmethod
    def from_json(cls, text: str) -> "Protocol":
        d = json.loads(text)
        return cls(base=tuple(d["base"]), hz=float(d["hz"]),
                   segments=[Segment(**s) for s in d["segments"]])


def default_protocol(amplitude: float = 0.10) -> Protocol:
    """One small step out and back per joint, then a slow shoulder sine."""
    segs = [Segment("hold", 1.0, label="settle at reset pose")]
    for j in range(6):
        segs.append(Segment("step", 1.5, joint=j, amplitude=amplitude,
                            label=f"j{j + 1} step +{amplitude:.2f} rad"))
        segs.append(Segment("hold", 1.0, label=f"j{j + 1} return"))
    segs.append(Segment("sine", 4.0, joint=1, amplitude=0.08, freq_hz=0.5,
                        label="j2 sine 0.08 rad 0.5 Hz"))
    segs.append(Segment("hold", 1.0, label="final settle"))
    return Protocol(segments=segs)


@dataclass
class Log:
    """What either side records, tick by tick. Same shape on real and sim."""

    t: np.ndarray            # (N,) seconds
    q_cmd: np.ndarray        # (N, 6) commanded joint targets
    q: np.ndarray            # (N, 6) measured joint positions
    qd: np.ndarray           # (N, 6) measured joint velocities
    eff: np.ndarray          # (N, 6) measured joint effort (N.m); NaN if unavailable
    segment: np.ndarray      # (N,) segment index
    source: str              # "real" / "i2rt-sim" / "newton-sim" ...
    protocol_json: str

    def save(self, path) -> None:
        np.savez(path, t=self.t, q_cmd=self.q_cmd, q=self.q, qd=self.qd,
                 eff=self.eff, segment=self.segment,
                 source=np.array(self.source), protocol_json=np.array(self.protocol_json))

    @classmethod
    def load(cls, path) -> "Log":
        d = np.load(path, allow_pickle=False)
        return cls(t=d["t"], q_cmd=d["q_cmd"], q=d["q"], qd=d["qd"], eff=d["eff"],
                   segment=d["segment"], source=str(d["source"]),
                   protocol_json=str(d["protocol_json"]))

    @property
    def protocol(self) -> Protocol:
        return Protocol.from_json(self.protocol_json)


@dataclass
class StepMetrics:
    joint: int
    rise_s: float          # 10-90% of the commanded change
    overshoot_pct: float
    settle_s: float        # last exit from the 2% band
    steady_err_rad: float  # |final - target|


def step_metrics(y: np.ndarray, t: np.ndarray, start: float, goal: float) -> StepMetrics:
    """Classic step-response figures for one joint, from rest at `start`."""
    d = goal - start
    if abs(d) < 1e-9:
        raise ValueError("a step needs a non-zero commanded change")
    frac = (np.asarray(y, float) - start) / d
    t = np.asarray(t, float) - t[0]
    if frac.max() >= 0.9 and (frac >= 0.1).any():
        rise = float(t[np.argmax(frac >= 0.9)] - t[np.argmax(frac >= 0.1)])
    else:
        rise = float("nan")
    overshoot = max(0.0, (float(frac.max()) - 1.0) * 100.0)
    outside = np.where(np.abs(frac - 1.0) > 0.02)[0]
    settle = float(t[outside[-1]]) if len(outside) else 0.0
    return StepMetrics(joint=-1, rise_s=rise, overshoot_pct=overshoot,
                       settle_s=settle, steady_err_rad=float(abs(y[-1] - goal)))


def _steps(log: Log) -> list[StepMetrics]:
    proto = log.protocol
    base = np.asarray(proto.base, float)
    out = []
    for i, s in enumerate(proto.segments):
        if s.kind != "step":
            continue
        idx = np.where(log.segment == i)[0]
        if len(idx) < 3:
            continue
        m = step_metrics(log.q[idx, s.joint], log.t[idx], base[s.joint],
                         base[s.joint] + s.amplitude)
        m.joint = s.joint
        out.append(m)
    return out


@dataclass
class Comparison:
    a: str
    b: str
    rms_diff_deg: np.ndarray          # per joint, a.q vs b.q over the whole run
    max_diff_deg: np.ndarray
    track_rms_deg_a: np.ndarray       # per joint, a.q vs a.q_cmd
    track_rms_deg_b: np.ndarray
    steps_a: list[StepMetrics]
    steps_b: list[StepMetrics]
    tolerance_deg: float

    @property
    def ok(self) -> bool:
        return bool(np.all(self.rms_diff_deg <= self.tolerance_deg))

    def __str__(self) -> str:
        rows = [f"sim-to-real comparison: {self.a}  vs  {self.b}",
                f"{'joint':>5} {'RMS diff':>9} {'max diff':>9} {'track '+self.a[:8]:>15} "
                f"{'track '+self.b[:8]:>15}"]
        for j in range(6):
            rows.append(f"{'j'+str(j+1):>5} {self.rms_diff_deg[j]:8.3f}d {self.max_diff_deg[j]:8.3f}d "
                        f"{self.track_rms_deg_a[j]:13.3f}d {self.track_rms_deg_b[j]:13.3f}d")
        rows.append("")
        rows.append(f"{'step':>6} {'rise ms':>16} {'overshoot %':>16} {'settle ms':>16} {'steady deg':>16}")
        rows.append(f"{'':>6} {self.a[:7]:>7} {self.b[:7]:>8} {self.a[:7]:>7} {self.b[:7]:>8} "
                    f"{self.a[:7]:>7} {self.b[:7]:>8} {self.a[:7]:>7} {self.b[:7]:>8}")
        for sa, sb in zip(self.steps_a, self.steps_b):
            rows.append(f"{'j'+str(sa.joint+1):>6} {sa.rise_s*1000:7.0f} {sb.rise_s*1000:8.0f} "
                        f"{sa.overshoot_pct:7.1f} {sb.overshoot_pct:8.1f} "
                        f"{sa.settle_s*1000:7.0f} {sb.settle_s*1000:8.0f} "
                        f"{math.degrees(sa.steady_err_rad):7.3f} {math.degrees(sb.steady_err_rad):8.3f}")
        rows.append("")
        rows.append(f"worst RMS difference {self.rms_diff_deg.max():.3f} deg against a "
                    f"{self.tolerance_deg:.3f} deg tolerance -- "
                    f"{'AGREE' if self.ok else 'DISAGREE'}")
        return "\n".join(rows)


def compare(a: Log, b: Log, tolerance_deg: float = 0.5) -> Comparison:
    """Two logs of the SAME protocol, compared joint by joint.

    Refuses logs of different protocols, because a difference then measures the
    protocols, not the robots.
    """
    if a.protocol_json != b.protocol_json:
        raise ValueError("the two logs ran different protocols; they cannot be compared")
    n = min(len(a.t), len(b.t))
    if n < 10:
        raise ValueError("logs are too short to compare")
    if abs(len(a.t) - len(b.t)) > max(2, 0.01 * n):
        raise ValueError(f"logs differ in length ({len(a.t)} vs {len(b.t)} ticks): "
                         f"one side dropped or duplicated ticks, so they are not aligned")
    diff = np.degrees(a.q[:n] - b.q[:n])
    return Comparison(
        a=a.source, b=b.source,
        rms_diff_deg=np.sqrt(np.mean(diff ** 2, axis=0)),
        max_diff_deg=np.abs(diff).max(axis=0),
        track_rms_deg_a=np.degrees(np.sqrt(np.mean((a.q - a.q_cmd) ** 2, axis=0))),
        track_rms_deg_b=np.degrees(np.sqrt(np.mean((b.q - b.q_cmd) ** 2, axis=0))),
        steps_a=_steps(a), steps_b=_steps(b), tolerance_deg=tolerance_deg)
