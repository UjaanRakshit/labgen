# Lab session — 2026-09-28

Every number below names its source or the measurement that produced it.

## Who / where

- Operator:
- Arm laptop (hostname), teleop checkout path: `GTW-ONX16-ubuntu`,
  `/home/pair/pair-mse/yam_teleop` (Jetson, project `.conda-env`).
- Sim machine (if used):

## 1. i2rt on the arm laptop

- `pip show i2rt` version: 1.1.2; see `i2rt_version.txt`.
- Source (PyPI / git commit): installed project environment; provenance of the
  wheel is described in the teleop checkout's `OPERATING.md`.

## 2. Real-arm recordings

| arm | channel | file | ran to completion? | notes |
| --- | --- | --- | --- | --- |
| right | can_right | real_right.npz | yes | Current-pose protocol, six +0.10 rad eased steps and returns. Initial reset-pose attempt and guided session are also recorded in `real_right.txt`. |
| left | can_left | real_left.npz | yes | Same current-pose protocol, all six eased steps and returns, no abort. `real_left.txt`. |

Dry run (i2rt sim) console output:

```
robot: SimRobot  source=i2rt-sim (KINEMATIC: teleports, no dynamics)
moving to the rig's reset pose over 4 s ...
ran 2100 ticks in 21.0 s; tick jitter p99 0.84 ms
wrote data/lab/2026-09-28/i2rt_sim.npz
```

The initial no-motion preflight is in `preflight.txt`. It failed on both arms:
motor 1 did not return a readable gear-ratio register on either `can_right` or
`can_left`. The owner then reported that the arms had power. Repeating the
survey produced `preflight_powered.txt`: exit status 0, with all seven motors
answering on each bus. Both CAN interfaces were up at 1 Mbit/s, and no teleop
process was running. Preflight still warns that the operator joint boxes
constrain no joints; the recordings are one arm at a time. The owner confirmed
in this session that the shared six-joint reset pose was physically checked as
clear for both arms; the file itself has no recorded date or arm-channel
metadata. No arm had been opened or commanded at this point.

After the owner confirmed e-stop attendance, a clear workspace, stopped teleop,
and someone to support the arm when torque drops, the recorder opened the right
arm. It refused the move to reset and closed; see `real_right.txt`. During a
subsequent supervised hand-guiding session under gravity compensation, joint 2
reached about 0.89 rad and joint 3 about 0.82 rad, then the readings returned
near zero. The owner said they intentionally returned it. The session was
aborted without any position trajectory; the driver closed with torque off.
The initial current-pose attempt exposed a branch error and also refused motion;
the corrected version passed a simulated real-recorder test before retrying.
The successful current-pose run finished all 2,000 ticks in 20.0 s, then closed
with torque off. Peak measured joint speed was 0.8913 rad/s, below the 3.0
rad/s abort threshold. `right_metrics.txt` has per-joint measured steps and
tracking error. Joints 3–6 did not reach 90% of their 0.10 rad targets within
their 1.5 s step segments. The saved Newton prediction uses the old reset pose
and protocol, so it is not directly comparable to this recording.
The owner described the right-arm movements as small jerks, without another
specific abnormal sound or movement. The left-arm run then finished all 2,000
ticks in 20.0 s and closed with torque off. Its peak measured speed was 0.8718
rad/s. `left_metrics.txt` shows that left joints 3–6 also did not reach 90% of
their targets within 1.5 s.

`sim2real_compare.py` output, right arm:

```
Pending a Newton run with this recording's exact start pose and protocol. The
old data/sim2real/newton_prediction.npz is not comparable.
```

`sim2real_compare.py` output, left arm:

```
Pending a Newton run with this recording's exact start pose and protocol. The
old data/sim2real/newton_prediction.npz is not comparable.
```

This Jetson teleop environment has no `newton`, `warp`, or `isaaclab` package.
On the sim machine, run `scripts/sim2real_newton.py` separately with
`--protocol-log` pointing to each real `.npz`, then compare the matching logs.

## 3. Measurements

### Gripper
- Gripper model mounted (expected `linear_4310`): Both teleop configs declare
  `linear_4310`; physical model marking not yet checked.
- Finger pad face, width across the jaw (mm):
- Finger pad face, length along the finger (mm): Operator reports a 95 mm
  rubber strip along the finger, measured with a ruler. This is the strip's
  length, not necessarily the distal contact patch used in the sim.
- Pad thickness / compliant layer (mm):
- Pad material (rubber / plastic / bare metal): Operator reports rubber contact
  pads on both arms, apparently identical; visual identification, 2026-09-28.
- Jaw gap fully open (mm), fully closed (mm): 100 mm fully open, measured
  between the inner rubber faces with a ruler; fully closed not measured.
- Tool used: Ruler. Operator also reported 19 mm of depth for the entire finger
  assembly, not the rubber layer. Face width and rubber-only thickness are not
  yet measured.

### Second arm placement (relative to the first arm's base)
- Which arm is "first" (the reference): Right arm for the requested measurement.
- Offset of second base from first, in the first arm's frame, x / y / z (mm):
  0 / +650 / 0 mm. The left base centre is 650 mm to the left of the right
  base centre, with the same forward position and height.
- Heading of the second arm relative to the first (deg, + = counter-clockwise from above): 0 deg; both face the same direction.
- How measured (tape, laser, marked points): Centre of right base to centre of
  left base with a ruler.

### Anything else noticed
- The 100 mm fully-open gap is a ruler reading, so its precision does not
  establish a discrepancy from the mesh-derived 94 mm jaw stroke by itself.

## 4. Vial demos

- Number recorded / number that completed the task:
- File:
- What felt wrong driving it (lag, a direction backwards, the view):
