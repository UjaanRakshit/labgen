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

`sim2real_compare.py` output, right arm (Newton run on the sim machine,
2026-09-29, `--protocol-log real_right.npz`, i.e. this recording's exact start
pose and protocol; `newton_right.npz`):

```
sim-to-real comparison: real-can_right  vs  newton-sim
joint  RMS diff  max diff  track real-can  track newton-s
   j1    0.410d    1.674d         0.571d         0.291d
   j2    0.495d    1.894d         0.641d         0.301d
   j3    0.602d    2.531d         0.715d         0.354d
   j4    0.679d    1.958d         0.860d         0.532d
   j5    0.486d    1.382d         0.733d         0.487d
   j6    0.320d    1.134d         0.622d         0.478d

  step          rise ms      overshoot %        settle ms       steady deg
       real-ca  newton- real-ca  newton- real-ca  newton- real-ca  newton-
    j1     140      140     0.0      0.0    1490      810   0.462    0.045
    j2     120      140     5.7      0.0    1490      850   0.303    0.058
    j3     nan      130     0.0      4.7    1490      890   1.052    0.049
    j4     nan      430     0.0      0.0    1490     1420   2.058    0.103
    j5     nan      450     0.0      0.0    1490     1110   1.424    0.043
    j6     nan      390     0.0      0.0    1490      750   1.074    0.002

worst RMS difference 0.679 deg against a 0.500 deg tolerance -- DISAGREE
```

`sim2real_compare.py` output, left arm (Newton run on the sim machine,
2026-09-29, `--protocol-log real_left.npz`, i.e. this recording's exact start
pose and protocol; `newton_left.npz`):

```
sim-to-real comparison: real-can_left  vs  newton-sim
joint  RMS diff  max diff  track real-can  track newton-s
   j1    0.256d    1.599d         0.467d         0.291d
   j2    0.502d    2.024d         0.660d         0.302d
   j3    0.553d    2.854d         0.697d         0.363d
   j4    0.935d    2.594d         1.077d         0.533d
   j5    0.492d    1.514d         0.753d         0.487d
   j6    0.352d    1.004d         0.646d         0.478d

  step          rise ms      overshoot %        settle ms       steady deg
       real-ca  newton- real-ca  newton- real-ca  newton- real-ca  newton-
    j1     120      140     0.0      0.0    1490      810   0.309    0.045
    j2     110      140     4.9      0.0    1490      870   0.281    0.060
    j3     nan      130     0.0      4.6    1490      880   1.620    0.048
    j4     nan      430     0.0      0.0    1490     1410   2.691    0.100
    j5     nan      450     0.0      0.0    1490     1110   1.555    0.042
    j6     nan      390     0.0      0.0    1490      750   0.943    0.002

worst RMS difference 0.935 deg against a 0.500 deg tolerance -- DISAGREE
```

### Reading the comparison (sim machine, 2026-09-29)

- Shoulder timing matches: j1/j2 rise 110-140 ms real vs 140 ms sim.
- The real arm does not reach its targets. Steady error real 0.28-2.69 deg vs
  sim 0.002-0.10 deg; j3-j6 never reach 90 % within 1.5 s on either arm.
- It is hysteresis, not an offset: every step undershoots AND every return
  stops short on the other side (final q - base is positive on all joints, both
  arms). A gravity-compensation error would offset one way; lagging both ways is
  joint friction. So the real joints carry far more friction than the sim's
  Coulomb terms (0.3 N.m j1-3, 0.06 N.m j4-6, i2rt 1.1.2 config).
- Rough implied friction, steady error x kp, from ONE protocol at ONE (folded)
  pose, so an estimate, not a fit: j1 0.4-0.6, j2 ~0.4, j3 1.5-2.3 N.m (kp 80);
  j4 0.36-0.47, j5 0.25-0.27, j6 0.17-0.19 N.m (kp 10).

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
