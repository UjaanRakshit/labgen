# Lab session — YYYY-MM-DD

Copy to `data/lab/<date>/README.md` and fill in. Every number needs its tool
and how it was taken. A number without a source is worse than a blank: leave
it blank and say why.

## Who / where

- Operator:
- Arm laptop (hostname), teleop checkout path:
- Sim machine (if used):

## 1. i2rt on the arm laptop

- `pip show i2rt` version:
- Source (PyPI / git commit):

## 2. Real-arm recordings

| arm | channel | file | ran to completion? | notes |
| --- | --- | --- | --- | --- |
| right | confirm from teleop config | real_right.npz | | |
| left | confirm from teleop config | real_left.npz | | |

Dry run (i2rt sim) console output:

```
```

`sim2real_compare.py` output, right arm:

```
```

`sim2real_compare.py` output, left arm:

```
```

## 3. Measurements

### Gripper
- Gripper model mounted (expected `linear_4310`):
- Finger pad face, width across the jaw (mm):
- Finger pad face, length along the finger (mm):
- Pad thickness / compliant layer (mm):
- Pad material (rubber / plastic / bare metal):
- Jaw gap fully open (mm), fully closed (mm):
- Tool used:

### Second arm placement (relative to the first arm's base)
- Which arm is "first" (the reference):
- Offset of second base from first, in the first arm's frame, x / y / z (mm):
  (x forward from the first arm, y to its left, z up — say if you used another convention)
- Heading of the second arm relative to the first (deg, + = counter-clockwise from above):
- How measured (tape, laser, marked points):

### Anything else noticed
- 

## 4. Vial demos

- Number recorded / number that completed the task:
- File:
- What felt wrong driving it (lag, a direction backwards, the view):
