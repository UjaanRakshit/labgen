"""Does the model the sim uses move the SAME way as the model the real robot runs?

The real rig runs i2rt 1.1.2; the sim was built on i2rt 1.3.6. Between them the
arm links are identical, but the wrist and gripper frame were redefined:
joint6's origin rotated by pi and its axis vector flipped, and the finger joint
origins moved 149 mm. A flipped axis can be a pure relabelling (frame turned
AND axis negated -- the physical rotation unchanged) or a real sign inversion,
where +theta on the sim wrist turns the real wrist the other way. Those cannot
be told apart by reading the files, only by computing where things end up.

So: plain-numpy forward kinematics of both URDFs at the same joint angles,
compared in the WORLD frame. No simulator, so neither engine's conventions can
hide a difference.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def T(R=np.eye(3), p=np.zeros(3)):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


class URDF:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.child_of = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[j.get("name")] = dict(type=j.get("type"), xyz=xyz, R=rpy_to_R(*rpy),
                                              axis=axis, parent=parent, child=child)
            self.child_of[child] = j.get("name")

    def fk(self, q: dict) -> tuple[dict, dict]:
        """World transform of every link, and each moving joint's world axis."""
        link_T = {"base": np.eye(4)}
        axes = {}
        pending = dict(self.joints)
        while pending:
            progressed = False
            for name, j in list(pending.items()):
                if j["parent"] not in link_T:
                    continue
                Tp = link_T[j["parent"]] @ T(j["R"], j["xyz"])
                qi = q.get(name, 0.0)
                if j["type"] in ("revolute", "continuous"):
                    axes[name] = Tp[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
                    Tj = T(axis_angle(j["axis"], qi))
                elif j["type"] == "prismatic":
                    axes[name] = Tp[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
                    Tj = T(p=j["axis"] / np.linalg.norm(j["axis"]) * qi)
                else:
                    Tj = np.eye(4)
                link_T[j["child"]] = Tp @ Tj
                del pending[name]
                progressed = True
            if not progressed:
                raise RuntimeError(f"disconnected joints: {sorted(pending)}")
        return link_T, axes


def main() -> int:
    real, sim = URDF(sys.argv[1]), URDF(sys.argv[2])
    arm = [f"joint{i}" for i in range(1, 7)]
    rng = np.random.default_rng(0)
    lo = np.array([-2.618, 0.0, 0.0, -1.693, -1.571, -2.094])
    hi = np.array([3.1416, 3.665, 3.1416, 1.5708, 1.5708, 2.094])

    configs = [("home", np.zeros(6)),
               ("j6 +0.5 only", np.array([0, 0.9, 1.2, 0, 0.6, 0.5])),
               ("j6 -0.5 only", np.array([0, 0.9, 1.2, 0, 0.6, -0.5]))]
    configs += [(f"random {i}", rng.uniform(lo, hi)) for i in range(6)]

    print("world-frame comparison, real (i2rt 1.1.2) vs sim (i2rt 1.3.6)")
    print(f"{'config':14} {'link5 pos':>10} {'tip_left pos':>13} {'tip_right pos':>14} "
          f"{'j6 world axis':>14}")
    worst = dict(link5=0.0, tips=0.0, axis=0.0)
    for label, qv in configs:
        q = dict(zip(arm, qv))
        q.update(joint7=-0.02, joint8=-0.02)
        Lr, Ar = real.fk(q)
        Ls, As = sim.fk(q)
        d5 = np.linalg.norm(Lr["link5"][:3, 3] - Ls["link5"][:3, 3]) * 1000
        dl = np.linalg.norm(Lr["tip_left"][:3, 3] - Ls["tip_left"][:3, 3]) * 1000
        dr = np.linalg.norm(Lr["tip_right"][:3, 3] - Ls["tip_right"][:3, 3]) * 1000
        # Angle between the two world-frame joint6 axes, in degrees: 0 means the
        # same physical rotation, 180 means inverted.
        #
        # An earlier version compared the raw vectors against 1e-6 and printed
        # "DIFFERENT -- sign or frame inverted" while every row agreed to the
        # third decimal. joint1's origin already differs by 3.7e-6 rad between
        # the two files, so float noise alone tripped a threshold chosen with no
        # physical meaning. The positions are the real test: an inverted wrist
        # would put the tips tens of millimetres apart at j6 = +/-0.5.
        cosang = float(np.clip(Ar["joint6"] @ As["joint6"], -1.0, 1.0))
        da = float(np.degrees(np.arccos(cosang)))
        worst["link5"] = max(worst["link5"], d5)
        worst["tips"] = max(worst["tips"], dl, dr)
        worst["axis"] = max(worst["axis"], da)
        print(f"{label:14} {d5:8.3f}mm {dl:11.3f}mm {dr:12.3f}mm {da:11.5f} deg")

    print()
    print(f"worst link5 disagreement   {worst['link5']:.4f} mm   (the arm)")
    print(f"worst fingertip disagreement {worst['tips']:.4f} mm   (the gripper frame)")
    verdict = ("SAME physical rotation" if worst["axis"] < 0.01
               else "INVERTED" if worst["axis"] > 179.99 else "DIFFERENT")
    print(f"joint6 world axis          {worst['axis']:.5f} deg apart -> {verdict}")
    same = worst["link5"] < 0.01 and worst["tips"] < 0.01 and worst["axis"] < 0.01
    print()
    print("VERDICT: " + ("the two model versions describe the same physical robot "
                         "to within float precision; the file differences are a "
                         "frame relabelling" if same else
                         "the two model versions describe DIFFERENT robots"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
