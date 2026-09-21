"""Render one frame from teleop_sim's camera and save it.

Because an unset camera and a broken renderer look identical from the outside,
and shipping a blank window twice is not acceptable.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import newton
from newton.viewer import ViewerGL

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                   # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def main() -> int:
    model, arm, dofs, coords, lower, upper = T.build(Path(sys.argv[1]), Path(URDF))
    from grasp import GraspFK, N_ARM, with_fingers
    from labgen.control import YAM_JAWS, solve_pose
    fk = GraspFK(model, coords)
    lo, hi = lower[:N_ARM], upper[:N_ARM]
    seed = solve_pose(fk, T.READY_POSE, lo, hi, approach=T.DOWN)
    print("ready pose solve:", seed)

    s0 = model.state()
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = with_fingers(seed.q, YAM_JAWS.q_open)
    model.joint_q.assign(q_all.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    viewer = ViewerGL(width=960, height=540, headless=True, vsync=False)
    viewer.set_model(model)
    eye = T.LOOK_AT + np.array([0.75, -0.55, 0.55])
    for i in range(3):
        T.aim(viewer, eye, T.LOOK_AT)
        viewer.begin_frame(i / 60.0)
        viewer.log_state(s0)
        viewer.end_frame()
        f = viewer.get_frame()
    arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    out = Path(sys.argv[2])
    Image.fromarray(arr[:, :, :3]).save(out)
    nonbg = float((arr[:, :, :3].std(axis=2) > 3).mean())
    print(f"wrote {out}; {nonbg*100:.1f}% of pixels are not flat background")
    viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
