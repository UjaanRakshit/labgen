"""Run Isaac Lab Mimic's own tools on labgen tasks.

annotate_demos.py and generate_dataset.py only import Isaac Lab's task
packages. This registers labgen_tasks first, then runs the named tool
unchanged, in this process, with the remaining arguments.

    python mimic.py annotate --task Isaac-LabBench-YAM-VialHotplate-IK-Rel-Mimic-v0 \
        --input_file demos/x.hdf5 --output_file demos/x_annotated.hdf5 --auto
    python mimic.py generate --task ... --input_file demos/x_annotated.hdf5 \
        --output_file demos/x_generated.hdf5 --generation_num_trials 10
"""
import runpy
import sys

sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
# isaaclab_mimic ships in the same pinned Isaac Lab checkout but is not
# installed in its venv. Put its source on the path rather than pip-installing
# into that environment (CLAUDE.md rule 5): same code, same commit, and the
# environment itself is untouched.
sys.path.insert(0, "/home/ujaan/isaac/IsaacLab/source/isaaclab_mimic")
import isaaclab.app  # noqa: E402
import labgen_tasks  # noqa: E402,F401


# The real flag-adder, kept before the swap: isaaclab.app.add_launcher_args
# itself calls AppLauncher.add_app_launcher_args, so delegating to it recursed.
_ADD_ARGS = isaaclab.app.AppLauncher.add_app_launcher_args


class _KitlessApp:
    """What the tools call on `simulation_app`, for a process with no Kit."""

    def is_running(self):
        return True

    def is_exiting(self):
        return False

    def update(self):
        pass

    def close(self):
        pass


class KitlessAppLauncher:
    """Stand-in for isaaclab.app.AppLauncher on the kit-less Newton install.

    Both tools construct AppLauncher, which raises without Isaac Sim, and use
    it only to add CLI flags, start up, and poll/close the app. Kit-less,
    Isaac Lab's own launch_simulation() does nothing at all, so this does the
    same nothing -- the tools' logic runs unchanged.
    """

    def __init__(self, *args, **kwargs):
        self.app = _KitlessApp()

    @staticmethod
    def add_app_launcher_args(parser):
        _ADD_ARGS(parser)
        # Kit's flags that the tools read but the kit-less launcher does not
        # define. There is no window to open here, so headless is the truth.
        known = {a.dest for a in parser._actions}
        if "headless" not in known:
            parser.add_argument("--headless", action="store_true", default=True)
        if "enable_cameras" not in known:
            parser.add_argument("--enable_cameras", action="store_true", default=False)


isaaclab.app.AppLauncher = KitlessAppLauncher

import os  # noqa: E402

import os  # noqa: E402

if os.environ.get("LABGEN_IK_DEBUG"):
    # Diagnostic for the generator's intermittent "linalg.inv: matrix is
    # singular" in the IK. Reports non-finite IK inputs, then lets it run.
    import torch  # noqa: E402
    from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402

    _compute = DifferentialIKController.compute
    _calls = [0]

    def _checked(self, ee_pos, ee_quat, jacobian, joint_pos):
        _calls[0] += 1
        ins = dict(ee_pos=ee_pos, ee_quat=ee_quat, jacobian=jacobian, joint_pos=joint_pos,
                   ee_pos_des=self.ee_pos_des, ee_quat_des=self.ee_quat_des)
        bad = [k for k, v in ins.items() if not torch.isfinite(v).all()]
        if bad:
            print(f"IK DEBUG call {_calls[0]}: non-finite {bad}; ee_pos {ee_pos.tolist()} "
                  f"joint_pos {joint_pos.tolist()} des {self.ee_pos_des.tolist()}", flush=True)
        try:
            return _compute(self, ee_pos, ee_quat, jacobian, joint_pos)
        except Exception:
            print(f"IK DEBUG FAILED call {_calls[0]}: method {self.cfg.ik_method} "
                  f"params {self.cfg.ik_params} |J|max {float(jacobian.abs().max()):.3e} "
                  f"J shape {tuple(jacobian.shape)} dtype {jacobian.dtype}", flush=True)
            print("   J =", jacobian[0].tolist(), flush=True)
            print("   ee_pos", ee_pos.tolist(), "des", self.ee_pos_des.tolist(),
                  "joint_pos", joint_pos.tolist(), flush=True)
            raise

    DifferentialIKController.compute = _checked

TOOLS = {
    "annotate": "/home/ujaan/isaac/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py",
    "generate": "/home/ujaan/isaac/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
}

if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
    raise SystemExit(f"usage: mimic.py {{{'|'.join(TOOLS)}}} [tool args...]")
tool = TOOLS[sys.argv[1]]
sys.argv = [tool] + sys.argv[2:]
runpy.run_path(tool, run_name="__main__")
