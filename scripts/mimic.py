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

import torch  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402

_compute = DifferentialIKController.compute


def _synchronized_compute(self, ee_pos, ee_quat, jacobian, joint_pos):
    """DifferentialIKController.compute after a CUDA synchronize.

    Under the Mimic generator, the IK's damped least-squares solve crashed with
    "linalg.inv: matrix is singular" -- impossible for J J^T + lambda^2 I unless
    an input is garbage -- in 3 of 3 runs. Wrapping compute in a check that
    READ its inputs (forcing a sync) made 2 of 2 runs complete 5/5 with every
    input finite. A bare synchronize is what is left of that check: the reading
    is consistent with the Jacobian being consumed before the physics has
    finished writing it (the generator steps from an asyncio loop). The race
    itself is not pinned down; the synchronize is the measured workaround.
    """
    if ee_pos.is_cuda:
        torch.cuda.synchronize()
    return _compute(self, ee_pos, ee_quat, jacobian, joint_pos)


DifferentialIKController.compute = _synchronized_compute

TOOLS = {
    "annotate": "/home/ujaan/isaac/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py",
    "generate": "/home/ujaan/isaac/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
}

if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
    raise SystemExit(f"usage: mimic.py {{{'|'.join(TOOLS)}}} [tool args...]")
tool = TOOLS[sys.argv[1]]
sys.argv = [tool] + sys.argv[2:]
runpy.run_path(tool, run_name="__main__")
