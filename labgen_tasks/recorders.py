"""Record demos already annotated for Isaac Lab Mimic.

Mimic's annotate_demos.py replays each demo open-loop from its initial state
and records, before every step, the datagen info MimicGen needs: end-effector
poses, object poses, target end-effector poses and the subtask signals. That
replay has to reproduce the demo exactly, and on this backend it does not:
replay matched the recording to 0.000 mm until the pads touched the vial, then
contact nondeterminism took it 124 mm away, and Newton's deterministic mode
crashes inside MuJoCo-Warp's solver on this beta.

So record the same datagen info LIVE, during the demo, with the same recorder
terms the annotator uses (copied from annotate_demos.py, where they are
script-local). The file is annotated by construction, from the trajectory that
actually happened, and needs no replay.
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PreStepDatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        eef = {name: self._env.get_robot_eef_pose(eef_name=name)
               for name in self._env.cfg.subtask_configs.keys()}
        return "obs/datagen_info", {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsRecorder


@configclass
class MimicAnnotatedRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Actions and states (as for replay) plus Mimic's datagen info, live."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsRecorderCfg()
