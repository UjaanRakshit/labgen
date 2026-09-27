"""Action terms that reproduce the lab teleop's command stage.

The real arms are driven by yam_vr_teleop (github.com/nhern026/yam_vr_teleop).
Between its IK and the motors it shapes every command; a sim that skips that
stage lets the simulated arm do things the real one is never asked to do:

  arm      joint targets clamped to within max_command_offset_rad (0.25) of the
           MEASURED position, then rate-limited to max_command_velocity_rad_s
           (1.5) relative to the last command -- config.yaml `safety`, applied
           in that order in quest_teleop.py
  gripper  proportional: an opening in [0, 1] (1 = open), slewed at most one
           full range per second -- quest_teleop.py:971,
           clip(desired, cmd - dt, cmd + dt). Over the 46.9 mm finger stroke
           that is ~0.047 m/s per finger.

Both keep the actions' sign conventions, so +1 / -1 still mean open / closed
(COBALT-style binary input and Mimic's recorded gripper actions both work).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import (BinaryJointPositionActionCfg,
                                                   DifferentialInverseKinematicsActionCfg)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class ProportionalGripperAction(BinaryJointPositionAction):
    """1-d gripper action in [-1, 1]: +1 fully open, -1 fully closed, linear
    between. The commanded opening slews at `rate_per_s` full ranges per second.

    The slew is also what keeps a close from slamming the pads into thin glass:
    an instant target jump saturates the finger drive at its force cap, which
    damping cannot slow and Newton's joint velocity limit does not bound
    (measured 0.4-0.5 m/s at contact, which shoved a beaker 45 mm).
    """

    cfg: "ProportionalGripperActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._max_step = float(cfg.rate_per_s) * float(env.physics_dt)
        self._desired = torch.ones(self.num_envs, 1, device=self.device)
        self._opening = torch.ones(self.num_envs, 1, device=self.device)
        names = list(self._asset.joint_names)
        self._ids = torch.tensor([names.index(n) for n in self._joint_names], device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._desired = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5

    def apply_actions(self):
        # Every physics step, so the slew is in physics time.
        self._opening += (self._desired - self._opening).clamp(-self._max_step, self._max_step)
        target = self._close_command + self._opening * (self._open_command - self._close_command)
        self._processed_actions = target
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos.torch[:, self._ids]
        span = self._open_command - self._close_command
        opening = ((q - self._close_command) / span).mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        self._opening[ids] = opening[ids]
        self._desired[ids] = opening[ids]


@configclass
class ProportionalGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = ProportionalGripperAction
    rate_per_s: float = 1.0
    """Full ranges per second the commanded opening may move."""


class TeleopLimitedIKAction(DifferentialInverseKinematicsAction):
    """Relative IK with the lab teleop's joint-command limits, and a NaN guard.

    Limits (when set): clamp the IK solution to within `max_command_offset_rad`
    of the measured joints, then to within `max_command_velocity_rad_s * dt` of
    the previous command -- yam_vr_teleop's order.

    NaN guard: physics runs several steps per control step with the IK
    re-solving before each. If one diverges (NaN joint state -- see
    labgen_tasks.envs.LabgenMimicEnv), the next solve would invert a NaN
    Jacobian and raise "linalg.inv: matrix is singular", killing the process
    before the env's post-step recovery can run. So the solve is skipped.
    """

    cfg: "TeleopLimitedIKActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._dt = float(env.physics_dt)
        self._last = None
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def apply_actions(self):
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if not torch.isfinite(joint_pos).all():
            return
        ee_pos, ee_quat = self._compute_frame_pose()
        if ee_quat.norm() != 0:
            des = self._ik_controller.compute(ee_pos, ee_quat, self._compute_frame_jacobian(), joint_pos)
        else:
            des = joint_pos.clone()
        off = self.cfg.max_command_offset_rad
        if off is not None:
            des = torch.clamp(des, joint_pos - off, joint_pos + off)
        vel = self.cfg.max_command_velocity_rad_s
        if vel is not None:
            if self._last is None:
                self._last = joint_pos.clone()
            self._last[self._fresh] = joint_pos[self._fresh]
            self._fresh[:] = False
            step = vel * self._dt
            des = torch.clamp(des, self._last - step, self._last + step)
            self._last = des.clone()
        self._asset.set_joint_position_target_index(target=des, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True


@configclass
class TeleopLimitedIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = TeleopLimitedIKAction
    max_command_offset_rad: float | None = None
    max_command_velocity_rad_s: float | None = None
