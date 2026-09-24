"""A binary gripper action whose finger target moves at a set speed.

Isaac Lab's BinaryJointPositionAction jumps the finger target straight to open or
closed. With the drive saturated at its force cap from the first step, damping
cannot slow it, and the joint velocity limit is not enforced on the Newton
backend (measured: 0.5 m/s either way). The fingers hit a thin glass wall at
0.4-0.5 m/s, shoved a 250 mL beaker 45 mm and ended past its wall.

This ramps the target at `speed` instead, the way a real gripper's motion is
limited. Once the pads touch, the target keeps going and the PD error -- and so
the squeeze -- builds smoothly to the force cap. Same 1-d action and sign
convention as BinaryJointPositionAction (>= 0 open, < 0 close), so COBALT's
gripper command is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    cfg: "RateLimitedBinaryJointPositionActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._target = self._open_command.repeat(self.num_envs, 1).clone()
        self._max_step = float(cfg.speed) * float(env.physics_dt)

    def apply_actions(self):
        # Called every physics step, so the ramp is in physics time.
        delta = (self._processed_actions - self._target).clamp(-self._max_step, self._max_step)
        self._target += delta
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos.torch[:, self._joint_ids_torch()]
        self._target[ids] = q[ids]

    def _joint_ids_torch(self):
        names = list(self._asset.joint_names)
        return torch.tensor([names.index(n) for n in self._joint_names], device=self.device)


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type = RateLimitedBinaryJointPositionAction
    speed: float = 0.05
    """Finger target speed [m/s for prismatic joints]."""
