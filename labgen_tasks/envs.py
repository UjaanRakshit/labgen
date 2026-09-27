"""Env-level workarounds shared by labgen tasks."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLMimicEnv


def restore_actuator_gains(env) -> None:
    """Re-write every implicit actuator's gains and limits from its cfg.

    Isaac Lab 3.0 beta (pinned develop commit): `env.sim.reset()` rebuilds the
    Newton model and the articulations come back with stiffness 0, damping 0
    and effort limit 1e6 on EVERY joint -- the robot is limp, held up only by
    gravity compensation. Measured: joint2 80/5/28 -> 0/0/1e6, fingers
    20000/400/25 -> 0/0/1e6. Implicit actuators keep no local copy of their
    gains, so nothing re-applies them. Isaac Lab Mimic's annotator calls
    sim.reset() before replaying every demo, so without this no demo replays.
    """
    writers = (("stiffness", "write_joint_stiffness_to_sim_index", "stiffness"),
               ("damping", "write_joint_damping_to_sim_index", "damping"),
               ("joint_effort_limit", "write_joint_effort_limit_to_sim_index", "limits"),
               ("joint_velocity_limit", "write_joint_velocity_limit_to_sim_index", "limits"),
               ("armature", "write_joint_armature_to_sim_index", "armature"))
    for art in env.scene.articulations.values():
        for act in art.actuators.values():
            ids = act.joint_indices
            ids = None if isinstance(ids, slice) else ids
            for field, method, kw in writers:
                value = getattr(act.cfg, field, None)
                if value is None:
                    continue
                if not isinstance(value, (int, float)):
                    raise NotImplementedError(f"{field} given per joint; restore handles scalars only")
                getattr(art, method)(**{kw: float(value)}, joint_ids=ids)


class LabgenMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic env base for labgen tasks: survives the annotator's sim.reset()."""

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        restore_actuator_gains(self)
        return super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
