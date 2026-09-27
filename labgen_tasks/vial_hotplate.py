"""Vial task: pick the 20 mL vial, set it on the hotplate, then take it back off.

Built on the bench task (bench_yam): same arms, drives, contact and physics;
this adds the vial, a success check, and the subtask signals Isaac Lab Mimic
annotates demos with.

    grasp_1   robot0 is holding the vial
    place_1   the vial stands upright on the hotplate and the jaws have opened
    grasp_2   robot0 is holding the vial again, after place_1
    (final)   success: the vial stands upright back on the bench, released,
              having been on the hotplate first

The vial is grasped at its BODY (mid-height). vial_20ml's OD and height are
confirmed but its neck and shoulder are not in the catalog, so a neck grasp
would be on geometry nobody measured. The hotplate's footprint and plate
height are catalog TODOs too; the task inherits that flag.

Only robot0 does this task; robot1 holds station with its jaws open.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from labgen.catalog import CATALOG
from labgen_tasks import bench_yam as B
from labgen_tasks.envs import LabgenMimicEnv

REPO = Path(__file__).resolve().parents[1]
VIAL_JSON = REPO / "examples/bench_vial.json"
VIAL_USD = str(REPO / "out/bench_vial.usda")           # scripts/make_bench_vial.py

VIAL = CATALOG["vial_20ml"]
HOT = CATALOG["hotplate_stirrer"]
VIAL_GRASP_Z = VIAL.keypoints["body_grasp"][2]          # 30.5 mm up the body
PLATE_Z = HOT.dims["z"]                                 # plate surface above its base
PLATE_HX, PLATE_HY = HOT.dims["x"] / 2, HOT.dims["y"] / 2

# Thresholds. Tolerances on a physical state, chosen here, not measured facts.
NEAR_M = 0.025          # grasp point within this of the vial's body grasp point
REST_M = 0.006          # vial base within this of the surface it stands on
EDGE_M = 0.010          # vial centre at least this far inside the plate edge
UPRIGHT_COS = 0.966     # within 15 deg of vertical
# Jaw joint (0 closed, -0.047 open 94 mm). Closed on a 28 mm vial sits near
# -0.014; "holding" is anything between nearly shut and half open.
HOLD_Q = (-0.025, -0.004)
OPEN_Q = -0.028         # jaws opened past ~56 mm: released

_OBJ = B._objects(VIAL_JSON)


@configclass
class VialSceneCfg(B.BenchYamSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene",
                                       spawn=sim_utils.UsdFileCfg(usd_path=VIAL_USD))
    vial: RigidObjectCfg = _OBJ["vial"]


# ---- state predicates ---------------------------------------------------------

def _pose(env, name):
    o = env.scene[name]
    return o.data.root_pos_w.torch - env.scene.env_origins, o.data.root_quat_w.torch


def _upright(q_xyzw):
    x, y = q_xyzw[..., 0], q_xyzw[..., 1]
    return (1 - 2 * (x * x + y * y)) > UPRIGHT_COS           # body z . world z


def _jaw(env):
    art = env.scene["robot0"]
    return art.data.joint_pos.torch[:, art.joint_names.index("joint7")]


def vial_grasped(env) -> torch.Tensor:
    vp, vq = _pose(env, "vial")
    body = vp + PoseUtils.quat_apply(vq, torch.tensor([0.0, 0.0, VIAL_GRASP_Z],
                                                      device=vp.device).expand_as(vp))
    near = (B._grasp_pose(env, "robot0")[0] - body).norm(dim=-1) < NEAR_M
    q = _jaw(env)
    return near & (q > HOLD_Q[0]) & (q < HOLD_Q[1])


def vial_on_plate(env) -> torch.Tensor:
    vp, vq = _pose(env, "vial")
    hp, hq = _pose(env, "hotplate")
    rel = PoseUtils.quat_apply_inverse(hq, vp - hp)
    inside = (rel[:, 0].abs() < PLATE_HX - EDGE_M) & (rel[:, 1].abs() < PLATE_HY - EDGE_M)
    return inside & ((rel[:, 2] - PLATE_Z).abs() < REST_M) & _upright(vq)


def released(env) -> torch.Tensor:
    return _jaw(env) < OPEN_Q


def _placed_latch(env) -> torch.Tensor:
    """True once the vial has been set on the plate and let go, this episode.

    Latched on the env and cleared when an episode starts, because grasp_2 and
    success both mean "after it has been on the plate", which the current state
    alone cannot tell apart from the first pick carrying it over the plate.
    """
    if not hasattr(env, "_labgen_placed"):
        env._labgen_placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._labgen_placed &= env.episode_length_buf > 0
    env._labgen_placed |= vial_on_plate(env) & released(env)
    return env._labgen_placed


def place_1(env) -> torch.Tensor:
    return vial_on_plate(env) & released(env)


def grasp_2(env) -> torch.Tensor:
    return vial_grasped(env) & _placed_latch(env)


def task_success(env) -> torch.Tensor:
    vp, vq = _pose(env, "vial")
    on_bench = (vp[:, 2].abs() < REST_M) & _upright(vq)
    return _placed_latch(env) & on_bench & released(env)


def _bool_obs(fn):
    return lambda env: fn(env).float().unsqueeze(-1)


# ---- task cfg -----------------------------------------------------------------

@configclass
class VialObservationsCfg:
    @configclass
    class PolicyCfg(B.ObservationsCfg.PolicyCfg):
        vial_pose = ObsTerm(func=B.object_pose, params={"name": "vial"})

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_1 = ObsTerm(func=_bool_obs(vial_grasped))
        place_1 = ObsTerm(func=_bool_obs(place_1))
        grasp_2 = ObsTerm(func=_bool_obs(grasp_2))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class VialEventCfg(B.EventCfg):
    # The vial starts somewhere in a 8 x 8 cm square around its nominal spot,
    # so demos -- and what Mimic generates from them -- are not all one pose.
    # +-4 cm keeps it >= 7 cm from the petri dish and the test tube.
    randomize_vial = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                "velocity_range": {}, "asset_cfg": SceneEntityCfg("vial")})


@configclass
class VialTerminationsCfg(B.TerminationsCfg):
    success = DoneTerm(func=task_success)


@configclass
class VialHotplateEnvCfg(B.BenchYamIkRelEnvCfg):
    scene: VialSceneCfg = VialSceneCfg(num_envs=1, env_spacing=3.0)
    observations: VialObservationsCfg = VialObservationsCfg()
    events: VialEventCfg = VialEventCfg()
    terminations: VialTerminationsCfg = VialTerminationsCfg()


# ---- Isaac Lab Mimic ------------------------------------------------------------

EEF = "robot0"


class VialHotplateMimicEnv(LabgenMimicEnv):
    """Mimic's view of the vial task: one end effector (robot0), 14-d actions.

    Poses are the pad-midpoint grasp point in the env frame, the frame the IK
    action is closed on. Actions are relative-IK deltas in robot0's base frame,
    divided by the action scale, so a replayed action reproduces the delta.
    robot1 gets a zero delta and open jaws throughout.
    """

    def _base_R(self, env_ids=slice(None)):
        return PoseUtils.matrix_from_quat(self.scene[EEF].data.root_quat_w.torch[env_ids])

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        ids = slice(None) if env_ids is None else env_ids
        pos, quat = B._grasp_pose(self, EEF)
        return PoseUtils.make_pose(pos[ids], PoseUtils.matrix_from_quat(quat[ids]))

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict,
                                  action_noise_dict=None, env_id: int = 0) -> torch.Tensor:
        scale = self.cfg.actions.robot0_arm.scale
        tgt_pos, tgt_rot = PoseUtils.unmake_pose(target_eef_pose_dict[EEF])
        cur_pos, cur_rot = PoseUtils.unmake_pose(self.get_robot_eef_pose(EEF, env_ids=[env_id])[0])
        d_rot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(tgt_rot @ cur_rot.T))
        Rb = self._base_R([env_id])[0]
        pose = torch.cat([Rb.T @ (tgt_pos - cur_pos), Rb.T @ d_rot]) / scale
        if action_noise_dict is not None:
            pose = pose + action_noise_dict[EEF] * torch.randn_like(pose)
        a = torch.zeros(14, device=pose.device)
        a[0:6] = pose
        a[6] = gripper_action_dict[EEF].reshape(-1)[0]
        a[13] = 1.0                                   # robot1 jaws open
        if not torch.isfinite(a).all():
            raise FloatingPointError(
                f"non-finite Mimic action {a.tolist()} from target pos {tgt_pos.tolist()} "
                f"rot {tgt_rot.tolist()}, current pos {cur_pos.tolist()}, d_rot {d_rot.tolist()}, "
                f"gripper {gripper_action_dict[EEF]}")
        return a

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        scale = self.cfg.actions.robot0_arm.scale
        Rb = self._base_R()
        d_pos = (Rb @ (action[:, 0:3] * scale).unsqueeze(-1)).squeeze(-1)
        d_rot = (Rb @ (action[:, 3:6] * scale).unsqueeze(-1)).squeeze(-1)
        cur_pos, cur_rot = PoseUtils.unmake_pose(self.get_robot_eef_pose(EEF))
        ang = d_rot.norm(dim=-1)
        axis = torch.where(ang.unsqueeze(-1) > 1e-9, d_rot / ang.clamp_min(1e-9).unsqueeze(-1),
                           torch.zeros_like(d_rot))
        R = PoseUtils.matrix_from_quat(PoseUtils.quat_from_angle_axis(ang, axis))
        return {EEF: PoseUtils.make_pose(cur_pos + d_pos, R @ cur_rot)}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF: actions[..., 6:7]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        t = self.obs_buf["subtask_terms"]
        return {k: t[k][ids] for k in ("grasp_1", "place_1", "grasp_2")}


def _subtask(obj, signal, desc):
    return SubTaskConfig(
        object_ref=obj, subtask_term_signal=signal,
        subtask_term_offset_range=(0, 0) if signal is None else (5, 10),
        selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.01, num_interpolation_steps=5, num_fixed_steps=0,
        apply_noise_during_interpolation=False, description=desc)


@configclass
class VialHotplateMimicEnvCfg(VialHotplateEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "labgen_vial_hotplate"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.seed = 1
        self.subtask_configs[EEF] = [
            _subtask("vial", "grasp_1", "grasp the vial"),
            _subtask("hotplate", "place_1", "set the vial on the hotplate and let go"),
            _subtask("vial", "grasp_2", "grasp the vial again"),
            _subtask("hotplate", None, "set the vial back on the bench"),
        ]
