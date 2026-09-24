"""Bimanual YAM at the lab bench, relative-IK control: the baseline sim.

Shaped like COBALT's Isaac Lab tasks (Isaac-*-IK-Rel), so its phone teleop and
MimicGen can drive it: 14-d action = per arm [dpos(3), drot(3), gripper(1)],
scene entities `robot0` / `robot1`, and the policy observations COBALT's
IsaacLabSimulator reads (`robot{i}_eef_pos`, `_eef_quat`, `_base_ori`,
`_hand_orn`, all quaternions wxyz as it expects).

What is the real robot's, and what is not:
  arm drives   labgen.hardware.YAM_V1 -- the gains, torque and speed limits the
               rig's i2rt 1.1.2 controller runs
  gravity      compensated through the actuators (MuJoCo actuatorgravcomp), i.e.
               inside the motors' torque limits, as the real controller does.
               Real factor on j2-j4 is 1.1-1.2; modelled as 1.0.
  gripper      pad boxes of labgen.control.YAM_PAD, UNVERIFIED; its PD gains are
               a controller choice, force capped at GRIP_FORCE_N
  second arm   placement is a PLACEHOLDER until the rig is measured
  physics      Isaac Lab's own Newton/MuJoCo-Warp grasping preset (the stack
               task's): elliptic cone, impratio 10 -- the fix for MuJoCo's
               friction creep that the grasp acceptance measured
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import (BinaryJointPositionActionCfg,
                                                   DifferentialInverseKinematicsActionCfg)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab_newton.sim.schemas import MujocoJointCfg
from isaaclab_newton.physics import (MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg,
                                     NewtonShapeCfg)

from labgen.control import YAM_JAWS
from labgen_tasks.actions import RateLimitedBinaryJointPositionActionCfg
from labgen.hardware import YAM_V1
from labgen.isaaclab_cfg import RESET_POSE_RAD
from labgen.settle import CONTACT_KD, CONTACT_KE
from labgen.types import SceneSpec

ASSETS = Path(os.environ.get("LABGEN_ASSETS", "/home/ujaan/isaac/labgen"))
YAM_USD = str(ASSETS / "assets/yam/usd/yam.usdc")          # scripts/make_yam_usd.py
SCENE_USD = str(ASSETS / "bench_arm.usda")                 # labgen.usda from the SceneSpec
SCENE_JSON = Path(__file__).resolve().parents[1] / "examples/bench_arm.json"

# Grasp point: midpoint of the two pad faces, in the `gripper` link frame.
# Computed from yam.urdf's finger joints and YAM_PAD; identical at every jaw
# opening (the jaws are symmetric), 138 mm out along the gripper's z.
GRASP_OFFSET_M = (0.0, 0.0, 0.138)

# Per-finger force ceiling. The grasp acceptance passed 10/10 at every force
# from 2 to 100 N; 25 N sits mid-range. A controller setting, not a spec.
GRIP_FORCE_N = 25.0

# Finger target speed (labgen_tasks.actions). A binary close otherwise saturates
# the drive at the force cap from the first step -- damping cannot slow a
# saturated PD, and Newton does not enforce the joint velocity limit -- so the
# pads hit the glass at 0.4-0.5 m/s, shoved the beaker 45 mm and went through
# its wall. UNVERIFIED: the real linear gripper's speed is not in the rig config.
JAW_SPEED_M_S = 0.05

# PLACEHOLDER: the second arm across the bench, facing the first. Same value
# scripts/teleop_sim.py uses. Replace with the measured base-to-base offset.
ROBOT1_POS = (0.0, 0.70, 0.0)
ROBOT1_ROT_XYZW = (0.0, 0.0, 1.0, 0.0)          # yaw 180 deg; Isaac Lab 3.0 is xyzw

# Fully open = just inside the URDF finger limit (joint7/8 lower = -0.04695 m).
# YAM_JAWS.q_open rounds it to -0.047, 0.05 mm past the limit, which Isaac Lab
# rejects as an out-of-limit default.
JAW_OPEN_Q = -0.0469

_ARM = YAM_V1
_SHOULDER = ["joint1", "joint2", "joint3"]
_WRIST = ["joint4", "joint5", "joint6"]


def _yam(prim: str, pos, rot_xyzw) -> ArticulationCfg:
    joint_pos = {f"joint{i + 1}": q for i, q in enumerate(RESET_POSE_RAD)}
    joint_pos.update({"joint7": JAW_OPEN_Q, "joint8": JAW_OPEN_Q})
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=YAM_USD,
            # Gravity compensation through the ACTUATORS (MuJoCo actuatorgravcomp),
            # so it counts against the motors' torque limits exactly as the real
            # controller's does. Not disable_gravity: that is a PhysX attribute
            # Newton ignores -- the arms sagged 309 mm and 871 mm in 2 s with it,
            # relative IK re-targeting wherever they had sagged to.
            joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)],
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot_xyzw, joint_pos=joint_pos),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=_SHOULDER,
                stiffness=_ARM.kp[0], damping=_ARM.kd[0],
                joint_effort_limit=_ARM.motors[0].torque_max_nm,
                joint_velocity_limit=_ARM.motors[0].velocity_max_rad_s,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=_WRIST,
                stiffness=_ARM.kp[3], damping=_ARM.kd[3],
                joint_effort_limit=_ARM.motors[3].torque_max_nm,
                joint_velocity_limit=_ARM.motors[3].velocity_max_rad_s,
            ),
            # The finger gains the grasp acceptance passed on (scripts/arm_drive.py
            # FINGER_KE / FINGER_KD). The damping is what matters: at the force
            # cap it closes the jaws at ~25/400 = 0.06 m/s. With kd 60 they closed
            # ~7x faster and the stiff contact flung the beaker off the bench.
            # Controller choice, not a spec: the real gripper's gains are unknown.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7", "joint8"],
                stiffness=20000.0, damping=400.0,
                joint_effort_limit=GRIP_FORCE_N,
            ),
        },
    )


def _objects() -> dict[str, RigidObjectCfg]:
    """Every free body in the SceneSpec, attached to the prim the scene USD made.

    init_state must repeat the spec's pose: Isaac Lab resets a rigid object to
    its cfg's init_state, not to where the USD put it, so leaving it at the
    default would teleport every object to the env origin on the first reset.
    """
    spec = SceneSpec.read(SCENE_JSON)
    out = {}
    for o in spec.free_bodies:
        w, x, y, z = o.orientation_wxyz
        out[o.instance_id] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{o.instance_id}",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(o.position), rot=(x, y, z, w)),
        )
    return out


_OBJ = _objects()


@configclass
class BenchYamSceneCfg(InteractiveSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Scene",
                                       spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD))
    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/light",
                                       spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot0: ArticulationCfg = _yam("Robot0", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot1: ArticulationCfg = _yam("Robot1", ROBOT1_POS, ROBOT1_ROT_XYZW)
    hotplate: RigidObjectCfg = _OBJ["hotplate"]
    petri: RigidObjectCfg = _OBJ["petri"]
    test_tube: RigidObjectCfg = _OBJ["test_tube"]
    beaker: RigidObjectCfg = _OBJ["beaker"]


# ---- observations COBALT reads ---------------------------------------------

def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)


def _grasp_pose(env: ManagerBasedRLEnv, robot: str):
    art = env.scene[robot]
    b = art.body_names.index("gripper")
    pos = art.data.body_pos_w.torch[:, b]
    quat = art.data.body_quat_w.torch[:, b]
    off = torch.tensor(GRASP_OFFSET_M, device=pos.device).expand_as(pos)
    return pos + math_utils.quat_apply(quat, off) - env.scene.env_origins, quat


def eef_pos(env, robot: str = "robot0"):
    return _grasp_pose(env, robot)[0]


def eef_quat(env, robot: str = "robot0"):
    return _xyzw_to_wxyz(_grasp_pose(env, robot)[1])


def base_ori(env, robot: str = "robot0"):
    return _xyzw_to_wxyz(env.scene[robot].data.root_quat_w.torch)


def hand_orn(env, robot: str = "robot0"):
    """Gripper orientation in its own robot's base frame, wxyz."""
    art = env.scene[robot]
    q = math_utils.quat_mul(math_utils.quat_conjugate(art.data.root_quat_w.torch),
                            art.data.body_quat_w.torch[:, art.body_names.index("gripper")])
    return _xyzw_to_wxyz(q)


def object_pose(env, name: str = "beaker"):
    obj = env.scene[name]
    return torch.cat((obj.data.root_pos_w.torch - env.scene.env_origins,
                      _xyzw_to_wxyz(obj.data.root_quat_w.torch)), dim=-1)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot0_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot0")})
        robot0_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot0")})
        robot0_eef_pos = ObsTerm(func=eef_pos, params={"robot": "robot0"})
        robot0_eef_quat = ObsTerm(func=eef_quat, params={"robot": "robot0"})
        robot0_base_ori = ObsTerm(func=base_ori, params={"robot": "robot0"})
        robot0_hand_orn = ObsTerm(func=hand_orn, params={"robot": "robot0"})
        robot1_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot1")})
        robot1_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot1")})
        robot1_eef_pos = ObsTerm(func=eef_pos, params={"robot": "robot1"})
        robot1_eef_quat = ObsTerm(func=eef_quat, params={"robot": "robot1"})
        robot1_base_ori = ObsTerm(func=base_ori, params={"robot": "robot1"})
        robot1_hand_orn = ObsTerm(func=hand_orn, params={"robot": "robot1"})
        beaker_pose = ObsTerm(func=object_pose, params={"name": "beaker"})
        test_tube_pose = ObsTerm(func=object_pose, params={"name": "test_tube"})
        petri_pose = ObsTerm(func=object_pose, params={"name": "petri"})
        hotplate_pose = ObsTerm(func=object_pose, params={"name": "hotplate"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ---- actions: [robot0 dpose(6), robot0 grip(1), robot1 dpose(6), robot1 grip(1)] ----

def _ik(robot: str) -> DifferentialInverseKinematicsActionCfg:
    return DifferentialInverseKinematicsActionCfg(
        asset_name=robot, joint_names=_SHOULDER + _WRIST, body_name="gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                               ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=list(GRASP_OFFSET_M)),
    )


def _grip(robot: str) -> RateLimitedBinaryJointPositionActionCfg:
    return RateLimitedBinaryJointPositionActionCfg(
        speed=JAW_SPEED_M_S,
        asset_name=robot, joint_names=["joint7", "joint8"],
        open_command_expr={"joint[78]": JAW_OPEN_Q},
        close_command_expr={"joint[78]": YAM_JAWS.q_closed},
    )


@configclass
class ActionsCfg:
    robot0_arm = _ik("robot0")
    robot0_gripper = _grip("robot0")
    robot1_arm = _ik("robot1")
    robot1_gripper = _grip("robot1")


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """None: this is a teleop / data-collection env, not an RL task."""


@configclass
class BenchYamIkRelEnvCfg(ManagerBasedRLEnvCfg):
    scene: BenchYamSceneCfg = BenchYamSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # Contact: labgen.settle's CONTACT_KE / CONTACT_KD, the constants the
        # grasp acceptance passed on -- not Newton's default ke=2500, under which
        # a 25 N grip sinks ~10 mm into a 2 mm glass wall and the jaws closed
        # straight through the beaker. MuJoCo needs timeconst (2/kd = 2.5 ms)
        # >= 2 x the substep, so 5 ms physics x 4 substeps = 1.25 ms.
        # Decimation 10 -> 20 Hz control, COBALT's YAM control rate.
        self.decimation = 10
        self.episode_length_s = 60.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        # Otherwise Isaac Lab's stack task Newton preset, verbatim except the
        # constraint budget, which two arms and four objects need more of.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton", integrator="implicitfast",
                njmax=600, nconmax=300,
                impratio=10.0, cone="elliptic",
                update_data_interval=2, iterations=100, ls_iterations=15,
                ls_parallel=False, use_mujoco_contacts=False, ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            # Applies to shapes with no authored contact params: in this scene,
            # all of them. mu stays Newton's 1.0 for the robot's pads (UNVERIFIED,
            # as in the grasp acceptance); objects carry their catalog friction.
            default_shape_cfg=NewtonShapeCfg(ke=CONTACT_KE, kd=CONTACT_KD),
            num_substeps=4,
            debug_mode=False,
        )
        self.viewer.eye = (1.2, -0.6, 0.9)
        self.viewer.lookat = (0.0, 0.35, 0.05)
