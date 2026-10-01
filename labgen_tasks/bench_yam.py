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
  command      the lab teleop's own limits (yam_vr_teleop): joint targets within
               0.25 rad of measured and slewed at <= 1.5 rad/s; proportional
               gripper slewed at one full range per second
  gripper      pad boxes of labgen.control.YAM_PAD, UNVERIFIED; its PD gains are
               a controller choice, force capped at GRIP_FORCE_N
  arm layout   MEASURED (ruler, lab, 2026-09-28): robot0 = right arm at the origin,
               robot1 = left arm 650 mm to its left (+y), both facing +x
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
from labgen_tasks.actions import ProportionalGripperActionCfg, TeleopLimitedIKActionCfg
from labgen.hardware import YAM_V1
from labgen.isaaclab_cfg import RESET_POSE_RAD
from labgen.settle import CONTACT_KD, CONTACT_KE
from labgen.types import SceneSpec

REPO = Path(__file__).resolve().parents[1]


def _asset(env: str, *candidates: Path) -> str:
    """$env if set, else the first candidate that exists, else the first (so a
    missing asset fails loudly at spawn with a path that says what to build)."""
    if os.environ.get(env):
        return os.environ[env]
    return str(next((c for c in candidates if c.exists()), candidates[0]))


# Built by scripts/make_yam_usd.py (default out/yam/) and from the SceneSpec by
# labgen.usda (out/). The /home/ujaan paths are this project's original WSL
# machine, kept as fallbacks.
YAM_USD = _asset("LABGEN_YAM_USD", REPO / "out/yam/usd/yam.usdc",
                 Path("/home/ujaan/isaac/labgen/assets/yam/usd/yam.usdc"))
SCENE_USD = _asset("LABGEN_SCENE_USD", REPO / "out/bench_arm.usda",
                   Path("/home/ujaan/isaac/labgen/bench_arm.usda"))
SCENE_JSON = REPO / "examples/bench_arm.json"

# Grasp point: midpoint of the two pad faces, in the `gripper` link frame.
# Computed from yam.urdf's finger joints and YAM_PAD; identical at every jaw
# opening (the jaws are symmetric), 138 mm out along the gripper's z.
GRASP_OFFSET_M = (0.0, 0.0, 0.138)

# Per-finger force ceiling. The grasp acceptance passed 10/10 at every force
# from 2 to 100 N; 25 N sits mid-range. A controller setting, not a spec.
GRIP_FORCE_N = 25.0

# The lab teleop's command stage (yam_vr_teleop, github.com/nhern026/yam_vr_teleop):
# deployment/config.yaml `safety` for the arm, quest_teleop.py:971 for the
# gripper slew. SOURCED from the code that drives the real arms -- these are the
# limits the real arm is commanded under, applied by labgen_tasks.actions.
TELEOP_MAX_COMMAND_OFFSET_RAD = 0.25      # target never further than this from measured
TELEOP_MAX_COMMAND_VELOCITY_RAD_S = 1.5   # commanded joint angle change per second
TELEOP_GRIPPER_RATE_PER_S = 1.0           # opening command, full ranges per second (~0.047 m/s/finger)

# The left arm, measured at the lab (data/lab/2026-09-28/README.md, ruler, centre
# to centre): 650 mm to the LEFT of the right arm, same forward position and
# height, facing the same way. The YAM faces +x, so left is +y. Replaces a
# placeholder that had it 0.70 m across the bench, turned to face the first.
ROBOT1_POS = (0.0, 0.65, 0.0)
ROBOT1_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)          # same heading; Isaac Lab 3.0 is xyzw

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


def _objects(scene_json: Path = SCENE_JSON) -> dict[str, RigidObjectCfg]:
    """Every free body in the SceneSpec, attached to the prim the scene USD made.

    init_state must repeat the spec's pose: Isaac Lab resets a rigid object to
    its cfg's init_state, not to where the USD put it, so leaving it at the
    default would teleport every object to the env origin on the first reset.
    """
    spec = SceneSpec.read(scene_json)
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


def joint_target(env, robot: str = "robot0"):
    """Commanded joint positions (all 8: arm, then the two fingers). The lab's
    data format records the commanded target as the action; this is it."""
    return env.scene[robot].data.joint_pos_target.torch


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
        robot0_joint_target = ObsTerm(func=joint_target, params={"robot": "robot0"})
        robot0_eef_pos = ObsTerm(func=eef_pos, params={"robot": "robot0"})
        robot0_eef_quat = ObsTerm(func=eef_quat, params={"robot": "robot0"})
        robot0_base_ori = ObsTerm(func=base_ori, params={"robot": "robot0"})
        robot0_hand_orn = ObsTerm(func=hand_orn, params={"robot": "robot0"})
        robot1_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot1")})
        robot1_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot1")})
        robot1_joint_target = ObsTerm(func=joint_target, params={"robot": "robot1"})
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

def _ik(robot: str) -> TeleopLimitedIKActionCfg:
    return TeleopLimitedIKActionCfg(
        max_command_offset_rad=TELEOP_MAX_COMMAND_OFFSET_RAD,
        max_command_velocity_rad_s=TELEOP_MAX_COMMAND_VELOCITY_RAD_S,
        asset_name=robot, joint_names=_SHOULDER + _WRIST, body_name="gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                               ik_method="dls"),
        scale=0.5,
        body_offset=TeleopLimitedIKActionCfg.OffsetCfg(pos=list(GRASP_OFFSET_M)),
    )


def _grip(robot: str) -> ProportionalGripperActionCfg:
    return ProportionalGripperActionCfg(
        rate_per_s=TELEOP_GRIPPER_RATE_PER_S,
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
        # >= 2 x the substep: 25 ms physics steps x 20 substeps = 1.25 ms.
        #
        # Why 25 ms x 20 and not 5 ms x 4 (the same substep): the cost is per
        # PHYSICS step, not per substep -- relative IK re-solves both arms every
        # physics step. Measured per 20 Hz control step, pick test passing in
        # every case: 5 ms x 4 -> 74 ms, 10 ms x 8 -> 51 ms, 25 ms x 20 -> 38 ms.
        # Solver iterations (100 -> 8) and collision decimation changed nothing.
        # Decimation 1 -> 40 Hz control. Measured with the pick test passing at each:
        # 20 Hz (x2 steps) 47.7 ms/step, 40 Hz 21.3 ms of a 25 ms budget, 50 Hz
        # 18.3 ms of 20 (no headroom). 40 Hz leaves room for the teleop loop; the
        # phone itself samples at 20 Hz and the real rig's loop runs at 100 Hz.
        self.decimation = 1
        self.episode_length_s = 60.0
        self.sim.dt = 0.025
        self.sim.render_interval = 1
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
            num_substeps=20,
            debug_mode=False,
        )
        # Behind and above the arms, looking forward along +x: the operator's view.
        self.viewer.eye = (-0.75, 0.325, 0.85)
        self.viewer.lookat = (0.35, 0.325, 0.05)
