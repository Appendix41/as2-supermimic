"""AS2 robot adapted from Unitree's Apache-2.0 mjlab configuration.

Pinned original files and license are recorded in third_party/unitree_rl_mjlab.
Official MJCF assets are unmodified; the task adds only a head marker.
"""

import copy
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from as2_supermimic.robot.morphology import HEAD_OFFSET, LIMB_LEGS, ROOT_BODY

AS2_XML = Path(__file__).parent / "assets/unitree_as2/xmls/as2.xml"
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LIMB_LEGS)
AS2_ACTION_SCALE = 0.25  # Unitree velocity_env_cfg, no AS2 override.


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(AS2_XML))
  spec.body(ROOT_BODY).add_site(
    name="head", pos=(HEAD_OFFSET, 0.0, 0.075), size=(0.01, 0.0, 0.0), group=5,
  )
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.4),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)
AS2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(f".*_{joint}_joint",), stiffness=kp, damping=kd,
      effort_limit=effort, armature=armature,
    )
    for joint, kp, kd, effort, armature in (
      ("hip", 40.0, 2.0, 50.0, 0.026),
      ("thigh", 40.0, 2.0, 50.0, 0.026),
      ("calf", 60.0, 3.0, 75.0, 0.038),
    )
  ),
  soft_joint_pos_limit_factor=0.9,
)
_FOOT_REGEX = "^[FR][LR]_foot_collision$"
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority=1,  # Current mjlab requires full coverage; MJCF collision default is 1.
  friction={_FOOT_REGEX: (0.6,)},
  solimp={_FOOT_REGEX: (0.9, 0.95, 0.023)},
  contype=1, conaffinity=0,
)


def get_as2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=copy.deepcopy(INIT_STATE),
    collisions=(copy.deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=copy.deepcopy(AS2_ARTICULATION),
  )
