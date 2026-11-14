"""Reject checkpoints with incompatible robot, control or observation semantics."""

import hashlib
import json

from as2_supermimic.robot import morphology as m
from as2_supermimic.robot.as2 import AS2_ACTION_SCALE, AS2_ARTICULATION, AS2_XML, INIT_STATE


def robot_signature() -> dict:
  return {
    "id": m.MODEL_ID, "asset_sha256": hashlib.sha256(AS2_XML.read_bytes()).hexdigest(),
    "joints": list(m.JOINT_NAMES), "limbs": list(m.LIMB_LEGS),
    "root": m.ROOT_BODY, "default_joint_pos": dict(INIT_STATE.joint_pos),
    "action_scale": AS2_ACTION_SCALE,
    "actuators": [
      [list(a.target_names_expr), a.stiffness, a.damping, a.effort_limit, a.armature]
      for a in AS2_ARTICULATION.actuators
    ],
    "geometry": {
      name: getattr(m, name) for name in (
        "FOOT_RADIUS", "HEAD_OFFSET", "TRUNK_X", "TRUNK_Y", "TRUNK_BELOW_BASE",
        "TRUNK_ABOVE_BASE", "THIGH_RADIUS", "CALF_RADIUS", "STAND_HEIGHT",
        "CROUCH_HEIGHT", "MIN_BASE_HEIGHT",
      )
    },
  }


def validate_checkpoint(checkpoint: dict, *, training: bool = False) -> None:
  expected = json.dumps(robot_signature(), sort_keys=True)
  actual = json.dumps(checkpoint.get("robot"), sort_keys=True)
  if actual != expected:
    raise ValueError("Checkpoint is not this AS2 model/control layout. Go2 weights are incompatible.")
  required = ("actor", "cfg", "obs_dim", "obs_terms")
  if training:
    required += ("world_model", "critic", "iteration")
  missing = [key for key in required if key not in checkpoint]
  if missing:
    raise ValueError(f"Incomplete {'training' if training else 'policy'} checkpoint: {missing}")
