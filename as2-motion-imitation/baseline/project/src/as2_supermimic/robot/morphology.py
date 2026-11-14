"""AS2-specific names and shared task geometry; not hardware calibration."""

ROOT_BODY = "base_link"
ROOT_COM = (0.0071793, 0.0034909, 0.0010238)
# Preserve official MJCF joint order, independently of the limb-feature order.
JOINT_LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in JOINT_LEGS
                    for joint in ("hip", "thigh", "calf"))
LIMB_LEGS = ("FR", "FL", "RR", "RL")
FOOT_RADIUS = 0.026
HEAD_OFFSET = 0.318  # Front of official base2_collision: 0.275 + 0.043.

# Conservative proxies enclosing the official collision primitives, with rear
# shoulder coverage. They are task hit geometry, not replacements for MJCF geoms.
TRUNK_X = (-0.25, 0.32)
TRUNK_Y = (-0.112, 0.112)
TRUNK_BELOW_BASE = 0.067
TRUNK_ABOVE_BASE = 0.135  # Tilted head cylinder reaches 0.131627 m.
THIGH_RADIUS = 0.09  # Hip-pitch cylinder's endpoint distance reaches 0.089446 m.
CALF_RADIUS = 0.04

# Kinematic stance is approximately 0.30 m (not the 0.40 m reset/drop height).
STAND_HEIGHT = 0.30
CROUCH_HEIGHT = 0.25
MIN_BASE_HEIGHT = 0.20
CONTACT_HEIGHT = FOOT_RADIUS + 0.013
NEAR_GROUND_HEIGHT = FOOT_RADIUS + 0.028
MODEL_ID = "unitree-as2-supermimic-v1"
