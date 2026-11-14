"""AS2-specific checks in addition to the inherited upstream task tests."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity

from as2_supermimic.checkpoints import robot_signature, validate_checkpoint
from as2_supermimic.robot import morphology as m
from as2_supermimic.robot.as2 import AS2_XML, get_as2_robot_cfg
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.train import Actor, expand_actor_new_terms


@pytest.fixture(scope="module")
def model():
  return Entity(get_as2_robot_cfg()).spec.compile()


def test_official_assets_are_unchanged():
  root = Path(__file__).resolve().parents[1]
  manifest = json.loads((root / "third_party/unitree_rl_mjlab/manifest.json").read_text())
  for asset in manifest["files"]:
    path = root / asset["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"], asset["path"]
  assert AS2_XML.exists()


def test_names_order_stance_and_inertias(model):
  assert (model.nq, model.nv, model.nu) == (19, 18, 12)
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == m.JOINT_NAMES
  assert model.body(m.ROOT_BODY).id == 1
  np.testing.assert_allclose(model.body(m.ROOT_BODY).ipos, m.ROOT_COM)
  assert sum(model.body_mass) == pytest.approx(17.64)
  q = model.key("init_state").qpos
  for i, name in enumerate(m.JOINT_NAMES):
    expected = 0.9 if "thigh" in name else -1.8 if "calf" in name else (-0.1 if name[1] == "L" else 0.1)
    assert q[7 + i] == pytest.approx(expected)
    axis = (1, 0, 0) if "hip" in name else (0, 1, 0)
    np.testing.assert_equal(model.joint(name).axis, axis)
    assert model.joint(name).range[0] < expected < model.joint(name).range[1]
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  foot_z = np.array([data.site(leg).xpos[2] for leg in m.LIMB_LEGS])
  stance_z = data.qpos[2] - foot_z + m.FOOT_RADIUS
  np.testing.assert_allclose(stance_z, m.STAND_HEIGHT, atol=0.004)
  for leg in m.LIMB_LEGS:
    assert model.geom(f"{leg}_foot_collision").size[0] == pytest.approx(m.FOOT_RADIUS)
    assert model.geom(f"{leg}_foot_collision").condim == 3


def test_builtin_pd_matches_imagination_formula(model):
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    name = model.joint(jid).name
    calf = "calf" in name
    kp, kd, limit, armature = (60, 3, 75, 0.038) if calf else (40, 2, 50, 0.026)
    assert model.actuator_gainprm[i, 0] == kp
    assert model.actuator_biasprm[i, 2] == -kd
    assert model.dof_armature[model.jnt_dofadr[jid]] == pytest.approx(armature)
    np.testing.assert_equal(model.actuator_forcerange[i], [-limit, limit])
    data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]] + (-1 if i % 2 else 1) * 2.0
    data.qvel[model.jnt_dofadr[jid]] = 0.3
  mujoco.mj_forward(model, data)
  for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    q, qd = data.qpos[model.jnt_qposadr[jid]], data.qvel[model.jnt_dofadr[jid]]
    kp, kd = model.actuator_gainprm[i, 0], -model.actuator_biasprm[i, 2]
    expected = np.clip(kp * (data.ctrl[i] - q) - kd * qd, *model.actuator_forcerange[i])
    assert data.actuator_force[i] == pytest.approx(expected, abs=1e-8)


def primitive_points(model, data, gid):
  size, kind = model.geom_size[gid], model.geom_type[gid]
  if kind == mujoco.mjtGeom.mjGEOM_BOX:
    points = np.array(list(itertools.product((-1, 1), repeat=3))) * size
  elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = np.array([[size[0] * np.cos(a), size[0] * np.sin(a), z]
                       for z in (-size[1], size[1]) for a in angles])
  elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    points = np.concatenate([np.eye(3), -np.eye(3)]) * size[0]
  else:
    raise AssertionError(f"Unexpected collision primitive {kind}")
  return points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]


def test_shared_collision_proxy_encloses_official_primitives(model):
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  base = data.body(m.ROOT_BODY).xpos
  for gid in range(model.ngeom):
    name = model.geom(gid).name
    if not name.endswith("_collision"):
      continue
    points = primitive_points(model, data, gid)
    if name.startswith("base"):
      points -= base
      low = np.array([m.TRUNK_X[0], m.TRUNK_Y[0], -m.TRUNK_BELOW_BASE])
      high = np.array([m.TRUNK_X[1], m.TRUNK_Y[1], m.TRUNK_ABOVE_BASE])
      assert np.all(points >= low - 1e-5) and np.all(points <= high + 1e-5), name
    else:
      leg = name[:2]
      thigh = "_thigh" in name
      a = data.body(f"{leg}_{'thigh' if thigh else 'calf'}").xpos
      b = data.body(f"{leg}_calf").xpos if thigh else data.site(leg).xpos
      radius = m.THIGH_RADIUS if thigh else m.CALF_RADIUS
      t = np.clip(((points - a) @ (b - a)) / np.sum((b - a) ** 2), 0, 1)
      distances = np.linalg.norm(points - (a + t[:, None] * (b - a)), axis=1)
      assert distances.max() <= radius + 1e-5, (name, distances.max())


def test_reject_go2_and_misordered_checkpoints():
  with pytest.raises(ValueError, match="Go2 weights"):
    validate_checkpoint({"supermimic": True, "actor": {}})
  ck = {"robot": robot_signature(), "actor": {}, "cfg": {}, "obs_dim": 350,
        "obs_terms": st.OBS_TERMS}
  validate_checkpoint(ck)
  wrong = copy.deepcopy(ck)
  wrong["robot"]["joints"].reverse()
  with pytest.raises(ValueError, match="incompatible"):
    validate_checkpoint(wrong)
  with pytest.raises(ValueError, match="Incomplete training"):
    validate_checkpoint(ck, training=True)


def test_lidar_warm_start_preserves_flat_actor_exactly():
  torch.manual_seed(0)
  old_terms = st.OBS_TERMS
  new_terms = old_terms + (("lidar", 16),)
  old, new = Actor(350, (32, 32), 12), Actor(430, (32, 32), 12)
  expand_actor_new_terms(old.state_dict(), new, old_terms, new_terms, 5)
  x = torch.randn(8, 350)
  torch.testing.assert_close(old(x), new(torch.cat([x, torch.randn(8, 80)], -1)))
