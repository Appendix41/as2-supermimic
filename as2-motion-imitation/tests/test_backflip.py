import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from as2_motion_imitation.backflip_audit import ACCEPTANCE, summarize
from as2_motion_imitation.physics import Physics, make_model
from as2_motion_imitation.prepare_backflip import INITIAL, orientation, targets
from as2_motion_imitation.motion import MotionClip, sha256


ROOT = Path(__file__).resolve().parents[1]


def quaternion(angle):
    return np.stack((np.cos(angle/2), np.zeros_like(angle),
                     -np.sin(angle/2), np.zeros_like(angle)), axis=-1)


def test_continuous_back_rotation_not_euler_pitch():
    angles = np.linspace(0, 2*np.pi, 201)
    rotation, lateral, upright = orientation(quaternion(angles))
    np.testing.assert_allclose(rotation, angles, atol=1e-12)
    np.testing.assert_allclose(lateral, 1, atol=1e-12)
    np.testing.assert_allclose(upright, np.cos(angles), atol=1e-12)
    forward, _, _ = orientation(quaternion(-angles))
    assert forward[-1] < -6.2


def test_lateral_heading_remains_continuous_across_vertical():
    p = Physics(state_frame="lateral-axis-v2")
    headings, gravity = [], []
    for angle in np.linspace(0, 2*np.pi, 401):
        p.data.qpos[3:7] = quaternion(np.array(angle))
        mujoco.mj_forward(p.model, p.data)
        state, pose = p.state()
        headings.append(pose[2])
        gravity.append(state[1:4])
    np.testing.assert_allclose(headings, 0, atol=1e-7)
    assert np.abs(np.diff(gravity, axis=0)).max() < .02


def test_unknown_frame_rejected():
    with pytest.raises(ValueError, match="state frame"):
        Physics(state_frame="unversioned")


def test_search_sensors_do_not_change_dynamics():
    plain, measured = Physics(), Physics(make_model(contact_audit=True))
    for key in ("body_mass", "body_inertia", "jnt_range", "actuator_forcerange",
                "actuator_gainprm", "actuator_biasprm", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(plain.model, key), getattr(measured.model, key))
    for q in targets(INITIAL, np.arange(0, 2.4, .02))[0]:
        action = (q - plain.default_q) / .25
        plain.step(action)
        measured.step(action)
        np.testing.assert_array_equal(plain.data.qpos, measured.data.qpos)
    floor = measured.model.geom("floor").id
    touching = set()
    for con in measured.data.contact:
        if con.dist <= 0 and floor in (con.geom1, con.geom2):
            touching.add(con.geom2 if con.geom1 == floor else con.geom1)
    for i in range(measured.model.nsensor):
        sensor = measured.model.sensor(i)
        if sensor.name.startswith("audit_"):
            geom = measured.model.geom(sensor.name.removeprefix("audit_")).id
            found, distance = measured.data.sensordata[sensor.adr[0]:sensor.adr[0]+2]
            assert bool(found > .5 and distance <= 0) == (geom in touching)


def synthetic_trace():
    t = np.arange(0, 8.002, .002)
    progress = np.clip((t-1.8)/.6, 0, 1)
    base = np.zeros((len(t), 3))
    base[:, 2] = .272
    airborne = (t >= 1.8) & (t <= 2.4)
    return {
        "time": t, "base_pos": base, "base_quat": quaternion(progress * 2*np.pi),
        "base_vel": np.zeros_like(base), "base_ang_vel": np.zeros_like(base),
        "feet_contact": np.where(airborne, 0, 4), "nonfoot": np.zeros(len(t), bool),
        "joint_limit": np.zeros(len(t), bool), "joint_speed": np.zeros(len(t)),
        "standing_joint_error": np.zeros(len(t)), "torque": np.zeros((len(t), 12)),
        "torque_saturation": np.zeros(len(t)),
    }


@pytest.mark.parametrize("failure", [
    "ground_roll", "forward_flip", "half_flip", "nonfoot", "limit", "unstable",
    "lateral", "speed", "side_roll",
])
def test_backflip_protocol_rejects_false_success(failure):
    trace = synthetic_trace()
    criteria = json.loads(ACCEPTANCE.read_text())
    assert summarize(trace, criteria)["passed"]
    if failure == "ground_roll":
        trace["feet_contact"][:] = 4
    elif failure == "forward_flip":
        trace["base_quat"][:, 2] *= -1
    elif failure == "half_flip":
        trace["base_quat"] = quaternion(np.clip((trace["time"] - 1.8)/.6, 0, 1) * np.pi)
    elif failure == "nonfoot":
        trace["nonfoot"][1000] = True
    elif failure == "limit":
        trace["joint_limit"][1000] = True
    elif failure == "unstable":
        trace["base_vel"][-1000:, 0] = .4
    elif failure == "lateral":
        trace["base_pos"][:, 1] = np.linspace(0, .4, len(trace["time"]))
    elif failure == "speed":
        trace["joint_speed"][1000] = 35
    elif failure == "side_roll":
        trace["base_quat"][:, [1, 2]] = trace["base_quat"][:, [2, 1]]
    assert not summarize(trace, criteria)["passed"]


def test_cpu_gpu_lateral_frame_full_rotation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from as2_motion_imitation.gpu_physics import GpuPhysics

    p = Physics(state_frame="lateral-axis-v2")
    gpu = GpuPhysics(8, state_frame="lateral-axis-v2")
    qpos = np.tile(p.data.qpos, (8, 1))
    qvel = np.tile(p.data.qvel, (8, 1))
    qpos[:, 2] = 1.
    qpos[:, 3:7] = quaternion(np.linspace(0, 2*np.pi, 8))
    qvel[:, 4] = -8.
    gpu.reset_states(gpu.all_ids, torch.tensor(qpos, device="cuda", dtype=torch.float32),
                     torch.tensor(qvel, device="cuda", dtype=torch.float32))
    states, poses = gpu.state()
    for i in range(8):
        p.data.qpos[:], p.data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(p.model, p.data)
        s, pose = p.state()
        np.testing.assert_allclose(states[i].cpu(), s, atol=3e-5)
        np.testing.assert_allclose(poses[i].cpu(), pose, atol=3e-5)


@pytest.fixture(scope="module")
def backflip():
    return MotionClip(ROOT / "data/processed/as2_backflip.npz")


def test_legacy_reference_keeps_original_frame():
    assert MotionClip(ROOT / "data/processed/as2_triple_forward_jump.npz").state_frame == "forward-axis-v1"


def test_backflip_reference_contract_and_continuous_goals(backflip):
    from as2_motion_imitation.demonstration import RecordedCommands
    from as2_motion_imitation.task import TrackingTask

    assert backflip.state_frame == "lateral-axis-v2"
    assert backflip.frames == 401 and backflip.duration == 8.
    commands = RecordedCommands(ROOT / "data/raw/as2-native-backflip/physics_capture.npz", backflip)
    assert commands.actions.shape == (400, 12)
    time = torch.arange(401) * .02
    sample = backflip.sample(time)
    task = TrackingTask(backflip)
    assert not task.failure(sample["state"], time).any()
    assert sample["state"][:, 3].max() > .99
    assert torch.diff(sample["pose"][:, 2]).abs().max() < .1
    origin = backflip.poses[0].repeat(401, 1)
    goals = task.goal(sample["state"], sample["pose"], time, origin)
    assert torch.isfinite(goals).all()
    reward, _ = task.reward(sample["state"], sample["pose"], time, origin,
                            torch.zeros(401, 12), torch.zeros(401, 12))
    torch.testing.assert_close(reward, torch.full_like(reward, .02))
    provenance = backflip.metadata["provenance"]
    assert provenance["nominal_metrics"]["passed"]
    assert provenance["perturbation_successes"] == provenance["perturbation_trials"] == 20
    assert not provenance["pose_writes_after_reset"] and not provenance["external_forces"]


def test_backflip_actual_commands_reproduce_physical_states(backflip):
    p = Physics(state_frame=backflip.state_frame)
    with np.load(ROOT / "data/raw/as2-native-backflip/physics_capture.npz") as raw:
        np.testing.assert_allclose(p.data.qpos, raw["qpos"][0], atol=1e-12)
        for i, action in enumerate(raw["action"]):
            p.step(action)
            np.testing.assert_allclose(p.data.qpos, raw["qpos"][i+1], atol=2e-6, rtol=1e-5)
            np.testing.assert_allclose(p.data.qvel, raw["qvel"][i+1], atol=2e-6, rtol=1e-5)
    assert not p.data.xfrc_applied.any() and not p.data.qfrc_applied.any()


def test_checkpoint_rejects_wrong_or_implicit_old_frame(backflip, tmp_path):
    from as2_motion_imitation.learner import CHECKPOINT_SCHEMA, load_actor
    from as2_motion_imitation.task import REPLAY_SCHEMA, OBS_DIM
    from as2_supermimic.checkpoints import robot_signature

    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "replay_schema": REPLAY_SCHEMA,
        "robot": robot_signature(), "motion_sha256": sha256(backflip.path),
        "obs_dim": OBS_DIM,
    }
    path = tmp_path / "wrong-frame.pt"
    for frame in (None, "forward-axis-v1"):
        if frame:
            checkpoint["state_frame"] = frame
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="state frame"):
            load_actor(path, backflip)


def test_inverted_reference_has_finite_dynamics_only_actor_gradient(backflip):
    from as2_motion_imitation.task import HISTORY, OBS_DIM, TrackingTask, sensors
    from as2_supermimic.supermimic import state as st
    from as2_supermimic.supermimic.train import Actor
    from as2_supermimic.supermimic.world_model import WorldModel
    from as2_supermimic.target.terrain import SCAN_POINTS

    torch.manual_seed(13)
    times = torch.tensor([1.86, 1.92, 1.98, 2.04])
    reference = backflip.sample(times)
    state, pose = reference["state"], reference["pose"]
    actor, model = Actor(OBS_DIM, (32,), 12), WorldModel((32, 32))
    model.requires_grad_(False)
    task, last = TrackingTask(backflip), torch.zeros(4, 12)
    origin = backflip.poses[0].repeat(4, 1)
    action = actor(task.observe(sensors(state, last).repeat(1, HISTORY),
                                state, pose, times, origin))
    nxt, delta = model(state, action, last, torch.zeros(4, 4), torch.zeros(4, SCAN_POINTS))
    reward, _ = task.reward(nxt, st.apply_pose_delta(pose, delta), times + .02,
                            origin, last, last)
    reward.sum().backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 1e-9
    assert all(parameter.grad is None for parameter in model.parameters())
