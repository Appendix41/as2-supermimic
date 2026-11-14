import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from as2_motion_imitation.demonstration import RecordedCommands
from as2_motion_imitation.motion import MotionClip
from as2_motion_imitation.physics import Physics
from as2_motion_imitation.task import HISTORY, OBS_DIM, SENSOR_DIM, TrackingTask, sensors

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clip():
    return MotionClip(ROOT / "data/processed/as2_forward_jump.npz")


def test_recorded_controls_are_commands_not_measured_joints(clip):
    commands = RecordedCommands(ROOT / "data/raw/as2-native-forward-jump/physics_capture.npz", clip)
    at = commands.sample(torch.tensor([0.0, 1.84, 1.849, 8.0]))
    torch.testing.assert_close(at[1], at[2])
    assert commands.actions.shape == (400, 12)
    measured = (clip.sample(torch.tensor([1.84]))["joint_pos"] - clip.default_q) / 0.25
    assert (at[1] - measured[0]).abs().max() > 0.1


def test_reject_unpaired_control_capture(clip, tmp_path):
    with np.load(ROOT / "data/raw/as2-native-forward-jump/physics_capture.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays["qpos"][0, 0] += 0.2
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="not paired"):
        RecordedCommands(path, clip)

def test_recorded_command_index_survives_float32_clock_drift():
    commands = RecordedCommands.__new__(RecordedCommands)
    commands.actions = torch.arange(400)[:, None].expand(-1, 12).float()
    frames = torch.arange(400)
    time = frames * 0.02
    for k in range(401):
        expected = (frames + k).clamp(max=399).float()
        torch.testing.assert_close(commands.sample(time)[:, 0], expected, rtol=0, atol=0)
        time = (time + 0.02).clamp(max=8.0)


@pytest.fixture(scope="module")
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from as2_motion_imitation.gpu_physics import GpuPhysics

    return GpuPhysics(4)


def test_gpu_state_matches_cpu_at_moving_reference(gpu):
    p = Physics()
    with np.load(ROOT / "data/processed/as2_forward_jump.npz") as f:
        qpos, qvel = f["qpos"][[20, 80, 105, 120]], f["qvel"][[20, 80, 105, 120]]
    gpu.reset_states(gpu.all_ids, torch.tensor(qpos, device="cuda"),
                     torch.tensor(qvel, device="cuda"))
    state, pose = gpu.state()
    for i in range(4):
        p.data.qpos[:], p.data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(p.model, p.data)
        s, xy = p.state()
        np.testing.assert_allclose(state[i].cpu(), s, atol=3e-5, rtol=1e-4)
        np.testing.assert_allclose(pose[i].cpu(), xy, atol=3e-5, rtol=1e-4)


def test_gpu_partial_reset_does_not_reset_other_worlds(gpu):
    before_pos, before_vel = gpu.data.qpos.clone(), gpu.data.qvel.clone()
    ids = torch.tensor([1], device="cuda")
    gpu.reset_states(ids, gpu.standing_qpos[ids], gpu.standing_qvel[ids])
    other = torch.tensor([0, 2, 3], device="cuda")
    torch.testing.assert_close(gpu.data.qpos[other], before_pos[other], rtol=0, atol=0)
    torch.testing.assert_close(gpu.data.qvel[other], before_vel[other], rtol=0, atol=0)


def test_gpu_physics_keeps_actuator_and_collision_parameters(gpu):
    p = Physics()
    for field in ("body_mass", "actuator_gainprm", "actuator_biasprm",
                  "actuator_forcerange", "jnt_range", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(gpu.model, field), getattr(p.model, field))
    assert not gpu.data.xfrc_applied.any()
    assert not gpu.data.qfrc_applied.any()


def test_gpu_reference_reset_pairs_state_and_phase():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from as2_motion_imitation.gpu_physics import GpuTrackingEnv

    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz", "cuda:0")
    env = GpuTrackingEnv(clip, 8, seed=13)
    state, pose = env.read()
    reference = clip.sample(env.clock.time)
    torch.testing.assert_close(state[:, 0], reference["state"][:, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pose, reference["pose"], atol=1e-5, rtol=1e-5)
    action = torch.zeros(8, 12, device="cuda")
    obs, reward, failed, ended, _ = env.step(action)
    state, pose = env.read()
    expected, _ = TrackingTask(clip).reward(
        state, pose, env.clock.time, env.clock.origin, action, action,
    )
    torch.testing.assert_close(reward, expected)
    assert obs.shape == (8, OBS_DIM)
    assert torch.isfinite(obs).all()


def test_actor_checkpoint_evaluation_has_no_driver_dependency(tmp_path, clip):
    from dataclasses import asdict
    from as2_motion_imitation.learner import CHECKPOINT_SCHEMA, TrainCfg, load_actor
    from as2_motion_imitation import SCHEMA_VERSION
    from as2_motion_imitation.motion import sha256
    from as2_motion_imitation.task import REPLAY_SCHEMA
    from as2_supermimic.checkpoints import robot_signature
    from as2_supermimic.supermimic.train import Actor

    cfg = TrainCfg(actor_hidden=(32,), recorded_commands="does-not-exist.npz")
    actor = Actor(OBS_DIM, (32,), 12)
    path = tmp_path / "actor.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "motion_schema": SCHEMA_VERSION,
                "replay_schema": REPLAY_SCHEMA, "robot": robot_signature(),
                "motion_sha256": sha256(clip.path), "obs_dim": OBS_DIM,
                "cfg": asdict(cfg), "actor": actor.state_dict()}, path)
    loaded, _ = load_actor(path, clip)
    x = torch.randn(2, OBS_DIM)
    torch.testing.assert_close(loaded(x), actor(x))
