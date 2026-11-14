import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from as2_motion_imitation.learner import Imagination, TrainCfg, load_actor
from as2_motion_imitation.motion import MotionClip, MotionClock, sha256, slerp
from as2_motion_imitation.physics import DT, Physics
from as2_motion_imitation.task import (
    CRITIC_DIM, HISTORY, OBS_DIM, SENSOR_DIM, TrackingEnv, TrackingTask, sensors,
)
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.buffer import ReplayBuffer
from as2_supermimic.supermimic.train import Actor, Critic
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.terrain import SCAN_POINTS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clip():
    return MotionClip(ROOT / "data/processed/as2_crouch.npz")


def test_endpoints_interpolation_and_batch(clip):
    times = torch.tensor([0, 0.01, clip.duration, clip.duration + 1])
    out = clip.sample(times)
    torch.testing.assert_close(out["joint_pos"][0], clip.arrays["joint_pos"][0])
    torch.testing.assert_close(out["joint_pos"][1], clip.arrays["joint_pos"][:2].mean(0))
    torch.testing.assert_close(out["joint_pos"][-1], clip.arrays["joint_pos"][-1])
    assert out["ended"].tolist() == [False, False, True, True]
    assert clip.sample(torch.zeros(2, 4))["state"].shape == (2, 4, 70)
    assert clip.sample(0.0)["state"].shape == (70,)
    with pytest.raises(ValueError, match="nonfinite"):
        clip.sample([float("nan")])


def test_slerp_shortest_arc_and_sign():
    a = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])
    b = torch.tensor([[-1., 0, 0, 0], [0., 0, 0, 1.]])
    q = slerp(a, b, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(q[0], a[0])
    torch.testing.assert_close(q[1], torch.tensor([2**-0.5, 0, 0, 2**-0.5]))
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(2))


@pytest.mark.parametrize("field,value,error", [
    ("units", {"length": "cm", "angle": "rad", "time": "s"}, "units"),
    ("quaternion", "xyzw", "convention"),
    ("sha256", "bad", "SHA256"),
    ("schema", "old", "schema"),
])
def test_metadata_rejection(clip, tmp_path, field, value, error):
    path = tmp_path / "bad.npz"
    path.write_bytes(clip.path.read_bytes())
    meta = copy.deepcopy(clip.metadata)
    meta[field] = value
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=error):
        MotionClip(path)


def test_joint_name_mapping_and_invalid_fk(clip, tmp_path):
    with np.load(clip.path) as f:
        arrays = {k: f[k] for k in f.files}
    perm = np.arange(11, -1, -1)
    for key in ("joint_pos", "joint_vel"):
        arrays[key] = arrays[key][:, perm]
    arrays["joint_names"] = arrays["joint_names"][perm]
    path = tmp_path / "reverse.npz"
    np.savez_compressed(path, **arrays)
    meta = copy.deepcopy(clip.metadata)
    meta["sha256"] = sha256(path)
    path.with_suffix(".json").write_text(json.dumps(meta))
    mapped = MotionClip(path)
    torch.testing.assert_close(mapped.states, clip.states)
    arrays["body_pos_w"][0, 4, 2] += 0.1
    np.savez_compressed(path, **arrays)
    meta["sha256"] = sha256(path)
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="FK"):
        MotionClip(path)


def test_async_clock_no_wrap(clip):
    clock = MotionClock(clip, 3)
    clock.reset(torch.tensor([1, 2]), torch.tensor([clip.duration - DT, 2.0]))
    end = clock.advance(DT)
    assert end.tolist() == [False, True, False]
    clock.reset(torch.tensor([0]))
    assert clock.time[1] == clip.duration
    assert clock.packed().shape == (3, 5)
    with pytest.raises(ValueError):
        clock.reset(torch.tensor([0]), clip.duration)


def test_physics_reference_and_shared_contract(clip):
    env = TrackingEnv(clip, 2)
    for _ in range(3):
        a = torch.randn(2, 12) * 0.1
        previous = env.last_action.clone()
        obs, reward, failed, ended, _ = env.step(a)
        s, pose = env.read()
        expected = env.task.observe(env.history, s, pose, env.clock.time, env.clock.origin)
        torch.testing.assert_close(obs, expected)
        pure_reward, _ = TrackingTask(clip).reward(s, pose, env.clock.time, env.clock.origin, a, previous)
        torch.testing.assert_close(reward, pure_reward)
        assert not failed.any() and not ended.any()
        assert obs.shape == (2, OBS_DIM)
    # Check the reference feature conversion against an independent physical snapshot.
    p = Physics()
    s, pose = p.state()
    torch.testing.assert_close(clip.states[0], torch.tensor(s), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(clip.poses[0], torch.tensor(pose), atol=1e-4, rtol=1e-4)


def test_replay_masks_reset_windows():
    b = ReplayBuffer(1, 16, OBS_DIM, "cpu", gen_dim=5)
    for i in range(12):
        b.add(torch.zeros(1, 70), torch.zeros(1, 3), torch.zeros(1, 12),
              torch.zeros(1, 12), torch.zeros(1, OBS_DIM), torch.zeros(1, 5),
              torch.tensor([i == 5]), torch.zeros(1, SCAN_POINTS))
    env, idx, valid = b.sample_windows(100, 8)
    done = b.done[env[:, None], idx[:, :8]]
    torch.testing.assert_close(valid, torch.cumprod(1 - done, 1))
    assert (valid[:, -1] == 0).any()


def test_dynamics_only_reward_gradient(clip):
    torch.manual_seed(13)
    wm = WorldModel((32, 32))
    wm.requires_grad_(False)
    actor = Actor(OBS_DIM, (32,), 12)
    state = clip.states[140:144]
    pose = clip.poses[140:144]
    time = torch.tensor([2.8, 2.82, 2.84, 2.86])
    origin = clip.poses[0].repeat(4, 1)
    last = torch.zeros(4, 12)
    history = sensors(state, last).repeat(1, HISTORY)
    task = TrackingTask(clip)
    action = actor(task.observe(history, state, pose, time, origin))
    nxt, delta = wm(state, action, last, torch.zeros(4, 4), torch.zeros(4, SCAN_POINTS))
    # Zero action inputs explicitly eliminate any direct regularizer path.
    reward, _ = task.reward(nxt, st.apply_pose_delta(pose, delta), time + DT, origin, last, last)
    reward.sum().backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 1e-9
    assert all(p.grad is None for p in wm.parameters())


def test_imagination_clip_end_and_td_lambda(clip):
    cfg = TrainCfg(horizon=3)
    task = TrackingTask(clip)
    imag = Imagination(task, cfg)
    s = clip.states[-2:].clone()
    seed = {
        "s": s, "pose": clip.poses[-2:].clone(), "a_prev": torch.zeros(2, 12),
        "history": sensors(s, torch.zeros(2, 12)).repeat(1, HISTORY),
        "time": torch.full((2,), clip.duration - DT / 2),
        "origin": clip.poses[0].repeat(2, 1), "xi": torch.zeros(2, 4),
    }
    wm, actor, critic = WorldModel((32,)), Actor(OBS_DIM, (32,), 12), Critic(CRITIC_DIM, (32,))
    _, data = imag.rollout(wm, actor, critic, seed)
    assert (data["surv"][:, 1:] == 0).all()
    targets = imag.gae_targets(critic, data)
    torch.testing.assert_close(targets[:, 0], data["rewards"][:, 0])


def test_static_second_clip_and_checkpoint_rejection(clip, tmp_path):
    stand = MotionClip(ROOT / "data/processed/as2_stand.npz")
    torch.testing.assert_close(stand.sample([0.0])["state"], stand.sample([1.0])["state"])
    path = tmp_path / "old.pt"
    torch.save({"supermimic": True, "actor": {}}, path)
    with pytest.raises(ValueError, match="schema"):
        load_actor(path, clip)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_physics_and_cuda_tracking_agree(clip):
    gpu = MotionClip(clip.path, "cuda:0")
    env = TrackingEnv(clip, 2)
    env.step(torch.randn(2, 12) * 0.1)
    s, pose = env.read()
    time, origin = env.clock.time, env.clock.origin
    args = (env.history, s, pose, time, origin)
    cpu_obs = env.task.observe(*args)
    gpu_obs = TrackingTask(gpu).observe(*(v.cuda() for v in args))
    torch.testing.assert_close(cpu_obs, gpu_obs.cpu(), atol=1e-5, rtol=1e-5)
    rargs = (s, pose, time, origin, env.last_action, torch.zeros(2, 12))
    cpu_reward, _ = env.task.reward(*rargs)
    gpu_reward, _ = TrackingTask(gpu).reward(*(v.cuda() for v in rargs))
    torch.testing.assert_close(cpu_reward, gpu_reward.cpu(), atol=1e-6, rtol=1e-5)
