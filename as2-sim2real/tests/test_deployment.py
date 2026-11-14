"""Non-actuating deployment checks; no robot SDK import or network access."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from as2_motion_imitation.deployment import runtime, telemetry
from as2_motion_imitation.deployment.runtime import InferenceBundle, init_history, push_history


ROOT = Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    import torch
    from as2_motion_imitation.motion import MotionClip
    from as2_motion_imitation.task import OFFSETS
    from as2_supermimic.supermimic.train import Actor
    directory = tmp_path_factory.mktemp("synthetic-bundle")
    torch.manual_seed(71)
    actor = Actor(runtime.OBS_DIM, (512, 256, 128), 12)
    state = actor.state_dict()
    weights = {"mean": state["norm.mean"].numpy(), "var": state["norm.var"].numpy()}
    for i, layer in enumerate((0, 2, 4, 6)):
        weights[f"weight_{i}"] = state[f"net.{layer}.weight"].numpy()
        weights[f"bias_{i}"] = state[f"net.{layer}.bias"].numpy()
    np.savez(directory / "actor.npz", **weights)
    clip = MotionClip(ROOT / "data/processed/as2_forward_jump.npz")
    times = torch.tensor([i * 0.02 for i in range(400)])
    refs = clip.sample(times[:, None] + torch.tensor(OFFSETS))
    np.savez(directory / "reference.npz", states=refs["state"][..., :34].numpy(),
             phases=(times / clip.duration).numpy(), default_q=clip.default_q.numpy())
    manifest = {
        "schema": runtime.SCHEMA, "hardware_execution_enabled": False,
        "state_frame": "forward-axis-v1", "obs_dim": runtime.OBS_DIM, "action_dim": 12,
        "history": 5, "required_pose": False, "steps": 400, "dt": 0.02,
        "files": {name: runtime.sha256(directory / name)
                  for name in ("actor.npz", "reference.npz")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    result = InferenceBundle(directory)
    result.test_actor = actor.eval()
    result.test_directory = directory
    return result


def test_history_order_and_reset():
    first = np.arange(42, dtype=np.float32)
    history = init_history(first)
    np.testing.assert_array_equal(history.reshape(5, 42), np.tile(first, (5, 1)))
    new = first + 10
    updated = push_history(history, new)
    np.testing.assert_array_equal(updated[:-42], history[42:])
    np.testing.assert_array_equal(updated[-42:], new)
    np.testing.assert_array_equal(init_history(first), history)


@pytest.mark.parametrize("step", [-1, 400, 401, 0.0, True])
def test_one_shot_ticks_reject_wrapping(bundle, step):
    with pytest.raises(ValueError):
        bundle.observe(np.zeros(210), step)


def test_pose_not_accepted_or_stored(bundle):
    import inspect
    assert list(inspect.signature(bundle.observe).parameters) == ["history", "step"]
    assert not hasattr(bundle, "reference_poses")
    assert bundle.manifest["required_pose"] is False


def test_sensor_contract_uses_applied_action(bundle):
    last = np.linspace(-1, 1, 12, dtype=np.float32)
    sensor = bundle.sensor([1, 2, 3], [0, 0, -1], bundle.default_q, np.zeros(12), last)
    np.testing.assert_array_equal(sensor[:6], [1, 2, 3, 0, 0, -1])
    np.testing.assert_array_equal(sensor[6:30], np.zeros(24))
    np.testing.assert_array_equal(sensor[30:], last)
    with pytest.raises(ValueError):
        bundle.sensor([0] * 3, [0] * 3, bundle.default_q, np.zeros(12), last)


def test_quaternion_yaw_does_not_change_projected_gravity():
    for yaw in np.linspace(-np.pi, np.pi, 31):
        np.testing.assert_allclose(
            runtime.projected_gravity([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
            [0, 0, -1], atol=1e-6)
    with pytest.raises(ValueError):
        runtime.projected_gravity([0, 0, 0, 0])


def test_clamping_and_frozen_parameters(bundle):
    np.testing.assert_allclose(bundle.target(np.full(12, 100)), bundle.default_q + 1)
    np.testing.assert_allclose(bundle.target(np.full(12, -100)), bundle.default_q - 1)
    with pytest.raises(ValueError):
        bundle.mean[0] = 1
    with pytest.raises(ValueError):
        bundle.act(np.full(runtime.OBS_DIM, np.nan))


def test_all_400_reference_ticks_and_numpy_actor_match_torch(bundle):
    import torch
    from as2_motion_imitation.motion import MotionClip
    from as2_motion_imitation.task import TrackingTask
    task = TrackingTask(MotionClip(ROOT / "data/processed/as2_forward_jump.npz"))
    rng = np.random.default_rng(42)
    with torch.no_grad():
        for step in range(400):
            history = rng.normal(size=210).astype(np.float32)
            obs = bundle.observe(history, step)
            expected = task.observe(torch.from_numpy(history)[None], None, None,
                                    torch.tensor([step * 0.02]), None)
            np.testing.assert_allclose(obs, expected[0].numpy(), atol=1e-6)
            action = bundle.test_actor(expected).clamp(-4, 4)[0].numpy()
            np.testing.assert_allclose(bundle.act(obs), action, atol=2e-6)


def test_hash_tampering_rejected(tmp_path, bundle):
    manifest = bundle.manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "actor.npz").write_bytes(b"not the exported model")
    with pytest.raises(ValueError, match="Hash mismatch"):
        InferenceBundle(tmp_path)


def test_read_only_module_import_surface():
    for name in ("runtime", "offline", "telemetry", "readiness",
                 "bench_control", "control_selftest", "feedback_crc"):
        path = ROOT / f"src/as2_motion_imitation/deployment/{name}.py"
        tree = ast.parse(path.read_text())
        imported = [
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ] + [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        assert not any(x.startswith(("mujoco", "torch", "cyclonedds.pub")) for x in imported)
        assert not any(".sport." in x or ".rpc." in x or ".core.channel" in x for x in imported)
        called = [n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        assert "DataWriter" not in called
        assert "Publisher" not in called


def test_bad_interface_rejected_without_shell():
    with pytest.raises(ValueError):
        telemetry.network_preflight("end0;exit", "192.168.123.0/24")


def test_network_down_is_blocked_before_sdk(monkeypatch, tmp_path):
    import sys
    out = tmp_path / "blocked"
    monkeypatch.setattr(sys, "argv", ["telemetry", "--output", str(out)])
    monkeypatch.setattr(telemetry, "network_preflight", lambda *a: {"passed": False})
    monkeypatch.setattr(telemetry, "collect", lambda *a: pytest.fail("must not initialize DDS"))
    with pytest.raises(SystemExit) as error:
        telemetry.main()
    assert error.value.code == 2
    report = json.loads((out / "summary.json").read_text())
    assert report["status"] == "blocked_network"
    assert report["robot_command_writers_created"] == 0


def test_raw_motor_slots_not_labeled_as_policy_order():
    motor = SimpleNamespace(mode=1, q=0.1, dq=0.2, tau_est=0.3, motorstate=0)
    imu = SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0] * 3,
                          accelerometer=[0] * 3, rpy=[0] * 3)
    msg = SimpleNamespace(sample_info=SimpleNamespace(source_timestamp=123), tick=456,
                          mode_pr=0, mode_machine=1, version=[0, 0], imu_state=imu,
                          motor_state=[motor] * 35, wireless_remote=bytes(40))
    row = telemetry.record_message("rt/lowstate", msg, 789)
    assert len(row["motor_slots"]) == 35
    assert row["motor_slots"][-1]["slot"] == 34
    assert "joint_names" not in row
    assert row["dds_source_timestamp_ns"] == 123


def test_documented_index_map_matches_names_without_hardware_approval():
    from as2_supermimic.robot.morphology import JOINT_NAMES
    contract = json.loads((ROOT / "docs/evidence/AS2_INTERFACE_CONTRACT.json").read_text())
    assert contract["policy_joint_order"] == list(JOINT_NAMES)
    for name, slot in zip(contract["policy_joint_order"],
                          contract["documented_sdk_slot_for_each_policy_joint"]):
        leg, kind, _ = name.split("_")
        expected = leg + "_" + ("BODY" if kind == "hip" else kind.upper())
        assert contract["documented_sdk_joint_order"][slot] == expected
    assert not contract["documented_mapping_verified_on_this_robot"]
    assert not contract["hardware_execution_enabled"]
