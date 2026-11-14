"""Compare CPU and GPU rollouts with the same actions and time GPU stepping."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from as2_motion_imitation.gpu_physics import GpuPhysics
from as2_motion_imitation.physics import DT, Physics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commands", type=Path,
                        default=ROOT / "data/raw/as2-native-forward-jump/physics_capture.npz")
    parser.add_argument("--state-frame", choices=("forward-axis-v1", "lateral-axis-v2"),
                        default="forward-axis-v1")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    gpu = GpuPhysics(args.num_envs, state_frame=args.state_frame)
    cpu = Physics(state_frame=args.state_frame)
    print("GPU initialized", flush=True)
    initial, initial_pose = gpu.state()
    expected, pose = cpu.state()
    state_initial_error = (initial - torch.tensor(expected, device="cuda")).abs().max().item()
    with np.load(args.commands) as f:
        commands = torch.tensor(f["action"], device="cuda", dtype=torch.float32)
    errors, cpu_states, gpu_states, cpu_poses, gpu_poses = [], [], [], [], []
    torch.cuda.synchronize()
    run_start = time.monotonic()
    for i, a in enumerate(commands):
        gpu.step(a.expand(args.num_envs, 12))
        cpu.step(a.cpu().numpy())
        s, p = gpu.state()
        c, cp = cpu.state()
        cpu_states.append(c)
        gpu_states.append(s[0].cpu().numpy())
        cpu_poses.append(cp)
        gpu_poses.append(p[0].cpu().numpy())
        errors.append(float((s - torch.tensor(c, device="cuda")).abs().max()))
    torch.cuda.synchronize()
    seconds = time.monotonic() - run_start
    capacity = gpu.capacity_status()
    cs, gs = np.array(cpu_states), np.array(gpu_states)
    cp, gp = np.array(cpu_poses), np.array(gpu_poses)
    result = {
        "num_envs": args.num_envs, "control_steps": len(commands),
        "state_frame": args.state_frame,
        "commands": str(args.commands.resolve()),
        "duration_seconds": len(commands) * DT,
        "initial_state_max_abs_error": state_initial_error,
        "full_rollout_state_max_abs_error": max(errors),
        "height_max_abs_error_m": float(np.abs(cs[:, 0] - gs[:, 0]).max()),
        "joint_max_abs_error_rad": float(np.abs(cs[:, 10:22] - gs[:, 10:22]).max()),
        "pose_max_abs_error": float(np.abs(cp - gp).max()),
        "final_cpu_pose": cp[-1].tolist(), "final_gpu_pose": gp[-1].tolist(),
        "paired_rollout_wall_seconds": seconds,
        "paired_transitions_per_second": args.num_envs * len(commands) / seconds,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "total_wall_seconds": time.monotonic() - start,
        "capacity": capacity,
    }
    np.savez_compressed(output / "paired-rollout.npz", cpu_state=cs, gpu_state=gs,
                        cpu_pose=cp, gpu_pose=gp)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["height_max_abs_error_m"] > 0.02 or result["joint_max_abs_error_rad"] > 0.10:
        raise RuntimeError("CPU/GPU rollout differs too much for this reference")


if __name__ == "__main__":
    main()
