"""Recorded-input parity and bounded CPU timing. Never imports a robot SDK."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from .runtime import InferenceBundle, init_history, push_history, sha256


def verify(bundle, fixture_path):
    if sha256(fixture_path) != bundle.manifest["files"]["parity_cases.npz"]:
        raise ValueError("Parity fixture hash mismatch")
    with np.load(fixture_path, allow_pickle=False) as data:
        cases = {key: data[key].copy() for key in data.files}
    obs_max = action_max = target_max = 0.0
    history = None
    for i, step in enumerate(cases["step"]):
        sensor = cases["sensor"][i]
        history = init_history(sensor) if step == 0 else push_history(history, sensor)
        obs = bundle.observe(history, int(step))
        action = bundle.act(obs)
        target = bundle.target(action)
        obs_max = max(obs_max, float(np.max(np.abs(obs - cases["observation"][i]))))
        action_max = max(action_max, float(np.max(np.abs(action - cases["action"][i]))))
        target_max = max(target_max, float(np.max(np.abs(target - cases["target"][i]))))
    passed = obs_max <= 5e-6 and action_max <= 3e-5 and target_max <= 1e-5
    return {
        "passed": passed, "samples": len(cases["step"]),
        "max_abs_observation_error": obs_max, "max_abs_action_error": action_max,
        "max_abs_target_error_rad": target_max,
        "limits": {"observation": 5e-6, "action": 3e-5, "target_rad": 1e-5},
        "scope": "Teacher-forced recorded inputs, not a closed-loop hardware test",
    }, cases


def benchmark(bundle, cases, cycles):
    elapsed, lateness = [], []
    history = init_history(cases["sensor"][0])
    for i in range(100):
        bundle.act(cases["observation"][i % len(cases["step"])] )
    deadline = time.perf_counter()
    for i in range(cycles):
        now = time.perf_counter()
        if now < deadline:
            time.sleep(deadline - now)
        start = time.perf_counter()
        lateness.append(max(0.0, start - deadline) * 1000)
        k = i % len(cases["step"])
        step = int(cases["step"][k])
        history = (init_history(cases["sensor"][k]) if step == 0
                   else push_history(history, cases["sensor"][k]))
        obs = bundle.observe(history, step)
        target = bundle.target(bundle.act(obs))
        if not np.isfinite(target).all():
            raise FloatingPointError("Nonfinite target")
        elapsed.append((time.perf_counter() - start) * 1000)
        deadline += bundle.dt
    ms = np.asarray(elapsed)
    late = np.asarray(lateness)
    return {
        "cycles": cycles, "requested_period_ms": bundle.dt * 1000,
        "compute_ms": {str(p): float(np.percentile(ms, p)) for p in (50, 95, 99, 100)},
        "wake_lateness_ms": {str(p): float(np.percentile(late, p)) for p in (50, 95, 99, 100)},
        "compute_over_20ms": int(np.sum(ms > 20)),
        "completion_deadline_misses": int(np.sum(late + ms > 20)),
        "scope": "CPU recorded sensor/history + goal + normalizer + actor + target; no DDS or safety loop",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=400)
    args = parser.parse_args()
    if not 0 <= args.cycles <= 1500:
        parser.error("cycles must be between 0 and 1500")
    bundle = InferenceBundle(args.bundle)
    parity, cases = verify(bundle, args.bundle / "parity_cases.npz")
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "architecture": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
        "backend": "numpy-fp32-cpu", "robot_commands_sent": 0,
        "manifest_sha256": sha256(args.bundle / "manifest.json"), "parity": parity,
    }
    if parity["passed"] and args.cycles:
        report["timing"] = benchmark(bundle, cases, args.cycles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not parity["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
