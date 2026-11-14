"""Search PD schedules for a simulated backflip reference."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.rollout
import numpy as np
from scipy.optimize import differential_evolution

from as2_motion_imitation.physics import DT, Physics, make_model
from as2_motion_imitation.prepare_jump import ease


ROOT = Path(__file__).resolve().parents[2]
PHASES = ("crouch", "launch", "drive", "tuck", "land")
NAMES = tuple(
    f"{phase}_{joint}" for phase in PHASES
    for joint in ("front_thigh", "front_calf", "rear_thigh", "rear_calf")
) + ("launch_s", "drive_s", "tuck_s", "flight_s", "land_s")
BOUNDS = [
    (0.6, 1.85), (-2.70, -1.7), (0.6, 1.85), (-2.70, -1.7),
    (-0.10, 1.85), (-2.70, -0.84), (-0.10, 1.85), (-2.70, -0.84),
    (-0.10, 1.85), (-2.70, -0.84), (-0.10, 1.85), (-2.70, -0.84),
    (0.4, 1.85), (-2.70, -1.9), (0.4, 1.85), (-2.70, -1.9),
    (0.2, 1.7), (-2.60, -1.0), (0.2, 1.7), (-2.60, -1.0),
    (0.04, 0.18), (0.04, 0.18), (0.04, 0.14), (0.12, 0.55), (0.08, 0.35),
]
INITIAL = np.array([
    1.2, -2.5, 1.2, -2.5,
    0.3, -0.9, 1.2, -2.5,
    1.5, -2.6, 0.0, -0.9,
    1.4, -2.6, 1.4, -2.6,
    0.9, -1.8, 0.9, -1.8,
    0.10, 0.12, 0.10, 0.30, 0.12,
])


def targets(parameters, times):
    parameters = np.atleast_2d(parameters)
    batch = len(parameters)
    stand = np.array([0.9, -1.8, 0.9, -1.8])
    knots = np.zeros((batch, 10))
    knots[:, :4] = [0.0, 0.4, 1.2, 1.5]
    knots[:, 4:9] = 1.5 + np.cumsum(parameters[:, 20:25], axis=1)
    knots[:, 9] = knots[:, 8] + 0.8
    phases = parameters[:, :20].reshape(batch, 5, 4)
    poses = np.stack([
        np.broadcast_to(stand, (batch, 4)), np.broadcast_to(stand, (batch, 4)),
        phases[:, 0], phases[:, 0], phases[:, 1], phases[:, 2],
        phases[:, 3], phases[:, 3], phases[:, 4], np.broadcast_to(stand, (batch, 4)),
    ], axis=1)
    index = (times[None, :, None] >= knots[:, None]).sum(-1) - 1
    index = np.clip(index, 0, 8)
    ids = np.arange(batch)[:, None]
    alpha = ease((times[None] - knots[ids, index]) /
                 (knots[ids, index + 1] - knots[ids, index]))
    joints = poses[ids, index] * (1 - alpha[..., None]) + poses[ids, index + 1] * alpha[..., None]
    q = np.zeros((batch, len(times), 12))
    q[..., [0, 6]] = -0.1
    q[..., [3, 9]] = 0.1
    q[..., 1], q[..., 4] = joints[..., 0], joints[..., 0]
    q[..., 2], q[..., 5] = joints[..., 1], joints[..., 1]
    q[..., 7], q[..., 10] = joints[..., 2], joints[..., 2]
    q[..., 8], q[..., 11] = joints[..., 3], joints[..., 3]
    return q


def orientation(quaternions):
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    r00, r20 = 1 - 2 * (y*y + z*z), 2 * (x*z - w*y)
    rotation = np.unwrap(np.arctan2(r20, r00), axis=-1)
    lateral = 1 - 2 * (x*x + z*z)
    upright = 1 - 2 * (x*x + y*y)
    return rotation, lateral, upright


class Search:
    def __init__(self, output, threads=12, duration=4.0, contact_audit=False,
                 strict_limits=False, robust_seeds=()):
        self.output = output
        self.p = Physics(make_model(contact_audit=contact_audit))
        self.model = self.p.model
        self.rollout = mujoco.rollout.Rollout(nthread=threads)
        self.data = [mujoco.MjData(self.model) for _ in range(threads)]
        self.scenario_seeds = (0, *robust_seeds)
        self.initial = np.zeros((len(self.scenario_seeds),
                                 mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
        for i, seed in enumerate(self.scenario_seeds):
            self.p.reset(seed, .01 if i else 0.)
            mujoco.mj_getState(self.model, self.p.data, self.initial[i],
                              mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.duration = duration
        self.times = np.arange(round(duration / DT)) * DT
        self.best = -float("inf")
        self.count = 0
        self.started = time.monotonic()
        self.contact_audit = contact_audit
        self.strict_limits = strict_limits
        audit_sensors = [self.model.sensor(i) for i in range(self.model.nsensor)
                         if self.model.sensor(i).name.startswith("audit_")]
        self.contact_adr = np.array([sensor.adr[0] for sensor in audit_sensors], dtype=int)
        self.foot_sensors = np.array(["_foot_collision" in sensor.name for sensor in audit_sensors])

    def evaluate(self, parameters):
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters[None]
        scenarios = len(self.scenario_seeds)
        initial = np.tile(self.initial, (len(parameters), 1))
        parameters = np.repeat(parameters, scenarios, axis=0)
        commands = targets(parameters, self.times)
        controls = np.repeat(commands[:, :, self.p.act_order], 10, axis=1)
        state, sensors = self.rollout.rollout(self.model, self.data, initial, control=controls)
        # rollout FULLPHYSICS stores time, qpos, qvel. Metrics are search proxies;
        # acceptance separately checks every physical contact and joint limit.
        qpos, qvel = state[..., 1:20], state[..., 20:38]
        rotation, lateral, upright = orientation(qpos[..., 3:7])
        times = (np.arange(state.shape[1]) + 1) * self.model.opt.timestep
        height = qpos[..., 2]
        before_floor = np.cumsum((height < 0.12) & (times > 1.55), axis=1) == 0
        progress = np.max(np.where(before_floor, rotation, 0), axis=1)
        best_rotation = np.max(rotation, axis=1)
        end = qpos[:, -100:]
        end_vel = qvel[:, -100:]
        limit = self.model.jnt_range[self.p.jids]
        violation = np.maximum(limit[:, 0] - qpos[..., self.p.qadr], 0) + np.maximum(
            qpos[..., self.p.qadr] - limit[:, 1], 0)
        score = (
            5 * np.minimum(progress, 2*np.pi)
            - 4 * np.maximum(progress - 2*np.pi, 0)
            + 1.5 * height.max(axis=1)
            - 3 * (1 - lateral.min(axis=1))
            - 2 * np.abs(qpos[..., 1]).max(axis=1)
            - 10 * violation.max(axis=(1, 2))
            - 3 * np.maximum(np.abs(qvel[..., self.p.vadr]).max(axis=(1, 2)) - 30, 0)
            + 2 * upright[:, -100:].mean(axis=1)
            - 8 * np.abs(end[..., 2].mean(axis=1) - initial[:, 3])
            - 0.1 * np.minimum(np.linalg.norm(end_vel[..., :6], axis=-1).mean(axis=1), 20)
            - 2 * np.abs(rotation[:, -1] - np.clip(progress, 0, 2*np.pi))
        )
        if self.strict_limits:
            excess = violation.max(axis=(1, 2))
            score -= 100 * excess + 10 * (excess > 0)
            margin = np.minimum(qpos[..., self.p.qadr] - limit[:, 0],
                                limit[:, 1] - qpos[..., self.p.qadr]).min(axis=(1, 2))
            score -= 20 * np.maximum(.025 - margin, 0)
        contact_duration = np.zeros(len(parameters))
        air_rotation = np.zeros(len(parameters))
        if self.contact_audit:
            contacts = (sensors[..., self.contact_adr] > .5) & (
                sensors[..., self.contact_adr + 1] <= 0)
            nonfoot = contacts[..., ~self.foot_sensors].any(axis=-1)
            airborne = ~contacts.any(axis=-1)
            contact_duration = nonfoot.sum(axis=-1) * self.model.opt.timestep
            for i in range(len(parameters)):
                edges = np.diff(np.r_[False, airborne[i], False].astype(np.int8))
                spans = [(s, e) for s, e in zip(np.flatnonzero(edges == 1),
                                                np.flatnonzero(edges == -1)) if e-s > 20]
                if spans:
                    air_rotation[i] = max(rotation[i, e-1] - rotation[i, s] for s, e in spans)
            rotation_cap = 4.7 if scenarios > 1 else 5.5
            score += 5 * np.clip(air_rotation, 0, rotation_cap) - 15 * contact_duration - 5 * nonfoot.any(axis=-1)
            if scenarios > 1:
                final_feet = contacts[:, -100:, self.foot_sensors].sum(axis=-1)
                joint_error = np.abs(qpos[:, -100:, self.p.qadr] -
                                     initial[:, None, 1+self.p.qadr]).max(axis=-1)
                stable = ((final_feet == 4) & (joint_error < .12)
                          & (np.linalg.norm(end_vel[..., :3], axis=-1) < .05)
                          & (np.linalg.norm(end_vel[..., 3:6], axis=-1) < .15)
                          & (upright[:, -100:] > np.cos(.15)))
                score += 5 * stable.mean(axis=1)
        scores = score.reshape(-1, scenarios)
        worst = np.arange(len(scores)) * scenarios + scores.argmin(axis=1)
        for i in worst:
            value = score[i]
            row = {"trial": self.count, "score": float(value),
                   "scenario_seeds": self.scenario_seeds,
                   "worst_scenario_seed": self.scenario_seeds[i % scenarios],
                   "max_clean_back_rotation": float(progress[i]),
                   "max_back_rotation": float(best_rotation[i]),
                   "final_back_rotation": float(rotation[i, -1]),
                   "peak_height": float(height[i].max()), "final_height": float(height[i, -1]),
                   "min_lateral_alignment": float(lateral[i].min()),
                   "max_joint_speed": float(np.abs(qvel[i, :, self.p.vadr]).max()),
                   "max_joint_violation": float(violation[i].max()),
                   "nonfoot_contact_seconds": float(contact_duration[i]),
                   "airborne_back_rotation": float(air_rotation[i]),
                   "parameters": dict(zip(NAMES, parameters[i].tolist()))}
            with (self.output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.count += scenarios
            if value > self.best:
                self.best = float(value)
                (self.output / "best-params.json").write_text(json.dumps(row["parameters"], indent=2) + "\n")
                (self.output / "best-search-metrics.json").write_text(json.dumps(row, indent=2) + "\n")
                np.savez_compressed(self.output / "best-search-trace.npz",
                                    qpos=qpos[i], qvel=qvel[i], time=times,
                                    commands=commands[i], back_rotation=rotation[i])
                print(json.dumps({k: v for k, v in row.items() if k != "parameters"} |
                                 {"elapsed_s": time.monotonic() - self.started}), flush=True)
        return -scores.min(axis=1)

    def vectorized(self, parameters):
        return self.evaluate(parameters.T)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--contact-audit", action="store_true")
    parser.add_argument("--local-radius", type=float)
    parser.add_argument("--strict-limits", action="store_true")
    parser.add_argument("--robust-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--duration", type=float, default=4.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    x0 = INITIAL
    if args.initial:
        values = json.loads(args.initial.read_text())
        x0 = np.array([values[name] for name in NAMES])
    (output / "protocol.json").write_text(json.dumps({
        "names": NAMES, "bounds": BOUNDS, "initial": x0.tolist(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "physics_modified": False, "forces_or_pose_writes": False,
        "scope": "Bounded reference-controller search, not policy training or acceptance",
    }, indent=2) + "\n")
    search = Search(output, args.threads, contact_audit=args.contact_audit,
                    strict_limits=args.strict_limits, robust_seeds=args.robust_seeds,
                    duration=args.duration)
    try:
        initial_population = "latinhypercube"
        if args.local_radius is not None:
            bounds = np.array(BOUNDS)
            rng = np.random.default_rng(args.seed)
            initial_population = np.clip(
                x0 + rng.normal(size=(args.popsize * len(BOUNDS), len(BOUNDS)))
                * (bounds[:, 1] - bounds[:, 0]) * args.local_radius,
                bounds[:, 0], bounds[:, 1],
            )
        result = differential_evolution(
            search.vectorized, BOUNDS, maxiter=args.generations, popsize=args.popsize,
            seed=args.seed, x0=x0, vectorized=True, updating="deferred",
            polish=False, tol=0, atol=0, init=initial_population,
        )
        (output / "completion.json").write_text(json.dumps({
            "evaluations": search.count, "best_score": search.best,
            "generations": result.nit, "elapsed_s": time.monotonic() - search.started,
        }, indent=2) + "\n")
    finally:
        search.rollout.close()


if __name__ == "__main__":
    main()
