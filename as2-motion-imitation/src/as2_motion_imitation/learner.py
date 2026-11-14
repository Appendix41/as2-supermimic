"""Train a reference-conditioned policy with world-model rollouts.

Network, normalization and replay utilities live in as2_supermimic.
This module handles clip timing, reference goals and tracking rewards.
"""

import argparse
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from as2_motion_imitation import SCHEMA_VERSION
from as2_motion_imitation.motion import MotionClip, sha256
from as2_motion_imitation.physics import DT
from as2_motion_imitation.task import (
    CRITIC_DIM, HISTORY, OBS_DIM, REPLAY_SCHEMA, SENSOR_DIM,
    TrackingEnv, TrackingTask, push_history,
)
from as2_supermimic.checkpoints import robot_signature
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.buffer import ReplayBuffer
from as2_supermimic.supermimic.imagine import Imagination as BaselineImagination
from as2_supermimic.supermimic.train import Actor, Critic, clip_grads
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.terrain import SCAN_POINTS

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SCHEMA = "as2-motion-checkpoint-v1"


@dataclass
class TrainCfg:
    motion: str = "data/processed/as2_crouch.npz"
    output: str = "outputs/training/crouch-probe"
    seed: int = 7
    device: str = "cuda:0"
    num_envs: int = 8
    iterations: int = 150
    warmup_iterations: int = 40
    initial_collect: int = 420
    collect_steps: int = 4
    buffer_len: int = 600
    wm_window: int = 8
    horizon: int = 32
    wm_batch: int = 64
    imag_batch: int = 32
    wm_updates: int = 2
    wm_hidden: tuple = (1024, 1024, 1024)
    actor_hidden: tuple = (512, 256, 128)
    critic_hidden: tuple = (512, 256, 128)
    critic_epochs: int = 2
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.12
    reference_driver_fraction: float = 0.5
    log_interval: int = 10
    physics_backend: str = "cpu"
    reference_reset: bool = False
    recorded_commands: str = ""
    noise_correlation: float = 0.0
    phase_prioritized_fraction: float = 0.0
    phase_window: tuple = (1.5, 3.4)
    phase_windows: tuple = ()
    actor_lr_scale: float = 1.0
    actor_ramp_iterations: int = 0
    save_interval: int = 0
    model_audit_interval: int = 0
    model_audit_phases: tuple = (0.4, 1.6, 1.84, 2.1, 2.32, 2.5)
    model_gate_relative_error: float = 1.0
    model_gate_cosine: float = 0.6
    model_gate_rotation: bool = False


class Imagination:
    gae_targets = BaselineImagination.gae_targets

    def __init__(self, task, cfg):
        self.task, self.cfg = task, cfg

    def rollout(self, wm, actor, critic, seed):
        s, pose, a_prev, history = seed["s"], seed["pose"], seed["a_prev"], seed["history"]
        t, origin, xi = seed["time"], seed["origin"], seed["xi"]
        survival = torch.ones_like(t)
        total = torch.zeros_like(t)
        rewards, survivals, inputs = [], [survival], []
        for k in range(self.cfg.horizon):
            inputs.append(self.task.critic_input(s, pose, t, origin, xi))
            obs = self.task.observe(history, s, pose, t, origin)
            action = actor(obs).clamp(-4, 4)
            next_s, delta = wm(s, action, a_prev, xi,
                               torch.zeros(s.shape[0], SCAN_POINTS, device=s.device))
            next_pose = st.apply_pose_delta(pose, delta)
            next_t = (t + DT).clamp(max=self.task.clip.duration)
            reward, _ = self.task.reward(next_s, next_pose, next_t, origin, action, a_prev)
            active = (t < self.task.clip.duration - 1e-5).float()
            reward = reward * active
            total = total + self.cfg.gamma**k * survival * reward
            continuation = (1 - self.task.failure(next_s, next_t, soft=True))
            # End the rollout when the reference clip ends.
            continuation = continuation * (next_t < self.task.clip.duration - 1e-5).float()
            survival = survival * continuation
            rewards.append(reward)
            survivals.append(survival)
            history = push_history(history, next_s, action)
            s, pose, t, a_prev = next_s, next_pose, next_t, action
        inputs.append(self.task.critic_input(s, pose, t, origin, xi))
        objective = total + self.cfg.gamma**self.cfg.horizon * survival * critic(inputs[-1]).squeeze(-1)
        return objective, {
            "rewards": torch.stack(rewards, 1), "surv": torch.stack(survivals, 1),
            "critic_in": torch.stack(inputs, 1),
        }


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(2)
        self.device = cfg.device
        if cfg.buffer_len <= cfg.wm_window + 1 or cfg.initial_collect <= cfg.wm_window + 1:
            raise ValueError("replay and initial collection must exceed the model window")
        if not 0 <= cfg.reference_driver_fraction <= 1 or not 0 <= cfg.noise_correlation < 1:
            raise ValueError("invalid driver fraction or noise correlation")
        if cfg.physics_backend not in ("cpu", "mjwarp"):
            raise ValueError("physics_backend must be cpu or mjwarp")
        if cfg.num_envs > 64 and cfg.physics_backend == "cpu":
            raise ValueError("large batches require the explicit mjwarp backend")
        self.clip = MotionClip(ROOT / cfg.motion, cfg.device)
        if cfg.physics_backend == "mjwarp":
            from as2_motion_imitation.gpu_physics import GpuTrackingEnv

            self.env = GpuTrackingEnv(self.clip, cfg.num_envs, cfg.seed, cfg.reference_reset)
        else:
            sim_clip = MotionClip(ROOT / cfg.motion)
            self.env = TrackingEnv(sim_clip, cfg.num_envs, cfg.seed)
        self.commands = None
        if cfg.recorded_commands:
            from as2_motion_imitation.demonstration import RecordedCommands

            self.commands = RecordedCommands(ROOT / cfg.recorded_commands, self.clip)
        self.task = TrackingTask(self.clip)
        self.wm = WorldModel(tuple(cfg.wm_hidden)).to(cfg.device)
        self.actor = Actor(OBS_DIM, tuple(cfg.actor_hidden), 12).to(cfg.device)
        self.critic = Critic(CRITIC_DIM, tuple(cfg.critic_hidden)).to(cfg.device)
        torch.nn.init.zeros_(self.actor.net[-1].weight)
        torch.nn.init.zeros_(self.actor.net[-1].bias)
        torch.nn.init.zeros_(self.critic.net[-1].weight)
        torch.nn.init.zeros_(self.critic.net[-1].bias)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr * cfg.actor_lr_scale,
                                          betas=(0.9, 0.95))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, OBS_DIM, cfg.device, gen_dim=5)
        self.xi = self.env.xi.to(cfg.device)
        self.imag = Imagination(self.task, cfg)
        self.output = (ROOT / cfg.output).resolve()
        self.output.relative_to(ROOT / "outputs")
        self.output.mkdir(parents=True, exist_ok=False)
        self.output.joinpath("config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        self.real_falls = 0
        self.collected_transitions = 0
        self.actor_updates = 0
        self.noise = torch.zeros(cfg.num_envs, 12, device=cfg.device)
        self.model_gate_passed = cfg.model_audit_interval == 0
        self.response_audit = None
        if cfg.model_audit_interval:
            from as2_motion_imitation.model_audit import ResponseAudit

            self.response_audit = ResponseAudit(
                self.clip, self.commands, self.output, cfg.model_audit_phases,
            )
        self.env.reset(torch.arange(cfg.num_envs), stagger=True)

    @torch.no_grad()
    def collect(self, steps):
        losses = []
        for _ in range(steps):
            env = self.env
            s, pose = env.read()
            s, pose = s.to(self.device), pose.to(self.device)
            obs = env.observe().to(self.device)
            gen = env.clock.packed().to(self.device)
            prev = env.last_action.to(self.device).clone()
            action = self.actor(obs).clamp(-4, 4)
            driven = int(env.n * self.cfg.reference_driver_fraction)
            if driven:
                # Reference-PD actions are used only for training data collection.
                t = env.clock.time[:driven].to(self.device)
                if self.commands is not None:
                    action[:driven] = self.commands.sample(t)
                else:
                    target = self.clip.sample(t)["joint_pos"]
                    action[:driven] = (target - self.clip.default_q) / 0.25
            rho = self.cfg.noise_correlation
            self.noise = rho * self.noise + (1 - rho**2)**0.5 * torch.randn_like(action)
            action += self.noise * self.cfg.action_noise
            action.clamp_(-4, 4)
            _, reward, failed, ended, _ = env.step(
                action if self.cfg.physics_backend == "mjwarp" else action.cpu()
            )
            nxt, nxt_pose = env.read()
            done = failed | ended
            nxt, nxt_pose = nxt.to(self.device), nxt_pose.to(self.device)
            relief = torch.zeros(env.n, SCAN_POINTS, device=self.device)
            self.buf.add(s, pose, action, prev, obs, gen, done.to(self.device), relief)
            # Record terminal-state statistics before resetting.
            self.wm.update_stats(s, action, prev, self.xi, relief, nxt, st.pose_delta(pose, nxt_pose))
            self.actor.norm.update(obs)
            self.real_falls += int(failed.sum())
            self.collected_transitions += env.n
            if not torch.isfinite(nxt).all():
                raise FloatingPointError("nonfinite physics state, stopping this run")
            if done.any():
                ids = done.nonzero().flatten()
                self.noise[ids.to(self.device)] = 0
                env.reset(ids, stagger=True)
            losses.append(reward.mean().item() / DT)
        return sum(losses) / len(losses)

    def fit_world_model(self):
        losses = []
        for _ in range(self.cfg.wm_updates):
            b, k = self.buf, self.cfg.wm_window
            batch = self.cfg.wm_batch
            env, idx, valid = b.sample_windows(batch * 4, k)
            selected = self.phase_selection(b.gen[env, idx[:, 0], 1], batch)
            env, idx, valid = env[selected], idx[selected], valid[selected]
            e = env[:, None]
            states, poses = b.s[e, idx], b.pose[e, idx]
            delta = torch.stack([st.pose_delta(poses[:, j], poses[:, j + 1]) for j in range(k)], 1)
            loss = self.wm.loss(states[:, 0], b.a[e, idx[:, :k]], b.a_prev[e, idx[:, :k]],
                                self.xi[env], b.relief[e, idx[:, :k]].float(),
                                states[:, 1:], delta, valid)
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grads(self.wm.parameters(), 1.0, float("inf"))
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("nonfinite world model update")
            self.opt_wm.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def phase_selection(self, times, batch):
        count = int(batch * self.cfg.phase_prioritized_fraction)
        windows = self.cfg.phase_windows or (self.cfg.phase_window,)
        mask = torch.zeros_like(times, dtype=torch.bool)
        for low, high in windows:
            mask |= (times >= low) & (times <= high)
        eligible = mask.nonzero().flatten()
        if count == 0 or len(eligible) == 0:
            return torch.arange(batch, device=self.device)
        selected = eligible[torch.randint(len(eligible), (count,), device=self.device)]
        other = torch.randperm(len(times), device=self.device)[:batch - count]
        return torch.cat((selected, other))

    def seed(self):
        b = self.buf
        batch = self.cfg.imag_batch
        env, t = b.sample_states(batch * 4)
        selected = self.phase_selection(b.gen[env, t, 1], batch)
        env, t = env[selected], t[selected]
        gen = b.gen[env, t]
        if (gen[:, 0] != 0).any():
            raise ValueError("unknown clip ID in replay")
        return {
            "s": b.s[env, t], "pose": b.pose[env, t], "a_prev": b.a_prev[env, t],
            "history": b.obs[env, t, :SENSOR_DIM * HISTORY].float(),
            "time": gen[:, 1], "origin": gen[:, 2:5], "xi": self.xi[env],
        }

    def improve_policy(self):
        self.actor_updates += 1
        ramp = min(1.0, self.actor_updates / max(self.cfg.actor_ramp_iterations, 1))
        for group in self.opt_actor.param_groups:
            group["lr"] = self.cfg.lr * self.cfg.actor_lr_scale * ramp
        for network in (self.wm, self.critic):
            network.requires_grad_(False)
        self.opt_wm.zero_grad(set_to_none=True)
        try:
            objective, data = self.imag.rollout(self.wm, self.actor, self.critic, self.seed())
            loss = -objective.mean()
            self.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grads(self.actor.parameters(), 1.0, 0.1)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("nonfinite BPTT update")
            self.opt_actor.step()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        targets = self.imag.gae_targets(self.critic, data).reshape(-1)
        x = data["critic_in"][:, :-1].detach().reshape(-1, CRITIC_DIM)
        self.critic.norm.update(x)
        critic_loss = None
        for _ in range(self.cfg.critic_epochs):
            critic_loss = (self.critic(x).squeeze(-1) - targets).square().mean()
            self.opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            clip_grads(self.critic.parameters(), 1.0, float("inf"))
            self.opt_critic.step()
        return {
            "imagined_return": objective.mean().item(),
            "imagined_tracking_score": data["rewards"].mean().item() / DT,
            "actor_grad_norm": float(norm), "critic_loss": critic_loss.item(),
            "actor_updates": self.actor_updates, "actor_lr": self.opt_actor.param_groups[0]["lr"],
        }

    def save(self, iteration):
        path = self.output / f"model_{iteration}.pt"
        torch.save({
            "schema": CHECKPOINT_SCHEMA, "motion_schema": SCHEMA_VERSION,
            "replay_schema": REPLAY_SCHEMA, "robot": robot_signature(),
            "motion_sha256": sha256(self.clip.path), "iteration": iteration,
            "state_frame": self.clip.state_frame,
            "cfg": asdict(self.cfg), "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(), "world_model": self.wm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "obs_dim": OBS_DIM,
            "collected_transitions": self.collected_transitions, "actor_updates": self.actor_updates,
            "model_gate_passed": self.model_gate_passed,
            "recorded_commands_sha256": sha256(self.commands.path) if self.commands else None,
            "replay_saved": False, "rng_saved": False, "resume_supported": False,
        }, path)
        return path

    def run(self):
        self.save(0)
        start = time.monotonic()
        print(json.dumps({"event": "initial_collection_started", "num_envs": self.cfg.num_envs,
                          "backend": self.cfg.physics_backend, "steps": self.cfg.initial_collect}), flush=True)
        self.collect(self.cfg.initial_collect)
        print(json.dumps({"event": "initial_collection_finished",
                          "transitions": self.collected_transitions,
                          "elapsed_seconds": time.monotonic() - start}), flush=True)
        for it in range(1, self.cfg.iterations + 1):
            score = self.collect(self.cfg.collect_steps)
            loss = self.fit_world_model()
            stats = {"iteration": it, "world_model_loss": loss,
                     "collection_score_mixed_driver": score, "cumulative_failures": self.real_falls,
                     "collected_transitions": self.collected_transitions}
            if self.response_audit and (
                it % self.cfg.model_audit_interval == 0 or it == self.cfg.iterations
            ):
                audit = self.response_audit.evaluate(self.wm, it)
                self.model_gate_passed = (
                    audit["joint_response_cosine"] >= self.cfg.model_gate_cosine
                    and audit["joint_response_relative_error"] <= self.cfg.model_gate_relative_error
                )
                if self.cfg.model_gate_rotation:
                    self.model_gate_passed &= all(
                        audit[f"{name}_response_cosine"] >= self.cfg.model_gate_cosine
                        and audit[f"{name}_response_relative_error"] <= self.cfg.model_gate_relative_error
                        for name in ("gravity", "angular")
                    )
                stats["model_audit"] = audit
            if it > self.cfg.warmup_iterations and self.model_gate_passed:
                stats.update(self.improve_policy())
            stats["model_gate_passed"] = self.model_gate_passed
            if it % self.cfg.log_interval == 0 or it == self.cfg.iterations:
                stats["elapsed_seconds"] = time.monotonic() - start
                stats["device"] = self.device
                stats["physics_backend"] = self.cfg.physics_backend
                stats["num_envs"] = self.cfg.num_envs
                if self.cfg.physics_backend == "mjwarp":
                    stats["physics_capacity"] = self.env.physics.capacity_status()
                stats["torch_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(self.device) if self.device.startswith("cuda") else 0
                )
                print(json.dumps(stats, allow_nan=False), flush=True)
                with (self.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(stats, allow_nan=False) + "\n")
                self.output.joinpath("status.json").write_text(json.dumps(stats, indent=2) + "\n")
            if self.cfg.save_interval and it % self.cfg.save_interval == 0:
                print(f"saved {self.save(it)}", flush=True)
        print(f"saved {self.save(self.cfg.iterations)}", flush=True)


def load_actor(checkpoint, clip):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != CHECKPOINT_SCHEMA or ckpt.get("replay_schema") != REPLAY_SCHEMA:
        raise ValueError("incompatible checkpoint schema; chase/Go2 policies cannot resume here")
    if ckpt.get("robot") != robot_signature() or ckpt.get("motion_sha256") != sha256(clip.path):
        raise ValueError("checkpoint robot or motion hash differs")
    if ckpt.get("obs_dim") != OBS_DIM:
        raise ValueError("incompatible observation schema")
    if ckpt.get("state_frame", "forward-axis-v1") != clip.state_frame:
        raise ValueError("incompatible state frame")
    actor = Actor(OBS_DIM, tuple(ckpt["cfg"]["actor_hidden"]), 12)
    actor.load_state_dict(ckpt["actor"])
    return actor.eval(), ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()
    cfg = TrainCfg(**tomllib.loads(args.config.read_text())["train"])
    for name in ("output", "iterations", "seed", "num_envs"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
