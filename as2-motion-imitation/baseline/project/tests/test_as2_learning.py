"""Regression checks for replay boundaries and model-based policy gradients."""

import torch

from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.buffer import GEN_DIM, ReplayBuffer, pack_generator
from as2_supermimic.supermimic.imagine import ImagineCfg, Imagination
from as2_supermimic.supermimic.train import Actor, Critic
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.generator import BallTarget, BallTargetCfg
from as2_supermimic.target.terrain import SCAN_POINTS


def test_replay_windows_mask_episode_boundaries_after_ring_wrap():
  torch.manual_seed(4)
  buffer = ReplayBuffer(1, 12, 350, "cpu")
  for i in range(25):
    buffer.add(torch.full((1, st.STATE_DIM), float(i)), torch.zeros(1, 3),
               torch.zeros(1, 12), torch.zeros(1, 12), torch.zeros(1, 350),
               torch.zeros(1, GEN_DIM), torch.tensor([i % 3 == 0]),
               torch.zeros(1, SCAN_POINTS))
  env, idx, valid = buffer.sample_windows(128, 8)
  sequence = buffer.s[env[:, None], idx, 0]
  assert torch.all(sequence[:, 1:] - sequence[:, :-1] == 1)
  assert sequence.min() >= 13 and sequence.max() <= 24
  expected = torch.cumprod((sequence[:, :-1].long() % 3 != 0).float(), 1)
  torch.testing.assert_close(valid, expected)
  assert (valid == 0).any()


def test_policy_gradient_flows_through_frozen_world_model():
  torch.manual_seed(3)
  n = 4
  cfg = BallTargetCfg()
  gen = BallTarget(cfg, n, "cpu")
  gen.spawn(torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
            torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
  consts = st.RobotConsts(
    torch.zeros(n, 12), torch.full((n, 12), 0.25), torch.full((12,), 40.0),
    torch.full((12,), 2.0), torch.full((12,), 50.0),
    torch.tensor([-2.0, 2.0]).expand(n, 12, 2).clone(),
  )
  imag = Imagination(ImagineCfg(horizon=4), consts, cfg, 5, "cpu")
  wm, actor, critic = WorldModel((32, 32)), Actor(350, (32, 32), 12), Critic(imag.critic_dim, (32, 32))
  for parameter in wm.parameters():
    parameter.requires_grad_(False)
  state = torch.zeros(n, 70)
  state[:, 0], state[:, 3] = 0.30, -1.0
  state[:, st.FEET] = torch.tensor([0.2, -0.15, -0.274] * 4)
  seed = {
    "env": torch.arange(n), "s": state, "pose": torch.zeros(n, 3),
    "a_prev": torch.zeros(n, 12), "obs": torch.randn(n, 350),
    "gen": pack_generator(gen), "xi": torch.zeros(n, 4), "enc_bias": torch.zeros(n, 12),
  }
  objective, data = imag.rollout(wm, actor, critic, seed)
  # This terminal state has no direct action-reward shortcut: its gradient
  # can reach the actor only through learned dynamics.
  gradients = torch.autograd.grad(data["critic_in"][:, -1, 0].sum(),
                                  tuple(actor.parameters()), retain_graph=True)
  assert all(torch.isfinite(g).all() for g in gradients)
  assert sum(g.abs().sum() for g in gradients) > 1e-6
  (-objective.mean()).backward()
  assert all(p.grad is None for p in wm.parameters())
  assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
  assert torch.isfinite(imag.gae_targets(critic, data)).all()
