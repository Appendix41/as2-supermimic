"""Load an AS2 expert policy with a compatible observation layout."""

import torch

from as2_supermimic.checkpoints import validate_checkpoint


class ActorDriver:
  def __init__(self, path: str, env_obs_dim: int, device, env_terms):
    from as2_supermimic.supermimic.train import Actor

    ck = torch.load(path, map_location=device, weights_only=True)
    validate_checkpoint(ck)
    self.obs_dim = ck["obs_dim"]
    old_terms = tuple(tuple(t) for t in ck["obs_terms"])
    if old_terms != tuple(env_terms[:len(old_terms)]) or self.obs_dim > env_obs_dim:
      raise ValueError("Expert observations must be a prefix of the AS2 task observations.")
    self.actor = Actor(self.obs_dim, tuple(ck["cfg"]["actor_hidden"]), 12).to(device)
    self.actor.load_state_dict(ck["actor"])
    self.actor.eval()

  @torch.no_grad()
  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(obs[:, :self.obs_dim])
