import pytest
import torch

from as2_supermimic.curriculum import STAGE_TASKS, configure_stage
from as2_supermimic.env.env_cfg import as2_chase_env_cfg
from as2_supermimic.target.generator import BallTarget


@pytest.mark.parametrize("stage", STAGE_TASKS)
def test_stage_modes_and_zero_probability_kinds(stage):
  cfg = as2_chase_env_cfg(terrain=stage == "terrain", obstacles=stage == "obstacles")
  configure_stage(cfg, stage, STAGE_TASKS[stage])
  command = cfg.commands["target"]
  assert sum(command.mode_probs) == pytest.approx(1.0)
  assert command.target.head_offset == 0.318
  gen = BallTarget(command.target, 4096, "cpu")
  gen.spawn(torch.arange(4096), torch.zeros(4096, 2), torch.zeros(4096), torch.zeros(4096, 2),
            torch.ones(4096, dtype=torch.bool), torch.ones(4096, dtype=torch.bool))
  permitted = {float(i) for i, prob in enumerate(command.target.kind_probs) if prob > 0}
  assert set(gen.kind.tolist()) <= permitted
  if stage in ("chase", "terrain"):
    assert command.mode_probs == (1.0, 0.0, 0.0)


def test_stage_task_mismatch_fails():
  with pytest.raises(ValueError, match="requires task"):
    configure_stage(as2_chase_env_cfg(), "terrain", "As2-Chase-Flat")
