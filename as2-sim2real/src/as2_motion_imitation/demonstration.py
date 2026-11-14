"""Load recorded control targets for world-model data collection."""

from pathlib import Path

import numpy as np
import torch

from as2_motion_imitation.physics import DT


class RecordedCommands:
    def __init__(self, path, clip):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as f:
            action, time, qpos = f["action"], f["action_time"], f["qpos"]
        with np.load(clip.path, allow_pickle=False) as f:
            ref_qpos = f["qpos"]
        if action.shape != (clip.frames - 1, 12) or not np.isfinite(action).all():
            raise ValueError("recorded control shape/values do not match reference")
        if not np.allclose(time, np.arange(len(action)) * DT, atol=1e-8):
            raise ValueError("recorded control timestamps must use the control clock")
        if qpos.shape != ref_qpos.shape or not np.allclose(qpos, ref_qpos, atol=1e-7):
            raise ValueError("recorded controls are not paired with this reference")
        if np.max(np.abs(action)) > 4:
            raise ValueError("recorded controls exceed learner action bounds")
        self.actions = torch.tensor(action, dtype=torch.float32, device=clip.device)

    def sample(self, time):
        # The source used zero-order hold, not interpolated measured joint angles.
        # Float32 clock accumulation can drift a few microseconds below a tick.
        index = torch.floor(time / DT + 1e-3).long().clamp(0, len(self.actions) - 1)
        return self.actions[index]
