"""Local action-response audit; diagnostic state copies are not policy trials."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from as2_motion_imitation.physics import Physics
from as2_supermimic.supermimic import state as st
from as2_supermimic.supermimic.world_model import WorldModel
from as2_supermimic.target.terrain import SCAN_POINTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    ckpt = torch.load(ROOT / "outputs/training/crouch-probe/model_150.pt",
                      map_location="cpu", weights_only=True)
    model = WorldModel(tuple(ckpt["cfg"]["wm_hidden"]))
    model.load_state_dict(ckpt["world_model"])
    model.requires_grad_(False)
    p = Physics()
    qpos, qvel, act = p.data.qpos.copy(), p.data.qvel.copy(), p.data.act.copy()
    state, _ = p.state()
    eps, horizon = 0.2, 8
    real, predicted = [], []
    for joint in range(12):
        real_pair, pred_pair = [], []
        for sign in (-1, 1):
            mujoco.mj_resetData(p.model, p.data)
            p.data.qpos[:], p.data.qvel[:], p.data.act[:] = qpos, qvel, act
            mujoco.mj_forward(p.model, p.data)
            a = np.zeros(12, np.float32)
            a[joint] = sign * eps
            s = torch.tensor(state)[None]
            prev = torch.zeros(1, 12)
            for _ in range(horizon):
                p.step(a)
                with torch.no_grad():
                    s, _ = model(s, torch.tensor(a)[None], prev,
                                  torch.tensor([[0.6, 0, 0, 0]]), torch.zeros(1, SCAN_POINTS))
                prev = torch.tensor(a)[None]
            real_pair.append(p.state()[0])
            pred_pair.append(s[0].numpy())
        real.append((real_pair[1] - real_pair[0]) / (2 * eps))
        predicted.append((pred_pair[1] - pred_pair[0]) / (2 * eps))
    real, predicted = np.array(real), np.array(predicted)
    terms = {}
    for name, sl in (("height", slice(0, 1)), ("joint_position", st.Q), ("feet", st.FEET)):
        a, b = real[:, sl].ravel(), predicted[:, sl].ravel()
        terms[name] = {
            "cosine_alignment": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
            "true_sensitivity_norm": float(np.linalg.norm(a)),
            "model_sensitivity_norm": float(np.linalg.norm(b)),
            "relative_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)),
        }
    report = {
        "scope": "Eight-step local finite-difference response at normally settled standing.",
        "epsilon_normalized_action": eps, "horizon_steps": horizon,
        "diagnostic_state_restoration": True, "not_an_evaluated_policy_rollout": True,
        "metrics": terms,
        "conclusion": "Prediction RMSE alone does not certify the action Jacobian needed by BPTT.",
        "next_step": "Validate dynamics sensitivities over recorded squat phases, then increase model-only warmup before actor updates.",
    }
    out = ROOT / "outputs/evaluation/probe-150"
    np.savez_compressed(out / "action-response.npz", real=real, predicted=predicted)
    (out / "action-response.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
