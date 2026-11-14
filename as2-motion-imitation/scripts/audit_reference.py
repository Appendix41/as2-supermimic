"""Pinned candidate audit. No untrusted pickle or policy checkpoint is loaded."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from as2_motion_imitation.physics import Physics
from as2_supermimic.robot.morphology import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7f03a3fb7f6b2e684899530862ba103e81570363"
BASE = f"https://huggingface.co/datasets/MIMUW-Robotics/kine2go/resolve/{REVISION}"
FILES = {
    "README.md": "README.md", "metadata.json": "metadata.json", "LICENSE": "LICENSE",
    "LICENSES/cassi_BSD-3-Clause.txt": "LICENSES-cassi_BSD-3-Clause.txt",
    **{f"data/solo8_jump_forward_a/{n}": f"solo8_jump_forward_a/{n}"
       for n in ("clip.json", "config.json", "motion.npy")},
}


def main():
    raw = ROOT / "data/raw/kine2go"
    receipts = []
    for remote, local in FILES.items():
        path = raw / local
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f"{BASE}/{remote}", timeout=45) as response:
                path.write_bytes(response.read())
        receipts.append({
            "url": f"{BASE}/{remote}", "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
        })
    d = raw / "solo8_jump_forward_a"
    x = np.load(d / "motion.npy", allow_pickle=False)
    meta = json.loads((d / "clip.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    names = cfg["env_cfg"]["dof_names"]
    declared_mapping = [names.index(n) for n in JOINT_NAMES]
    q = x[:, 6:18][:, declared_mapping]
    physics = Physics()
    limits = physics.model.jnt_range[physics.jids]
    violation = np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
    report = {
        "revision": REVISION, "reference_exists": True, "direct_AS2_mapping_accepted": False,
        "scope": "Only forward variant A audited completely; not a proof no usable Go2 jump exists.",
        "actual_shape": list(x.shape), "declared_n_frames": meta["n_frames"],
        "declared_fps": meta["fps"], "configured_motion_range": cfg["policy_cfg"]["motion_range"],
        "all_values_finite": bool(np.isfinite(x).all()),
        "raw_joint_columns_first_frame": x[0, 6:18].tolist(),
        "declared_joint_order": names,
        "declared_mapping_max_AS2_joint_violation_rad": float(violation.max()),
        "all_declared_velocity_fields_zero": bool(
            np.all(x[:, 18:36] == 0) and np.all(x[:, 55:61] == 0)
        ),
        "base_net_displacement_m": (x[-1, 48:51] - x[0, 48:51]).tolist(),
        "first_frame_min_foot_center_height_m": float(x[0, 36:48].reshape(4, 3)[:, 2].min()),
        "license_label": meta["license"],
        "license_caveat": "Dataset LICENSE says Solo8 license is inherited by analogy with cassi's top-level code grant.",
        "decision": "Do not guess missing joint order, velocities or boundaries. Use local AS2 squat per user fallback.",
        "selected": "data/processed/as2_crouch.npz",
        "selected_origin": "Locally scripted PD physics capture, not pre-existing official AS2/Go2 motion.",
        "receipts": receipts,
    }
    (raw / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    out = ROOT / "outputs/reference-selection/source-audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "receipts"}, indent=2))


if __name__ == "__main__":
    main()
