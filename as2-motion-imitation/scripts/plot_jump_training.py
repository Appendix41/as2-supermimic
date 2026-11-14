"""Plot recorded training metrics and an actor-only evaluation, without smoothing."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    config = json.loads((args.run / "config.json").read_text())
    it = [r["iteration"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(it, [r["world_model_loss"] for r in rows], color="#176b52")
    ax.set(title="World-model loss (8-step AR)", xlabel="Iteration", yscale="log")
    ax = axes[0, 1]
    audits = [r["model_audit"] for r in rows if "model_audit" in r]
    for key, name, color in (
        ("joint_response_relative_error", "Joint sensitivity relative error", "#b34c36"),
        ("joint_response_cosine", "Joint sensitivity cosine", "#176b52"),
    ):
        ax.plot([r["iteration"] for r in audits], [r[key] for r in audits],
                "o-", label=name, color=color)
    ax.set(title="Physical action-response audit", xlabel="Iteration")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    policy = [r for r in rows if "actor_grad_norm" in r]
    ax.plot([r["iteration"] for r in policy], [r["actor_grad_norm"] for r in policy],
            color="#795548")
    ax.set(title="BPTT actor gradient norm", xlabel="Iteration", yscale="log")
    ax = axes[1, 1]
    if args.evaluation:
        with np.load(args.evaluation / "seed-100.npz") as f:
            ax.plot(f["time"], f["base_pos"][:, 2], label="Actor-only physics", color="#b34c36")
        with np.load(ROOT / config["motion"]) as f:
            ax.plot(np.arange(len(f["qpos"])) * 0.02, f["qpos"][:, 2],
                    label="Scripted reference", color="#176b52")
        metrics = json.loads((args.evaluation / "metrics.json").read_text())
        ax.set(title=f"Seed 100 height; evaluation {metrics['successes']}/{metrics['trials']}",
               xlabel="Time (s)", ylabel="Root height (m)")
        ax.legend(fontsize=8)
    else:
        ax.plot(it, [r["collection_score_mixed_driver"] for r in rows], color="#52738b")
        ax.set(title="Collection score (mixed driver, NOT policy success)", xlabel="Iteration")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"AS2 motion training | {config['num_envs']} {config['physics_backend']} environments",
                 fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
