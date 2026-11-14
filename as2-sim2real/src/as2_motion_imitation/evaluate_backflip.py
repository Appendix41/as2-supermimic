"""Actor-only backflip evaluation using the shared physical rollout engine."""

import argparse
from pathlib import Path

from as2_motion_imitation.backflip_audit import ACCEPTANCE
from as2_motion_imitation.evaluate_jump import evaluate
from as2_motion_imitation.learner import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output,
             range(args.seed_start, args.seed_start + args.trials), not args.no_video,
             ROOT / "data/processed/as2_backflip.npz", ACCEPTANCE)


if __name__ == "__main__":
    main()
