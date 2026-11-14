# Training

## Hardware And Runtime

The three released models were trained on a single **NVIDIA RTX PRO 5000
Blackwell**, using `cuda:0`, **4096 parallel physics environments**, and the
`mjwarp` backend. All three runs used seed 23. The device name is recorded in
each training console header, not inferred from checkpoint filenames.

The runtime versions and environment notes are in [Reproduction](reproduction.md).

## Recorded Training Costs

| Run | Iterations | Actor Updates | Loop Time | Collected Transitions |
| --- | ---: | ---: | ---: | ---: |
| Single forward jump | 1000 | 700 | 1206.87 s / 20.1 min | 18,104,320 |
| Experimental backflip | 2400 | 1401 | 2563.74 s / 42.7 min | 41,369,600 |
| Proprioceptive single jump | 1400 | 1100 | 1805.87 s / 30.1 min | 24,657,920 |

These are the final `elapsed_seconds` values from the three recorded runs.
The timer begins before initial data collection and includes model warmup,
learning updates, and in-loop diagnostics. It excludes startup, reference
search, independent evaluation, video preparation, hardware deployment, and
the final checkpoint write after the last logged timestamp. These are
individual run measurements, not an average over repeated experiments.

## Shared Settings

| Setting | Value |
| --- | --- |
| World-model hidden layers | 1024 / 1024 / 1024 |
| Actor and critic hidden layers | 512 / 256 / 128 |
| World-model autoregressive window | 8 |
| World-model batch / updates per iteration | 512 / 4 |
| Imagination horizon / batch | 32 / 256 |
| Collection steps per environment per iteration | 4 |
| Initial collection, single jump / proprio single jump | 420 steps per environment |
| Initial collection, backflip | 500 steps per environment |

The single-jump and proprioceptive runs use 300 model-only warmup iterations.
The backflip run sets a minimum of 600; its model-response gate delayed the
first actor update until iteration 1000. The actual actor update counts above
come from the final status files.

| Run | Recorded Configuration | Final Training Status |
| --- | --- | --- |
| Single forward jump | [Config](results/training-single-jump-config.json) | [Status](results/training-single-jump-status.json) |
| Experimental backflip | [Config](results/training-backflip-config.json) | [Status](results/training-backflip-status.json) |
| Proprioceptive single jump | [Config](results/training-proprio-single-jump-config.json) | [Status](results/training-proprio-single-jump-status.json) |

The [machine-readable summary](results/training-summary.json) includes exact
timings, recorded device headers, and source-file hashes.

## Stage Videos

These videos replay evaluations of saved checkpoints, not a continuous recording
of the 4096 collection environments. Every stage comparison retains the complete
source rollout at original speed.

| Video | Checkpoint Iterations | Duration |
| --- | --- | ---: |
| [Single-jump stages](media/single-jump-training-stages.mp4) | 0, 400, 800, 1000 | 32.16 s |
| [Backflip stages](media/backflip-training-stages.mp4) | 0, 1400, 1600, 1800, 2000, 2200, 2400 | 56.28 s |
| [Backflip landing failures](media/backflip-landing-failures.mp4) | 1800, 2000 | 28.88 s |

The landing-failure reel includes each full rollout followed by a quarter-speed,
1.5x enlarged view of the landing interval. No new training or simulation was
run to prepare these videos. See [media provenance](media/README.md) for the
source and playback details.
