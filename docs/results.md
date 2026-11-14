# Experiment Results

This page distinguishes a demonstrated motion from passing every physical
acceptance condition. Packaging did not rerun training or change a threshold.
Simulation results come from the archived November 12, 2026 experiments;
the selected later hardware record is described separately below.

## Single Forward Jump

The iteration-1000 actor passed 20/20 nominal simulation trials from ordinary
standing. Mean continuous flight was 0.310 s, airborne forward displacement
0.10598 m, and total forward displacement 0.23532 m. The last two distances
describe different quantities.

These trials used a fixed model and small reset perturbations followed by
settling. Reused diagnostic seeds are not a blind robustness benchmark.
See [original numeric results](results/single-jump-metrics.json).

## Experimental Backflip Transfer

The AS2 adaptation learned an airborne backward-turning motion from the
physics-generated backflip reference. The final actor completed the
rotation-related objectives in all 20 archived trials. This is the
experimental motion-transfer result showcased on the front page.

| Recorded Measure | Final Actor |
| --- | --- |
| Checkpoint iteration | 2400 |
| Flight duration | 0.490-0.492 s |
| Airborne backward rotation | 4.77502-4.80272 rad |
| Total backward rotation, seed 100 | 6.251592 rad |
| Peak root height, seed 100 | 0.659010 m |
| Final position | Four feet in contact; low root velocity at 8 s for seed 100 |
| Rotation-related checks | Met in 20/20 trials |
| Original complete physical acceptance | 0/20 |

The original full acceptance additionally checks nonfoot ground contacts,
joint limits, and recovery to the initial standing pose. These conditions
were not met. Seed 100 had 101 nonfoot-contact samples, 111 joint-limit
violation samples, and a maximum joint violation of 0.021913 rad at 500 Hz.
Its final maximum standing-joint deviation was 0.155795 rad against the original
0.12 rad tolerance. A recorded stable-standing duration of zero does not mean
the robot was still moving rapidly or lying on the ground at the end.

We therefore report an **experimental learned backflip-motion demonstration**,
not a physically qualified backflip policy. No new relaxed pass count is
invented. The original evaluator and thresholds remain unchanged.

- [Learned actor video](media/backflip-experimental.mp4)
- [Scripted reference video, not a learned policy](media/backflip-scripted-reference.mp4)
- [Original metrics, with local path prefixes removed](results/backflip-metrics.json)
- [Independent record/action audit](results/backflip-audit.json)
- [Scripted reference metrics](results/backflip-reference-metrics.json)

## Historical Proprioceptive Single Jump

The iteration-1400 policy passed 20/20 nominal simulation trials, 55/55 separate
perturbation trials, and 20/20 NumPy closed-loop simulation trials in the
earlier work. Its nominal mean airborne displacement was 0.11089 m and
flight duration was 0.3184 s.

These are simulation measurements, not measurements of the hardware experiment.

[Historical simulation metrics](results/proprio-single-jump-metrics.json)

## Hardware Single Jump

The author reports successful single-jump transfer to the physical AS2.
The imported later AGX record reports `completed`, 400 policy steps in 7.98 s,
and return of control to `ai`. The deployed actor bundle matches the local
proprioceptive export byte for byte.

The author has also supplied a [real-robot video](media/hardware-single-jump.mp4).
This separate clip is not timestamp-matched to the selected run log.
Only one selected completed run record is published; no total hardware success
rate is inferred. The log does not measure flight or displacement. The current
controller snapshot postdates the run and the log does not embed model/source hashes.
See [integration and evidence](sim2real-integration.md).
