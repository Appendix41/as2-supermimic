# Media Provenance

The final-policy and reference MP4s are byte-identical copies of existing
simulation evaluation videos:

- `single-jump.mp4`: iteration 1000, learned actor, seed 100.
- `backflip-experimental.mp4`: iteration 2400, learned actor, seed 100.
- `backflip-scripted-reference.mp4`: accepted scripted physical reference,
  explicitly not a learned policy.
- `proprio-single-jump-simulation.mp4`: historical proprioceptive actor in
  simulation, not a real-robot success video.

## Training Stages

- `single-jump-training-stages.mp4`: iterations 0, 400, 800, and 1000, seed 100.
  Four full 201-frame clips at 25 fps, totaling 32.16 s. The original frames
  are re-encoded with a stage-label band; playback speed is unchanged.
- `backflip-training-stages.mp4`: byte-identical copy of the existing 56.28 s
  montage, covering iterations 0, 1400, 1600, 1800, 2000, 2200, and 2400.
  Its original Chinese stage captions are retained.
- `backflip-landing-failures.mp4`: byte-identical copy of the existing 28.88 s
  replay of iterations 1800 and 2000. Each full 8.04 s clip is followed by
  simulation time 1.60 to 3.20 s at quarter speed and 1.5x magnification.
  Slow motion repeats frames; it does not interpolate robot motion.

These are checkpoint evaluations, not recordings of the full collection batch.
The JPEG previews are frames from their corresponding MP4s. Video hashes,
chapter offsets, and preview-frame indices are in
[training-media.json](training-media.json). The copied backflip montage's
original [provenance](backflip-replay-provenance.json) uses paths relative to its
original replay folder, not this publication folder.

## Front-Page GIFs

GIFs use the full corresponding policy videos, 640 x 360 pixels at 12 fps,
a 96-color palette, and Bayer dithering. There is no playback-rate remapping,
trajectory alteration, synthetic image generation, or removal of landing
periods. Small duration differences reflect frame-rate/GIF quantization.

## Hardware

The later real-hardware project has been inspected and its controller/report
imported, but no hardware video was found there. A physical-robot clip can be
added separately when supplied. This folder does not substitute simulation
video for hardware evidence.
