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
The retained JPEG stills are frames from their corresponding MP4s. Video hashes,
chapter offsets, and still-frame indices are in
[training-media.json](training-media.json). The copied backflip montage's
original [provenance](backflip-replay-provenance.json) uses paths relative to its
original replay folder, not this publication folder.

## Final-Policy GIFs

GIFs use the full corresponding policy videos, 640 x 360 pixels at 12 fps,
a 96-color palette, and Bayer dithering. There is no playback-rate remapping,
trajectory alteration, synthetic image generation, or removal of landing
periods. Small duration differences reflect frame-rate/GIF quantization.

## Training Preview GIFs

The three training-table previews are looping GIFs at 480 pixels wide and 12 fps.
They retain original-speed simulation time 1.2 through 3.6 s from each selected
checkpoint, with the original stage labels. The landing-failure preview uses
the two original-speed chapters; its slow-motion sequences remain in the full MP4.

These are excerpts, not full rollouts. No interpolation or speed-up is applied.
Exact source intervals, hashes, frame counts, and durations are in
[preview-gifs.json](preview-gifs.json).

## Hardware

`hardware-single-jump.mp4` is real-robot footage supplied by the author on
November 14, 2026. The full 176-frame clip is 960 x 540 at 30 fps, approximately
5.87 seconds. It is transcoded from HEVC to H.264 for web playback, with the
original AAC audio copied and nonessential container metadata removed.
There are no temporal cuts, speed changes, spatial crops, or motion interpolation.
The original supplied file is preserved in the local-only audit directory.

`hardware-single-jump.gif` shows the complete clip at 640 x 360 and 12 fps,
at original speed and without audio. It uses a 96-color palette and Bayer
dithering. Frame-rate and GIF timing quantization account for small duration
differences.

The user identifies this as the project's physical AS2 forward-jump footage.
It was supplied separately from the AGX snapshot; its identity with the
selected historical log, recording date, and exact source/model version have
not been established. See [hardware-video.json](hardware-video.json) for hashes
and conversion details.
