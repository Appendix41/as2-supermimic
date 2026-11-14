# Media Provenance

All MP4s are byte-identical copies of the existing simulation evaluation videos:

- `single-jump.mp4`: iteration 1000, learned actor, seed 100.
- `backflip-experimental.mp4`: iteration 2400, learned actor, seed 100.
- `backflip-scripted-reference.mp4`: accepted scripted physical reference,
  explicitly not a learned policy.
- `proprio-single-jump-simulation.mp4`: historical proprioceptive actor in
  simulation, not a real-robot success video.

GIFs use the full corresponding policy videos, 640 x 360 pixels at 12 fps,
a 96-color palette, and Bayer dithering. There is no playback-rate remapping,
trajectory alteration, synthetic image generation, or removal of landing
periods. Small duration differences reflect frame-rate/GIF quantization.

The later real-hardware project has been inspected and its controller/report
imported, but no hardware video was found there. A physical-robot clip can be
added separately when supplied. This folder does not substitute simulation
video for hardware evidence.
