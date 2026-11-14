# Single-Jump Sim-to-Real Integration

## Integrated Scope

On November 14, 2026, the later AGX project `DeepSeek dsh/dsh - supermimic`
was inspected over SSH. Selected files were imported, then comments and
docstrings were edited in the publication copy. Original files are unchanged.
No deployment program, DDS client, or robot service command was run during
publication preparation.

The author reports a successful real-robot single forward jump. This release
contains its later controller snapshot, one completed run record, and
author-supplied real-robot footage, not triple-jump records or real-robot
backflip control.

| Directory | Responsibility |
| --- | --- |
| `as2-sim2real/` | Earlier proprioceptive training, export, interfaces, regression tests |
| `hardware-single-jump/` | Later AGX one-shot controller, deployed runtime, selected hardware record |
| Release attachment `as2-proprio-actor-bundle` | Matching deployed actor, reference, parity data, original manifest |

## Training To Hardware

The inspected controller imports `InferenceBundle` and calls the learned
`bundle.act()` at each policy tick. It is not simply playing recorded joint
targets. The surrounding settle, hold, return-to-standing, and service handoff
are scripted. This distinction applies to the current source snapshot.

The deployed actor, reference, parity cases, manifest, and runtime match the
existing local proprio export byte for byte. The bundle identifies checkpoint:

```text
as2-proprio-single-jump-model_1400.pt
c939893b8c0021f48a5952dcd94221e5c9a8455adc8b3ecc1c958773b8d66b38
```

The current bundle is therefore tied to the archived training model, rather
than being a new unidentified policy. The historical run itself does not
embed artifact hashes, so this is not a cryptographic proof of which bytes
were loaded at run time.

| Contract | Current Controller Snapshot |
| --- | --- |
| Observation | 347D: five 42D sensor frames, four 34D reference states, phase |
| Live inputs | IMU gyro/quaternion, joint q/dq, previous computed clipped action |
| Joint policy order | FL, FR, RL, RR; hip, thigh, calf within each leg |
| SDK slots | `3,4,5,0,1,2,9,10,11,6,7,8` |
| IMU conversion | Quaternion in `wxyz`; projected gravity from deployed runtime |
| Joint/IMU transforms | No additional sign/offset correction or IMU rotation in this entry point |
| Position targets | `default_q + 0.25 * clip(action, -4, 4)` |
| Gains per leg | Kp `40,40,60`; Kd `2,2,3` |
| Inference | 400 steps at nominal 50 Hz; no clip looping |
| Publisher | Separate thread, nominal 10 ms sleep; not a hard-real-time guarantee |
| Control handoff | Release motion mode, run sequence, restore `ai` |

## Selected Evidence

The original [jump-152659.json](../hardware-single-jump/evidence/jump-152659.json)
has filesystem modification time November 13, 2026, 15:27:27 +08:00.
That timestamp is not an embedded signed run timestamp.

| Logged Field | Value |
| --- | --- |
| Status | `completed` |
| Policy steps | 400 / 400 |
| Policy duration | 7.98 s |
| Maximum reported absolute roll/pitch | 0.355637 rad |
| Maximum target deviation from default pose | 0.718394 rad |
| Publisher frame count | 1555 |
| Total elapsed time | 26.26 s |
| Owner after restore / final recovery | `ai` / `ai` |

These are controller-completion and feedback diagnostics, not flight-height,
displacement, landing-quality, or population success-rate measurements.
The user confirms the physical jump. The report's `phase` remains `init` on
success because this
snapshot updates that field only on faults; it is not evidence of no execution.
The retained handback trace covers control restoration, not the whole jump.

The controller file's modification time is November 13, 2026, 15:29:21 +08:00,
after the selected report. We preserve the latest inspected snapshot and the
original record separately; we do not claim a frozen run-source archive.
[snapshot.json](../hardware-single-jump/snapshot.json) records hashes and
these limits.

## Hardware Footage

The author separately supplied a [5.87-second real-robot clip](media/hardware-single-jump.mp4)
on November 14, 2026. The README includes a looping preview of the complete clip.
The published MP4 preserves all 176 frames and the original audio, with H.264
encoding for web playback.

This is visual evidence supplied for the project, not a timestamp-matched
recording of `jump-152659.json`. The clip has not been bound to that log or to
specific source/model hashes. Its receipt date is not a confirmed recording
date. [Media provenance](media/hardware-video.json) records the original and
published hashes and the conversion.

## Preserved Boundaries

The old bundle manifest still has `hardware_execution_enabled: false`,
`sdk_motor_map: null`, and earlier validation notes. The NumPy loader expects
that schema, while the later controller implements an explicit `--execute`
branch and its own mapping. Those old fields were not rewritten to manufacture
new acceptance or to imply that this entry point cannot actuate.

The controller header identifies unproven actuator envelopes, firmware timeout,
and low-level stopping behavior. It has feedback-age, tilt, button, board-state,
and sampled-CRC checks, but those checks are not a safety certification.
The previous-action history follows the latest computed target without
transport acknowledgment, so the stronger "actual applied action" wording in
the older training contract is not fully guaranteed by this entry point.
These differences remain in the published controller; only comments and
docstrings were edited.

The credential-bearing automatic jump launcher is excluded entirely. Probe
scripts, multi-jump records, broad raw telemetry, private connection details,
virtual environments, and vendor binaries are also not published.

See [hardware folder instructions](../hardware-single-jump/README.md) for
offline inspection and environment details. The public README does not
automatically launch hardware.
