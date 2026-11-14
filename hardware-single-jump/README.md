# Hardware Single Jump

This folder preserves the later single-jump controller retrieved from the AGX
on November 14, 2026, plus one completed run record from November 13.
The author reports a successful physical forward jump. The selected log
records 400 learned-policy steps and return of control to `ai`.

## Contents

| Path | Role |
| --- | --- |
| `as2_jump.py` | One-shot controller; comments edited for publication |
| `src/` | Deployed NumPy runtime; docstrings edited for publication |
| `evidence/jump-152659.json` | One original completed single-jump run record |
| `snapshot.json` | Source/bundle hashes, provenance limits, observed environment |
| `tests/` | New offline publication checks; no robot SDK execution |

The controller snapshot was modified after the selected run. The log contains
no source or weight hashes. These are matched current files and historical
evidence, not a frozen, cryptographically bound run archive.
`snapshot.json` records the original and published file hashes separately.
Comment/docstring edits leave the calculation and control statements unchanged.

## Offline Inspection

Use Python with NumPy. Download and extract `as2-proprio-actor-bundle.tar.gz`
under `checkpoints/`, following the [model instructions](../checkpoints/README.md).
Keep its runtime files and export report together. From the repository root:

```bash
PYTHONPATH=hardware-single-jump/src python3 -B hardware-single-jump/as2_jump.py \
  --bundle checkpoints/as2-proprio-actor-bundle

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest hardware-single-jump/tests \
  -c pytest.ini -q -p no:cacheprovider
```

Without `--execute`, the entry point validates the bundle and prints
the plan, then exits before importing the robot SDK. The tests also forbid
SDK imports in that subprocess. The tests accept `AS2_INFERENCE_BUNDLE` for an
alternative bundle location.

For the local staging layout, the bundle is at
`../release-assets/as2-proprio-actor-bundle`, relative to the repository root.

## Hardware Runtime

The inspected deployment uses Python 3.10.22, NumPy 1.26.4, Cyclone DDS 0.10.2,
and `unitree_sdk2py` package version 1.0.1. Its deployment receipt identifies SDK
commit `814556d15970dd2ecf1c9984e845ca02ab07e206`; the copied SDK directory has no
Git metadata, so that commit was not independently verified against its files.
SDK sources, virtualenvs, and platform-specific shared libraries are not vendored.

Hardware execution additionally needs the SDK, its DDS runtime, an explicitly
selected robot-facing interface, verified joint/IMU conventions, and an operator
prepared to stop the robot. The entry point exposes `--execute` and `--interface`;
adding `--execute` is a real actuation request, not an offline verification step.
There is no automatic hardware-start wrapper in this release.

The preserved controller is experimental. Its own header notes unproven
actuator envelopes, firmware timeout behavior, and low-level stop behavior.
The previous-action history records the latest computed clipped action, without
a transport acknowledgment; it is not proof that every target was applied.
The publisher runs separately at approximately 100 Hz while inference is 50 Hz.
These limitations are documented, not silently fixed in the copied source.

The original bundle's `hardware_execution_enabled: false`, null motor-map field,
and old validation notes remain untouched. The later controller supplies its
own motor mapping and explicit actuation branch. These metadata values are not
a physical safety interlock and should not be changed merely to run this code.

See [integration details](../docs/sim2real-integration.md) for the training
lineage, joint mapping, gains, result interpretation, and evidence boundaries.
