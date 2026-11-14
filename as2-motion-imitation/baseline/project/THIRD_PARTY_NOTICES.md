# Source and Asset Notices

## Go2-SuperMimic

The package in `src/as2_supermimic/`, inherited tests in `tests/upstream/`, and
`scripts/eval.py` are derived from Ziyue Zheng's Go2-SuperMimic:

```text
https://github.com/Zhengtianjin/go2-supermimic
commit d9eee9b3f248e5300e636c433255db2c1c3e086a
Copyright (c) 2026 Ziyue Zheng
MIT License: third_party/go2-supermimic/LICENSE
```

Namespace and robot/task adaptation do not make this a new SuperMimic algorithm.
The original research checkout is retained without edits.
`scripts/audit_upstream.py` generates a per-file provenance manifest and a diff
after normalizing the namespace. World-model and replay implementations are
preserved; trainer world-model fitting and policy-improvement functions remain
the upstream functions.

## Unitree AS2

The XML and STL assets in `src/as2_supermimic/robot/assets/unitree_as2/`
are unmodified copies from:

```text
https://github.com/unitreerobotics/unitree_rl_mjlab
commit 1425b15f73bd4095f0df53709d7c389c3eb9e790
Apache-2.0 license: third_party/unitree_rl_mjlab/LICENCE
```

The original `as2_constants.py` and shared `velocity_env_cfg.py` are retained
under `third_party/unitree_rl_mjlab/` as configuration evidence, not imported as
a second training framework. `robot/as2.py` adapts that configuration, including
an explicit collision-priority default required by the installed mjlab.
A head marker is added to the in-memory MjSpec for the chase task.
Asset source URLs, Git blob identities and SHA256 digests are in
`third_party/unitree_rl_mjlab/manifest.json`.

This software targets the official simulated AS2 asset. It does not certify a
particular AS2 EDU hardware/firmware revision, and includes no real-robot control.
