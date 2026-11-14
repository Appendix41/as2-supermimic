# Third-Party Notices

This file documents third-party components distributed with the AS2
reference-conditioned motion-imitation project. These notices concern reused
code and assets; the task, observation contracts, and deployment scope are
described in the [method overview](docs/method.md).

## Go2-SuperMimic

Source: `Zhengtianjin/go2-supermimic`, commit
`d9eee9b3f248e5300e636c433255db2c1c3e086a`.
Copyright (c) 2026 Ziyue Zheng. MIT License.

The original notices and full license are retained in both workspaces:

- `baseline/project/THIRD_PARTY_NOTICES.md`
- `baseline/project/third_party/go2-supermimic/LICENSE`
- `baseline/project/third_party/go2-supermimic/port-manifest.json`

This attribution applies to the reused baseline components. The AS2 reference
conditioning, tracking tasks, and deployment work are documented separately;
their additions do not remove the license obligations for reused code.

## Unitree AS2

Model/configuration source: `unitreerobotics/unitree_rl_mjlab`, commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790`.
The original Apache-2.0 license, asset manifest, and source configuration
are retained under `baseline/project/third_party/unitree_rl_mjlab/`.

AS2 XML/STL files remain under the corresponding baseline robot package.
The publication operation does not alter these assets.

## Method And Runtime

The learning method is attributed to SuperMimic. The runtime uses mjlab,
MuJoCo, MuJoCo Warp, PyTorch, and their dependencies under their respective
licenses. These dependency environments are not vendored into this repository.

The later hardware controller imports the Unitree SDK2 Python API. The SDK
and Cyclone DDS binaries are external dependencies, not copied into this
publication. The inspected SDK package metadata declares BSD-3-Clause.
Its deployment-receipt identity and the imported controller provenance are
documented in [Sim-to-Real](docs/sim2real-integration.md).

See [licensing status](docs/licensing.md) for the author's pending license
selection, including the imported follow-up hardware contributions.
