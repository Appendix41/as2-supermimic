# Checkpoints

Full checkpoints are distributed separately from the Git source tree in
[GitHub Releases](https://github.com/Appendix41/as2-supermimic/releases).
The initial upload is draft `v0.1.0-rc1` in a private repository; downloading it
requires an authenticated account with access to the draft.

| File | Workspace | Status |
| --- | --- | --- |
| `as2-single-jump-model_1000.pt` | simulation | Nominal simulated single jump |
| `as2-backflip-experimental-model_2400.pt` | simulation | Learned backflip-motion experiment; original physical limitations retained |
| `as2-proprio-single-jump-model_1400.pt` | sim2real | Model identified by the byte-matched AGX inference bundle |

From the repository root, download and verify the attachments:

```bash
gh release download v0.1.0-rc1 --repo Appendix41/as2-supermimic \
  --dir checkpoints --pattern '*.pt' \
  --pattern 'as2-proprio-actor-bundle.tar.gz' \
  --pattern 'manifest.json' --pattern 'SHA256SUMS'

cd checkpoints
sha256sum --check SHA256SUMS --ignore-missing
tar -xzf as2-proprio-actor-bundle.tar.gz
sha256sum --check SHA256SUMS --ignore-missing
cd ..
```

`--ignore-missing` checks the selected downloads without requiring the separate
attachment README. `manifest.json` records checkpoint and bundle identities.
Each policy must be paired with its matching reference, observation schema,
and code workspace. These files do not contain the complete replay/simulator/RNG
state needed for exact training continuation.

The `as2-proprio-actor-bundle.tar.gz` attachment contains the lightweight inference
export used by `hardware-single-jump/`. Keep its `actor.npz`, `reference.npz`,
`parity_cases.npz`, and `manifest.json` together; the archive also includes
the original `export-verification.json`. The selected historical
hardware log does not embed their hashes; see
[integration provenance](../docs/sim2real-integration.md).
