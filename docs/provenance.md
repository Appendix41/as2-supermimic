# Provenance And Preservation

The release is assembled from selected project files. The original workspaces
are unchanged. Some Python comments and docstrings have been edited in the
release copy; calculation and control logic are unchanged. Configurations,
reference data, checkpoints and run records retain their original content.
`source-manifest.json` records original project-relative paths and
SHA256 digests of copied public files. Remote imports use `agx-snapshot/`
source labels; private host paths and access details remain outside the
publication tree.

Publication edits:

- Public metric JSON files remove the workstation's absolute project prefix
  from string values. Numerical results and acceptance booleans are unchanged.
  Both source and public-file hashes are recorded.
- GIF previews are downsampled from complete actor-only simulation videos,
  without changing the playback rate or synthesizing motion.
- Python comment/docstring edits record both source and published hashes.
  ASTs and non-comment tokens are compared with docstrings normalized.
  Imports, statements, parameters and ordinary string values must match.
  Docstring introspection and CLI help derived from docstrings may differ.
  License notices and tool directives are preserved.

The forward-jump generator is a byte-preserved exception: reference metadata
records its complete file hash. Its original header remains so that source
verification continues to work without rewriting the historical record.

`tools/run.sh` and `tools/prepare_test_data.py` are new packaging helpers.
They do not alter the learner, motion generator, controller, or acceptance
logic. Machine-specific old launchers and original-environment integrity
checkers are omitted from the public workspace entry points.

The old simulation deployment test for a superseded pose-dependent v1 actor
bundle is not included. The proprioceptive single-jump tests remain in the
separate sim-to-real workspace. Broad raw real-robot telemetry is not distributed;
the corresponding historical test must be explicitly deselected unless
authorized data is supplied.

Baseline runtime code, assets, tests, and licenses are retained; selected
module descriptions have been shortened in the release copy.
Internal handoff documents and machine-specific snapshot metadata are not
published as user documentation. The public selection is not represented as
the complete original baseline snapshot.

The later hardware snapshot is now included under `hardware-single-jump/`.
The controller and three runtime files were retrieved from the AGX and their
comments/docstrings were edited for publication. The completed report is an
unedited copy. The matching four-file inference bundle is
shared with the historical export rather than duplicated. Its environment
receipt was inspected locally but is not published with private source paths.

The current controller snapshot postdates the selected run, and the run
does not contain source/weight hashes. Matching current files is not a frozen
run archive. These limits are explicit in `hardware-single-jump/snapshot.json`.
The new hardware tests check preservation, plan-only behavior, and recorded
input parity without a robot SDK; they do not certify hardware safety.

Automatic hardware launchers, including a script containing a hardcoded
credential, were not copied into the publication or local source snapshot.
No original script was edited to remove that credential.
