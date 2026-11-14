# Reproduction

## Environment

The historical runs and the publication-copy CPU regression tests use:

| Component | Version |
| --- | --- |
| Python | 3.12.16 |
| MuJoCo | 3.16.0 |
| MuJoCo Warp | 3.16.0 |
| Warp | 1.18.0 |
| PyTorch | 2.14.1+cu130 |
| mjlab | Commit `8bb2bc9e187797689ec82c78c35019afd2fd0376` |

The source commit identifies mjlab more precisely than its `1.6.0` version
label. The historical installation audit found different dependency metadata
in the similarly numbered PyPI package. Do not substitute by version label.

This candidate has been tested using the existing validated runtime, not by
reinstalling in a clean environment. A fresh-install recipe/lock remains a
publication task. Do not dependency-resolve the copied baseline `pyproject.toml`
into an existing working environment.

Activate a matching environment so `python3` resolves to its interpreter, or
set `PYTHON_BIN` to that environment's Python executable. The new wrapper does
not hardcode the original machine's interpreter path.

## Workspaces

From the repository root:

```bash
bash tools/run.sh simulation -c "import as2_motion_imitation; print(as2_motion_imitation.__file__)"
bash tools/run.sh sim2real -c "import as2_motion_imitation; print(as2_motion_imitation.__file__)"
```

Only one workspace and its baseline are added to `PYTHONPATH` at a time.
The wrapper changes the current directory into that workspace and directs
outputs/caches there. Never combine the two same-named packages.

## Model Files

Download the matching Release attachments into `checkpoints/` using the
[download and checksum instructions](../checkpoints/README.md), or use an
absolute checkpoint path. The initial private draft requires repository access.

From the repository root:

```bash
bash tools/run.sh simulation -m as2_motion_imitation.evaluate_jump \
  --checkpoint ../checkpoints/as2-single-jump-model_1000.pt \
  --output outputs/evaluation/my-single-jump

bash tools/run.sh simulation -m as2_motion_imitation.evaluate_backflip \
  --checkpoint ../checkpoints/as2-backflip-experimental-model_2400.pt \
  --output outputs/evaluation/my-backflip --trials 20
```

These are simulation evaluations. No physical robot is contacted.
New evaluations were not performed merely to prepare this publication copy.

## CPU Regression Tests

Some unchanged historical tests expect recorded test inputs under `outputs/`.
Materialize the explicitly included fixtures, then run the suites:

```bash
python3 tools/prepare_test_data.py --release-assets checkpoints

CUDA_VISIBLE_DEVICES="" bash tools/run.sh simulation -m pytest \
  tests baseline/project/tests -c ../pytest.ini -q -p no:cacheprovider

CUDA_VISIBLE_DEVICES="" bash tools/run.sh sim2real -m pytest \
  tests baseline/project/tests -c ../pytest.ini -q -p no:cacheprovider \
  -k "not recorded_real_lowstate_input_is_computable_but_not_calibrated"
```

The single deselected historical test reads real-robot telemetry that is not
distributed. This is an explicit data-availability exclusion, not relaxation
of motion acceptance. GPU tests are skipped because these commands hide CUDA.

On November 14, 2026, the publication copies produced:

| Suite | Passed | Skipped | Deselected |
| --- | --- | --- | --- |
| Simulation + baseline | 106 | 6 | 0 |
| Historical sim-to-real + baseline | 292 | 6 | 1 |

These are software regression results on the existing runtime, not new skill
success rates, clean-install validation, or hardware acceptance.
Both suites were rerun after the publication comment/docstring edits, with
the same pass and skip counts.

## Hardware

The later controller and deployed runtime are now preserved separately in
[hardware-single-jump](../hardware-single-jump/README.md). Its bundle matches the
proprio export; use that folder's offline plan instructions rather than an
older bootstrap script.

On November 14, 2026, 14 new publication checks passed, including all 8,000
recorded-input parity cases and a plan-only subprocess that forbids robot SDK
imports. These checks used the same local Python 3.12 runtime above, not a fresh
installation of the AGX's Python 3.10 runtime.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest hardware-single-jump/tests \
  -c pytest.ini -q -p no:cacheprovider
```

The tests locate the bundle in `checkpoints/as2-proprio-actor-bundle`, the local
sibling `../release-assets/`, or an explicit `AS2_INFERENCE_BUNDLE` directory.
Tests needing the bundle skip when it is absent; the reported 14-pass run had
all four bundle files available and skipped none.

The [integration notes](sim2real-integration.md) distinguish the current snapshot
from the earlier run record. Nothing in this packaging workflow starts DDS,
changes a robot service, or actuates motors.
