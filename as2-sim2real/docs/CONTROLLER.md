# Integrated Single-Jump Controller

Version: `as2-single-jump-controller-v1`, November 12, 2026.
This adds execution software to the unchanged 347D inference bundle. It is
not evidence that the robot's calibration, actuator limits or physical
stopping have passed acceptance. No valid hardware config is supplied.

## Components

- `deployment/controller.py`: configuration, calibrated inverse motor map,
  official frame construction, submitted-command ledger, guard and state
  machine. No DDS imports.
- `deployment/controller_worker.py`: spawned actor-only process. Receives
  the 347D observation, returns a numbered action. Never accesses motors.
- `deployment/controller_transport.py`: official SDK subscribers, CRC,
  direct callback fault latches, one LowCmd publisher and MotionSwitcher.
  Imports SDK only after validating the hardware profile.
- `deployment/controller_run.py`: integration, bounded loop, signals,
  ownership handoff, in-memory audit and output files.
- `deployment/controller_selftest.py`: actual actor process SIGKILL,
  synthetic complete runs and optional SDK serialization/CRC checks.
- `scripts/verify_controller_physics.py`: supervisor, histories, targets
  and gains connected to the existing MuJoCo AS2 model.

## Flow

1. Validate bundle hashes/config. Hardware additionally requires explicit
   approval records, robot identifier, known service name, calibration,
   actuator envelope and `--confirm-handoff`. The bundle's own
   `hardware_execution_enabled: false` remains unchanged: it describes
   the inference export, not a substitute for controller approval.
2. Hardware takes an exclusive local deployment lock. Start the actor
   process and wait for its checkpoint-bound ready message. No motor
   commands are sent during startup.
3. Validate feedback, neutral remote, stationary standing and health.
   Enable callback latching, check the expected motion owner, recheck
   the feedback/stop signal, release once, and confirm owner name is empty.
   Any unknown release outcome prevents position commands.
4. Start a separate ownership polling thread, using bounded SDK RPCs.
   Lost/unknown ownership prevents further commands. A local lock does
   not exclude another host's publisher; external exclusivity remains
   a deployment prerequisite.
5. SETTLE interpolates measured joint targets to default_q using explicitly
   configured takeover gains, then collects fresh command/sensor history.
   Gains do not start at zero. Initial pose and speed checks must pass;
   this is not a stand-up-from-floor routine.
6. POLICY requests 400 numbered actions at 50 Hz with no catch-up steps,
   wrapping, replay or retries. The supervisor polls at a nominal 2 ms
   interval and retransmits the last accepted target between policy ticks.
   A successful local write is required before history/phase commit.
7. HOLD retains the final target for a bounded configured interval.
   STOPPING sends bounded zero-kp/zero-tau damping only with fresh,
   CRC-valid feedback and retained ownership, then closes publishing.
   STOPPED is terminal. No button can re-arm the instance.

Policy decoding uses only IMU/joints, locally submitted targets and fixed
reference/phase. SportModeState is NOT needed after handoff. Previous
action is reconstructed from the actual encoded target, not the unsubmitted
actor output or measured q. "Submitted" means the SDK accepted a DDS write;
the protocol does not provide a per-command motor acknowledgement.

## Fault Handling

Checks include CRC, receipt age, tick advance/wrap/replay, mode_machine,
mode_pr, all mainboard words, motor errors/warnings, finite sensors,
joint range, speed, estimated/predicted torque, temperature, voltage,
tilt, tracking error, target slew, actor sequence/lease and loop timing.
Neither measured data nor a rejected actor output is silently repaired.

L2+B is a direct latched stop. Other remote buttons conservatively stop
the controller too, including START. The callback latches brief events
even if the latest sample later shows released buttons. The November 12
real observation showed L2+B and sport mode 0 without mainboard bit 12;
the implementation therefore never relies on bit 12 alone.

The parent owns publishing and survives actor process death. SIGINT and
SIGTERM request terminal stopping. Failure of the parent, operating system,
robot network or a blocked native write cannot be handled by that same
parent. Firmware command timeout and a verified external stop remain
mandatory. DDS acceptance does not prove physical braking.

On stale/invalid feedback, unknown motion ownership or unsupported joint
mode, publishing stops instead of guessing a mode or competing with a
different controller. There is no automatic `SelectMode`, StandUp,
RecoveryStand or restoration of active vendor motion. Recovery is a
separate on-site procedure after publishing has ceased.

Finite damping/STOP may leave the robot unsupported. This exit strategy,
stop_kd and firmware timeout behavior require physical acceptance; do not
interpret a software terminal state as a safely standing or stopped robot.
No guarantee of hardware real-time operation is made.

## Non-Actuating Commands

Local:

```bash
bash scripts/run.sh -m as2_motion_imitation.deployment.controller_run \
  --mode validate --bundle outputs/deployment/proprio-v2/bundle \
  --config configs/controller-synthetic.json --output outputs/controller-validation

OPENBLAS_NUM_THREADS=1 bash scripts/run.sh \
  -m as2_motion_imitation.deployment.controller_run \
  --mode simulate --bundle outputs/deployment/proprio-v2/bundle \
  --config configs/controller-synthetic.json --output outputs/controller-synthetic
```

Orin, from the dedicated deployment root:

```bash
bash scripts/run_controller.sh --mode validate --bundle bundle \
  --config configs/controller-synthetic.json --output outputs/controller-validation
bash scripts/run_controller.sh --mode simulate --bundle bundle \
  --config configs/controller-synthetic.json --output outputs/controller-synthetic
bash scripts/run_controller_selftest.sh --bundle bundle \
  --config configs/controller-synthetic.json --sdk-frames --nominal-trials 3 \
  --output outputs/controller-selftest.json
```

Output paths must be new. `validate` creates no SDK entity; structural
validity and `hardware_ready` are separate fields. `simulate` does not
import SDK, open DDS or call a robot service. Its target-following mock is
NOT physics. `--sdk-frames` imports only official message/CRC types, not a
publisher. Never run a vendor motion example to test this integration.

The hardware CLI exists but no ready-to-actuate config is shipped.
`configs/controller-synthetic.json` has broad software fixture limits,
unverified identity transforms and all acceptance fields false. Do not
copy those numbers to a hardware profile or toggle approvals to force a run.
Provide measured hardware limits/calibration and evidence for each named
acceptance item, bound to the actual robot/checkpoint/service.

## Timing And Evidence

The controller leaves 20 ms supervisor deadline, 15 ms actor deadline and
10 ms tick lateness limits unchanged after initial Orin deadline failures.
Bounded session logs stay in memory and are written on exit. Cyclic garbage
collection is done before handoff and deferred until the bounded session
ends; previous GC state is restored. This affects only this process, not
shared services. This is a mitigation, not proof that all scheduling
outliers or DDS blocking are eliminated.

The log contains accepted/rejected target submissions, policy request
feedback and previous submitted action. `motor_acknowledged` stays false.
On abrupt parent death, the final in-memory log may be lost.

The Orin Python 3.10 SDK runtime originally lacked NumPy. The bootstrap
copies existing NumPy 1.26.4 and native libraries into the dedicated
runtime, checks every file, and refuses overwrites. It performs no package
installation or shared-environment change. Its source interpreter remains
an external dependency, as for the original SDK environment.

Latest evidence and known limitations:
`outputs/deployment/proprio-v2/controller-integration-20261112/`.
