# AS2 Proprioceptive Single-Jump Deployment

Release: proprio-v2, 347 actor inputs, checkpoint model_1400.pt.
Checkpoint SHA256:
`c939893b8c0021f48a5952dcd94221e5c9a8455adc8b3ecc1c958773b8d66b38`.

Dedicated Orin directory: `/home/jrz/project/as2-supermimic-proprio-v2`.
The older `/home/jrz/project/as2-supermimic-deploy` contains a different,
359-input policy and must not be overwritten or treated as this release.

## Scope

The original release contains NumPy inference and explicit DDS readers.
The subsequent controller-v1 addition now includes a guarded motor
publisher and service handoff implementation. Its default mode only
validates configuration; it has no automatic start. No valid hardware
configuration is supplied, and the inference bundle's
`hardware_execution_enabled` remains false. Read `docs/CONTROLLER.md`.
Offline success is not physical motion acceptance.

The system Python runs offline inference. The dedicated Python 3.10 venv
runs the DDS recorder, using copied SDK/CycloneDDS libraries. No pip install,
shared dependency upgrade, power-mode change or service change is needed.
The venv retains a dependency on the existing base Python interpreter.
For controller-v1, a checked copy of existing Python 3.10 NumPy is added
inside the dedicated runtime. Shared environments remain unchanged.

## Bounded Tests

From the release root:

```bash
bash scripts/run_offline.sh --output runs/offline.json --cycles 400
bash scripts/run_telemetry.sh --interface end0 --duration 10 --output runs/telemetry
bash scripts/run_readiness.sh --interface end0 --duration 10 --output runs/readiness
bash scripts/run_control_selftest.sh --sdk-frames --output runs/control-selftest.json
bash scripts/run_feedback_crc.sh --duration 5 --output runs/feedback-crc.json
bash scripts/run_remote_stop.sh --duration 120 --output runs/remote-stop
```

Output paths must not already exist. Offline verification compares 8000
recorded inputs with the training implementation, followed by 400 timed
cycles. It excludes DDS and motor transport. Telemetry only receives data;
it does not run the actor or publish commands.
The separate readiness reader adds mainboard fault/E-stop flags, battery
SOC and raw motor temperature/voltage fields. It never grants motor
takeover approval. Its latest-sample diagnostic timing is not a full
control-loop benchmark.
The control self-test kills its own synthetic command producer and checks
the independent supervisor's memory-only stopped decision. It does not
kill another project, send motor commands or prove a physical watchdog.
The feedback CRC check reads actual LowState; it does not run the actor.
The remote-stop reader records CRC-valid remote button transitions, raw
mainboard state changes and sport mode changes. It announces LISTENER_READY
only after LowState and mainboard feedback arrive and exits within the
selected observation duration (maximum 120 seconds, excluding setup).
Arrange the press/release with the on-site operator after that announcement.
The official remote manual describes hold L2 plus tap B as damping/soft
stop. The robot must be supported before requesting this transition.
Do not request START or automatically restore active control afterward.
Mainboard bit 12 and L2+B are recorded independently; their equivalence,
causal relationship and operation after service release are not assumed.
A completed observation is not an emergency-stop qualification.

Actor input needs calibrated IMU/joints, known applied action history and
local phase. While the vendor controller owns the robot, its last applied
normalized policy action is unknown. Never substitute measured joint
position or unexecuted shadow actions and call that a closed-loop test.

## Physical Acceptance Gates

Before any motor takeover:

1. An on-site operator must confirm a clear area, appropriate restraint and
   a verified emergency stop that works in the intended low-level mode.
2. Verify joint order, signs, zero offsets, IMU axes and units on this robot.
   A plausible static packet is not calibration.
3. Verify motor limits, PD gains, unused command slots, CRC, mode matching,
   command cadence and a local independent timeout/watchdog.
4. Measure actual payload/COM and inspect actuator/battery/thermal limits.
5. Validate service release and return of control, including error paths.
6. Perform restrained, low-energy takeover and stable standing acceptance.
   Only then consider a separately authorized single jump.

Do not use high-level StopMove as an assumed low-level emergency stop.
Do not disable services or start vendor motor examples as a connectivity test.
Do not change the calibration to verified merely to bypass decoder checks.

Local results:
`outputs/deployment/proprio-v2/orin-20261112/` in the as2-sim2real project.
Follow-up status:
`outputs/deployment/proprio-v2/hardware-preflight-20261112/`.
Software/protocol bring-up:
`outputs/deployment/proprio-v2/control-bringup-20261112/`.
Remote-stop signal observation:
`outputs/deployment/proprio-v2/remote-stop-20261112/`.
Integrated controller and non-actuating validation:
`outputs/deployment/proprio-v2/controller-integration-20261112/`.
