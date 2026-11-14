# Input Contract: as2-proprio-reference-v2

## Actor: 347 dimensions

Five chronological frames, oldest first, each 42 dimensions:

| Field | Count | Hardware source | Convention |
| --- | --- | --- | --- |
| Body angular velocity | 3 | LowState.imu_state.gyroscope | rad/s, calibrated policy body axes |
| Projected gravity | 3 | LowState.imu_state.quaternion | R_body_world^T * [0,0,-1], unit vector |
| Joint displacement | 12 | LowState.motor_state[slot].q | calibrated radians minus default_q |
| Joint velocity | 12 | LowState.motor_state[slot].dq | calibrated rad/s |
| Previous applied action | 12 | local command ledger | (previous transmitted target-default_q)/0.25 |

The quaternion is documented wxyz. It is not accelerometer normalization:
accelerometers during takeoff/flight do not provide gravity alone.
Motor slots in policy FL/FR/RL/RR order are documented as
3,4,5,0,1,2,9,10,11,6,7,8. Slot indexing alone does not verify signs,
zero offsets, IMU mounting, units or command safety on the physical robot.
The decoder requires explicit calibration and rejects an unverified record
unless explicitly called for offline diagnostics.

Four future reference windows at 0, 0.1, 0.25, 0.5 seconds:
height(1), gravity(3), reference yaw-frame linear velocity(3), body angular
velocity(3), joint displacement(12), joint velocity(12), totaling 4 x 34.
These are prerecorded desired values, not measured height/velocity or
localization. The last scalar is bounded clip phase from a local fixed-tick
clock. The reference is eight seconds, 400 controls at 50 Hz, no wrapping.

No actual XY, yaw, position error, velocity estimator, foot force/contact,
height estimator, SportModeState or world-model prediction is used by actor.
`TrackingTask.observe` accepts legacy simulation arguments only to keep the
training call graph stable, and deliberately does not use them.
The NumPy deployment API does not accept pose or simulator state at all.

## Privileged Training Only

World model: original 70D state and planar pose evolution.
Critic: original privileged state plus position-conditioned reference goal.
Reward/termination: original physical tracking and failure definitions.
These may use simulator ground truth, but are not required at execution.
No PPO, teacher action residual or reference-state writes in policy evaluation.

## Noise And Timing

Training adds zero-mean Gaussian sample noise before history insertion:
gyro 0.01 rad/s, gravity components 0.003 (then renormalized),
joint position 0.002 rad, joint velocity 0.03 rad/s.
Each sample is clipped to three standard deviations; previous applied action
is not corrupted. The same function is used in real and imagined training.
These are provisional robustness assumptions, not measured AS2 error bounds.

History updates once per control tick from one coherent feedback sample.
Reset repeats a validated current frame. Previous action may be zero only
when the previous commanded target really equals default_q.
Future hardware integration must use fresh timestamps, monotonic phase,
validated command cadence and timeout handling; it must not advance phase
through a stalled control loop or substitute missing data.

## Acceptance Boundary

Passing input/parity tests proves input computability, not hardware safety.
This phase does not confirm real motor signs/zeros, IMU mounting, payload
mass/COM, motor torque-speed envelope, service transition or emergency stop.
Nominal simulation and additional perturbation tests must be reported
separately; there is no unconditional sim-to-real guarantee.

Original reference, baseline models and old checkpoints remain unchanged.
New checkpoint/replay schemas reject old 359-input models.
