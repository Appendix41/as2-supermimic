# Reference-Conditioned Motion Imitation

AS2-SuperMimic learns to track a finite reference motion using robot feedback.
Reference conditioning is present during both policy training and inference;
it is not limited to a demonstration used before training.

## Policy Inputs

Both training workspaces use five 42D sensor frames. Each frame contains body
angular velocity, projected gravity, joint position, joint velocity, and the
previous-action term. At each tick, reference states are sampled at offsets
of 0.0, 0.1, 0.25, and 0.5 seconds, followed by normalized clip phase.

| Policy | History | Reference Goal | Total | Live Global Root Pose |
| --- | --- | --- | --- | --- |
| Simulation tracking | 210D | Four 37D targets + phase | 359D | Used to compute root-pose error |
| Proprioceptive single jump | 210D | Four 34D targets + phase | 347D | Not required by the actor |

The simulation goal includes three root-pose-error values for each reference
sample. The proprioceptive actor omits those live pose-error terms; privileged
pose information remains available to its training critic and reward.
The reference itself can still contain root-state information. Removing live
pose measurements does not mean removing the reference trajectory.

The contracts are implemented in the [simulation task](../as2-motion-imitation/src/as2_motion_imitation/task.py)
and the [proprioceptive task](../as2-sim2real/src/as2_motion_imitation/task.py).
They are distinct model interfaces, not interchangeable checkpoint formats.

## Learning And Control

1. Prepare an AS2 reference clip and the associated recorded commands used by the selected training configuration.
2. Collect physical transitions using the configured mix of reference-driven and policy-driven collection, including reference-state resets where enabled.
3. Fit an autoregressive world model from replayed transitions.
4. Optimize the actor through imagined rollouts using reference-tracking rewards, BPTT, critic bootstrap, and TD(lambda).
5. Evaluate the learned actor in closed-loop physics with feedback and reference inputs, without the collection-time reference driver.

The tracking reward compares joint positions, root height, projected gravity,
foot positions, velocity, and root pose, with an action-change regularizer.
The [learner](../as2-motion-imitation/src/as2_motion_imitation/learner.py) advances
reference time inside imagined rollouts and ends each finite clip without
looping it indefinitely.

For the hardware-oriented single jump, the NumPy export bakes the reference
at 400 control ticks spaced 0.02 seconds apart. The later controller calls
the learned actor for each policy tick; settling, holding, and control handoff
are scripted. See [sim-to-real integration](sim2real-integration.md) for the
hardware input mapping and previous-action acknowledgment limitation.

## Scope

The released checkpoints each target one training clip. Reference-conditioned
inputs do not establish zero-shot tracking of arbitrary unseen motions or a
single general multi-motion policy. Simulation backflip results are reported
as experimental motion transfer, not hardware backflip deployment.

This page describes the motion-tracking task and implementation behavior.
Third-party component provenance and license obligations are documented
separately in [third-party notices](../THIRD_PARTY_NOTICES.md).
