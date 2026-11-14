#!/usr/bin/env python3
"""Run one policy clip, hold, then return control to ai.

The deployment used slot mapping (3,4,5,0,1,2,9,10,11,6,7,8), checked with
an operator-assisted leg probe. Joint directions, PD hold and service handoff
were checked on that robot; verify them again for a different setup.

Actuator limits, firmware command timeout and low-level L2+B stopping remain
unverified. Keep an operator at the remote. Any button latches a fault and
requests damping through the controller; this is not an independent stop.
"""

import argparse
import gc
import json
import math
import threading
import time

import numpy as np

from as2_motion_imitation.deployment.runtime import (
    InferenceBundle, init_history, push_history, projected_gravity,
)

SLOTS = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


class Fault(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle", default="bundle")
    parser.add_argument("--interface", default="end0")
    parser.add_argument("--kp", default="40,40,60")
    parser.add_argument("--kd", default="2,2,3")
    parser.add_argument("--settle-rad-s", type=float, default=0.15)
    parser.add_argument("--hold-s", type=float, default=0.6)
    parser.add_argument("--tilt-rad", type=float, default=0.9)
    parser.add_argument("--report")
    args = parser.parse_args()

    kp3 = [float(v) for v in args.kp.split(",")]
    kd3 = [float(v) for v in args.kd.split(",")]
    kp = [kp3[0], kp3[1], kp3[2]] * 4
    kd = [kd3[0], kd3[1], kd3[2]] * 4

    bundle = InferenceBundle(args.bundle)
    plan = {"plan": "settle -> 400 policy steps -> hold -> restore ai",
            "kp": kp, "kd": kd, "steps": bundle.steps, "dt": bundle.dt,
            "settle_rad_s": args.settle_rad_s, "hold_s": args.hold_s}
    if not args.execute:
        plan["status"] = "plan_only"
        print(json.dumps(plan, indent=2))
        return 0

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MainBoardState_
    from unitree_sdk2py.utils.crc import CRC

    crc = CRC()
    factory = unitree_hg_msg_dds__LowCmd_
    st = {"q": None, "dq": None, "quat": None, "gyro": None, "ns": 0,
          "tick": None, "mask": 0, "crc": True, "board": (), "fault": None, "msg": None}

    def latch(reason):
        if st["fault"] is None:
            st["fault"] = reason

    n_msg = [0]

    def on_low(message):
        try:
            n_msg[0] += 1
            raw = bytes(message.wireless_remote)
            mask = int.from_bytes(raw[2:4], "little")
            st.update(msg=message, ns=time.monotonic_ns(), tick=int(message.tick), mask=mask)
            if mask:
                latch("remote_button")
            elif raw[:2] != b"\x55\x51":
                latch("remote_format")
            elif n_msg[0] % 50 == 0 and int(crc.Crc(message)) != int(message.crc):
                latch("feedback_crc")
        except Exception:
            latch("lowstate_decode")

    def refresh():
        message = st.get("msg")
        if message is None:
            return False
        st["q"] = [float(m.q) for m in message.motor_state[:12]]
        st["dq"] = [float(m.dq) for m in message.motor_state[:12]]
        st["quat"] = [float(v) for v in message.imu_state.quaternion]
        st["gyro"] = [float(v) for v in message.imu_state.gyroscope]
        return True

    def on_board(message):
        words = tuple(int(v) for v in message.state)
        st["board"] = words
        if len(words) != 6 or any(words):
            latch("mainboard_flag")

    ChannelFactoryInitialize(0, args.interface)
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(2.0)
    switcher.Init()
    for _ in range(10):
        code, _ = switcher.CheckMode()
        if code == 0:
            break
        time.sleep(0.3)

    subs = []
    for topic, kind, cb in (("rt/lowstate", LowState_, on_low),
                            ("rt/lf/mainboardstate", MainBoardState_, on_board)):
        sub = ChannelSubscriber(topic, kind)
        subs.append(sub)
        sub.Init(cb, 10)
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()

    def build(target_pol, gains=True):
        command = factory()
        command.mode_machine = 1
        command.mode_pr = 0
        command.reserve = [0, 0, 0, 0]
        for motor in command.motor_cmd:
            motor.mode = 0
            motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
            motor.reserve = 0
        for i, slot in enumerate(SLOTS):
            motor = command.motor_cmd[slot]
            motor.mode = 1
            motor.q = float(target_pol[i])
            motor.kp = float(kp[i]) if gains else 0.0
            motor.kd = float(kd[i]) if gains else 0.2
        command.crc = 0
        command.crc = crc.Crc(command)
        return command

    stream = {"target": None, "stop": threading.Event(), "sent": 0, "error": None}

    def streamer():
        try:
            while not stream["stop"].is_set():
                target = stream["target"]
                if target is not None:
                    publisher.Write(build(target))
                    stream["sent"] += 1
                time.sleep(0.010)
        except Exception as error:
            stream["error"] = f"{type(error).__name__}: {error}"
            latch("publisher_error")

    def owner(attempts=4):
        for _ in range(attempts):
            code, data = switcher.CheckMode()
            if code == 0 and isinstance(data, dict):
                return data.get("name")
            time.sleep(0.3)
        return None

    def fresh_switcher():
        client = MotionSwitcherClient()
        client.SetTimeout(2.0)
        client.Init()
        time.sleep(0.3)
        return client

    def restore(name="ai", budget_s=12.0):
        deadline = time.monotonic() + budget_s
        client = switcher
        tries = 0
        while time.monotonic() < deadline:
            code, data = client.CheckMode()
            if code == 0 and isinstance(data, dict) and data.get("name") == name:
                return True
            try:
                client.SelectMode(name)
            except Exception:
                pass
            tries += 1
            if tries % 3 == 0:
                try:
                    client = fresh_switcher()
                except Exception:
                    pass
            time.sleep(0.6)
        code, data = client.CheckMode()
        return code == 0 and isinstance(data, dict) and data.get("name") == name

    def guard(now_ns):
        refresh()
        if st["fault"]:
            raise Fault(st["fault"])
        if stream["error"]:
            raise Fault("stream_" + stream["error"])
        if now_ns - st["ns"] > 40_000_000:
            raise Fault("feedback_age")
        quat = st["quat"]
        w, x, y, z = quat
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        if abs(roll) > args.tilt_rad or abs(pitch) > args.tilt_rad:
            raise Fault(f"tilt_roll{roll:.2f}_pitch{pitch:.2f}")
        return roll, pitch

    trace = []
    trace_stop = threading.Event()
    trace_t0 = [0.0]
    sampler = [None]

    def sample_trace():
        while not trace_stop.is_set():
            message = st.get("msg")
            if message is not None:
                try:
                    trace.append({
                        "t": round(time.monotonic() - trace_t0[0], 3),
                        "mode": [int(m.mode) for m in message.motor_state[:12]],
                        "q": [round(float(message.motor_state[i].q), 3) for i in (2, 5, 8, 11)],
                        "rpy": [round(float(v), 3) for v in message.imu_state.rpy],
                    })
                except Exception:
                    pass
            time.sleep(0.01)

    def decimate(rows):
        out, prev = [], None
        for index, row in enumerate(rows):
            if index % 10 == 0 or row["mode"] != prev:
                out.append(row)
                prev = row["mode"]
        return out[-400:]

    report = dict(plan)
    report.update(status="running", phase="init", steps_done=0, max_tilt=0.0,
                  target_excursion=0.0, owner_before=None, owner_after_release=None,
                  owner_after_restore=None, recovery_owner=None)
    started = time.monotonic()
    thread = threading.Thread(target=streamer, daemon=True)
    phase = "init"
    try:
        deadline = time.monotonic() + 3
        while not refresh() and time.monotonic() < deadline:
            time.sleep(0.01)
        if st["q"] is None:
            raise Fault("no_feedback")
        if st["board"] and any(st["board"]):
            raise Fault("mainboard_flag")
        if st["mask"]:
            raise Fault("remote_not_neutral")
        report["owner_before"] = owner()
        if report["owner_before"] is None:
            raise Fault("owner_unknown")
        if report["owner_before"] == "":
            raise Fault("already_released")

        q_pol = np.array([st["q"][s] for s in SLOTS], dtype=np.float32)
        thighs = [q_pol[1], q_pol[4], q_pol[7], q_pol[10]]
        calves = [q_pol[2], q_pol[5], q_pol[8], q_pol[11]]
        if not all(0.5 < float(v) < 1.2 for v in thighs) or \
                not all(-1.95 < float(v) < -1.05 for v in calves):
            raise Fault("not_standing_" + ",".join(f"{float(v):.2f}" for v in thighs))
        target = q_pol.copy()
        start_pose = q_pol.copy()
        stream["target"] = tuple(float(v) for v in target)
        thread.start()
        time.sleep(0.2)

        phase = "release"
        code, _ = switcher.ReleaseMode()
        report["release_rc"] = code
        time.sleep(0.6)
        report["owner_after_release"] = owner(1)

        phase = "settle"
        goal = np.array(bundle.default_q, dtype=np.float32)
        settle_start = time.monotonic()
        while True:
            now = time.monotonic()
            guard(time.monotonic_ns())
            step = args.settle_rad_s * 0.005
            remaining = False
            for i in range(12):
                delta = goal[i] - target[i]
                if abs(delta) > 1e-6:
                    target[i] += max(-step, min(step, delta))
                    remaining = True
            stream["target"] = tuple(float(v) for v in target)
            if not remaining:
                break
            if time.monotonic() - settle_start > 6:
                raise Fault("settle_timeout")
            time.sleep(0.005)
        time.sleep(0.3)
        report["settle_s"] = round(time.monotonic() - settle_start, 2)
        q_check = np.array([st["q"][i] for i in SLOTS], dtype=np.float32)
        settle_err = float(np.max(np.abs(q_check - goal)))
        report["settle_tracking_err"] = settle_err
        if settle_err > 0.35:
            raise Fault(f"release_not_effective_err_{settle_err:.2f}")

        phase = "policy"
        gc_was_enabled = gc.isenabled()
        gc.disable()
        warm_frame = bundle.sensor(
            np.zeros(3, dtype=np.float32), np.array([0, 0, -1], dtype=np.float32),
            np.array(bundle.default_q, dtype=np.float32),
            np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32))
        warm_hist = init_history(warm_frame)
        for _ in range(20):
            bundle.act(bundle.observe(warm_hist, 0))
        history = None
        last_action = np.zeros(12, dtype=np.float32)
        t0 = time.monotonic()
        for step_index in range(bundle.steps):
            deadline_ns = t0 + step_index * bundle.dt
            while time.monotonic() < deadline_ns:
                time.sleep(0.0005)
            lateness = time.monotonic() - deadline_ns
            if lateness > 0.020 and step_index > 1:
                raise Fault(f"policy_tick_late_{lateness*1000:.1f}ms")
            roll, pitch = guard(time.monotonic_ns())
            report["max_tilt"] = max(report["max_tilt"], abs(roll), abs(pitch))
            q_now = np.array([st["q"][s] for s in SLOTS], dtype=np.float32)
            dq_now = np.array([st["dq"][s] for s in SLOTS], dtype=np.float32)
            frame = bundle.sensor(
                np.asarray(st["gyro"], dtype=np.float32),
                projected_gravity(st["quat"]), q_now, dq_now, last_action)
            history = init_history(frame) if step_index == 0 else push_history(history, frame)
            obs = bundle.observe(history, step_index)
            action = bundle.act(obs)
            target = bundle.target(action)
            if not np.isfinite(target).all():
                raise Fault("nonfinite_target")
            stream["target"] = tuple(float(v) for v in target)
            last_action = np.clip(action, -4, 4).astype(np.float32)
            report["steps_done"] = step_index + 1
            report["target_excursion"] = max(
                report["target_excursion"],
                float(np.max(np.abs(target - bundle.default_q))))
        report["policy_s"] = round(time.monotonic() - t0, 2)
        if gc_was_enabled:
            gc.enable()

        phase = "hold"
        hold_until = time.monotonic() + args.hold_s
        while time.monotonic() < hold_until:
            guard(time.monotonic_ns())
            time.sleep(0.01)

        phase = "stand"
        stand_goal = np.array(start_pose, dtype=np.float32)
        stand_deadline = time.monotonic() + 4.0
        while time.monotonic() < stand_deadline:
            guard(time.monotonic_ns())
            step_size = 0.15 * 0.01
            reached = True
            for i in range(12):
                delta = stand_goal[i] - target[i]
                if abs(delta) > 1e-6:
                    target[i] += max(-step_size, min(step_size, float(delta)))
                    reached = False
            stream["target"] = tuple(float(v) for v in target)
            if reached:
                break
            time.sleep(0.01)
        time.sleep(1.0)

        phase = "restore"
        trace_t0[0] = time.monotonic()
        trace_stop.clear()
        sampler[0] = threading.Thread(target=sample_trace, daemon=True)
        sampler[0].start()
        ok = restore("ai")
        report["owner_after_restore"] = owner(1)
        if not ok and report["owner_after_restore"] != "ai":
            raise Fault("restore_failed")
        time.sleep(3.0)
        report["status"] = "completed"
    except Fault as problem:
        report["status"] = "fault"
        report["fault"] = str(problem)
        report["phase"] = phase
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["phase"] = phase
    finally:
        # After a normal run, keep holding until ai takes over.
        # Stopping the stream before handoff can let the robot collapse.
        if report["status"] != "completed":
            try:
                stream["target"] = None
                for _ in range(25):
                    publisher.Write(build([0.0] * 12, gains=False))
                    time.sleep(0.01)
            except Exception as error:
                report["damping_error"] = f"{type(error).__name__}: {error}"
        handback_end = time.monotonic() + 12
        while time.monotonic() < handback_end:
            if owner(1) == "ai":
                break
            try:
                switcher.SelectMode("ai")
            except Exception:
                pass
            time.sleep(0.5)
        report["recovery_owner"] = owner(1)
        trace_stop.set()
        if sampler[0] is not None:
            sampler[0].join(1.0)
        report["handback_trace"] = decimate(trace)
        stream["stop"].set()
        if thread.is_alive():
            thread.join(0.8)
        report["frames"] = stream["sent"]
        report["elapsed_s"] = round(time.monotonic() - started, 2)
        try:
            publisher.Close()
        except Exception:
            pass
        for sub in subs:
            try:
                sub.Close()
            except Exception:
                pass
        print(json.dumps(report, indent=2))
        if args.report:
            with open(args.report, "w") as dump:
                json.dump(report, dump, indent=2)
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
