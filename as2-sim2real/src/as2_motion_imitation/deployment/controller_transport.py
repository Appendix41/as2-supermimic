"""SDK transport for hardware control; construct only when actuation is authorized."""

from threading import Event, Lock, Thread
import time

from .controller import Command, ControlFault, Snapshot, encode_command
from .feedback_crc import check_crc
from .remote_stop import decode_remote
from .telemetry import record_message


class SdkTransport:
    def __init__(self, interface, config, bundle):
        config.require_hardware(bundle)
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
        )
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MainBoardState_
        from unitree_sdk2py.utils.crc import CRC

        self._lock = Lock()
        self._low = None
        self._crc_valid = False
        self._board_ns, self._board = 0, ()
        self._fault = None
        self._watch = False
        self._expected_machine = config.expected_machine
        self._last_tick = None
        self._owner_ns = None
        self._owner_thread = None
        self._closing = Event()
        self.readers = []
        self.publisher = None
        self.factory, self.crc = unitree_hg_msg_dds__LowCmd_, CRC()
        try:
            ChannelFactoryInitialize(0, interface)
            for topic, kind, callback in (
                ("rt/lowstate", LowState_, self._on_low),
                ("rt/lf/mainboardstate", MainBoardState_, self._on_board),
            ):
                reader = ChannelSubscriber(topic, kind)
                self.readers.append(reader)
                # Direct callbacks latch brief stop events before replacing latest state.
                reader.Init(callback, 0)
            self.switcher = MotionSwitcherClient()
            self.switcher.SetTimeout(0.2)
            self.switcher.Init()
            self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.publisher.Init()
        except BaseException:
            self.close()
            raise

    def _latch(self, reason):
        if self._fault is None:
            self._fault = reason

    def _on_low(self, message):
        now = time.monotonic_ns()
        try:
            valid = check_crc(message, self.crc)["valid"]
            row = record_message("rt/lowstate", message, now)
            row["motor_slots"] = row["motor_slots"][:12]
            for motor, raw in zip(row["motor_slots"], message.motor_state[:12]):
                motor.update(temperature_raw=list(raw.temperature),
                             voltage_raw=float(raw.vol), warning=int(raw.reserve[3]))
            remote = decode_remote(message.wireless_remote)
            with self._lock:
                self._low, self._crc_valid = row, valid
                if not valid:
                    self._latch("transport_crc_error")
                if self._watch:
                    if (row["mode_machine"] != self._expected_machine or row["mode_pr"] != 0):
                        self._latch("transport_mode_changed")
                    if self._last_tick is not None and (row["tick"] - self._last_tick) % 2 ** 32 >= 2 ** 31:
                        self._latch("transport_tick_replayed")
                    if not remote["header_matches_recorded_fixture"]:
                        self._latch("transport_remote_format")
                    elif remote["l2_b"]:
                        self._latch("remote_stop")
                    elif remote["button_mask"]:
                        self._latch("remote_intervention")
                    if any(m["status"] or m["warning"] for m in row["motor_slots"]):
                        self._latch("transport_motor_fault")
                if valid:
                    self._last_tick = row["tick"]
        except Exception:
            with self._lock:
                self._crc_valid = False
                self._latch("transport_lowstate_decode")

    def _on_board(self, message):
        now = time.monotonic_ns()
        try:
            words = tuple(int(v) for v in message.state)
            with self._lock:
                self._board_ns, self._board = now, words
                if len(words) != 6 or any(words):
                    self._latch("transport_mainboard_fault")
        except Exception:
            with self._lock:
                self._latch("transport_mainboard_decode")

    def enable_watch(self):
        with self._lock:
            self._watch = True

    def snapshot(self):
        with self._lock:
            if self._low is None:
                return None
            if self._owner_ns is not None and time.monotonic_ns() - self._owner_ns > 350_000_000:
                self._latch("motion_owner_monitor_timeout")
            return Snapshot(self._low, self._crc_valid, self._board_ns, self._board, self._fault)

    def monitor_owner(self):
        self._owner_ns = time.monotonic_ns()
        self._owner_thread = Thread(target=self._check_owner_loop, daemon=True)
        self._owner_thread.start()

    def _check_owner_loop(self):
        while not self._closing.wait(0.1):
            try:
                code, mode = self.check_mode()
                with self._lock:
                    self._owner_ns = time.monotonic_ns()
                    if code != 0 or not isinstance(mode, dict) or mode.get("name") != "":
                        self._latch("motion_owner_lost")
            except Exception:
                with self._lock:
                    self._latch("motion_owner_query_failed")
                return

    def check_mode(self):
        return self.switcher.CheckMode()

    def release_mode(self):
        return self.switcher.ReleaseMode()

    def publish(self, command):
        if not isinstance(command, Command):
            raise ControlFault("invalid_transport_command")
        with self._lock:
            if command.kind == "position" and self._fault is not None:
                raise ControlFault(self._fault)
        # No timeout=0: in this pinned SDK it always returns False. A True
        # result means local DDS write acceptance, not receipt at a motor.
        return self.publisher.Write(encode_command(command, self.factory, self.crc))

    def close(self):
        self._closing.set()
        if self._owner_thread is not None:
            self._owner_thread.join(1)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        self.readers = []
        if self.publisher is not None:
            try:
                self.publisher.Close()
            except Exception:
                pass
            self.publisher = None
