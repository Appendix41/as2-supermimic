"""Separate inference process. This module never imports a robot SDK."""

import multiprocessing as mp
import time

from .controller import ControlFault, Reply, Request
from .runtime import InferenceBundle


def actor_process(connection, bundle_path):
    try:
        bundle = InferenceBundle(bundle_path)
        connection.send(("ready", bundle.manifest["checkpoint_sha256"]))
        expected = 0
        while True:
            request = connection.recv()
            if request is None:
                return
            if (not isinstance(request, Request) or request.sequence != expected
                    or request.step != expected or not 0 <= expected < bundle.steps):
                raise ControlFault("actor_request_sequence")
            action = bundle.act(request.observation)
            connection.send(Reply(request.sequence, request.step, tuple(float(v) for v in action)))
            expected += 1
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(Reply(-1, -1, (), f"{type(error).__name__}: {error}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ActorWorker:
    def __init__(self, bundle_path, checkpoint_sha256):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=actor_process, args=(child, str(bundle_path)))
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(15):
                raise ControlFault("actor_startup_timeout")
            if self.connection.recv() != ("ready", checkpoint_sha256):
                raise ControlFault("actor_startup_contract")
        except BaseException:
            child.close()
            self.close()
            raise

    def submit(self, request):
        if not self.process.is_alive():
            raise ControlFault("actor_process_dead")
        self.connection.send(request)

    def poll(self):
        if self.connection.poll():
            try:
                result = self.connection.recv()
            except (EOFError, OSError) as error:
                raise ControlFault("actor_connection_lost") from error
            if not isinstance(result, Reply):
                raise ControlFault("invalid_actor_reply")
            return result
        if not self.process.is_alive():
            raise ControlFault("actor_process_dead")
        return None

    def close(self):
        if self.process.pid is not None:
            if self.process.is_alive():
                try:
                    self.connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
                self.process.join(0.5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(1)
            else:
                self.process.join()
        self.connection.close()


def wait_for_snapshot(transport, clock=time.monotonic_ns, timeout_s=3.0):
    start = clock()
    while clock() - start < timeout_s * 1e9:
        snapshot = transport.snapshot()
        if snapshot is not None and snapshot.board_ns:
            return snapshot
        time.sleep(0.002)
    raise ControlFault("feedback_startup_timeout")
