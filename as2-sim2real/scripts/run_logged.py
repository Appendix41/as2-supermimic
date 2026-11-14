"""Run one bounded child command and retain the complete combined console."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.log.resolve().relative_to(root / "outputs")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    with args.log.open("x") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            sys.exit(child.wait())


if __name__ == "__main__":
    main()
