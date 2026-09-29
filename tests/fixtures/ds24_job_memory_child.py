from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MIB = 1024**2
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sleep() -> int:
    while True:
        time.sleep(1.0)


def _allocate_until_limited(chunk_mib: int) -> int:
    allocations: list[bytearray] = []
    try:
        while True:
            block = bytearray(chunk_mib * MIB)
            for offset in range(0, len(block), 4096):
                block[offset] = 1
            allocations.append(block)
            time.sleep(0.01)
    except MemoryError:
        return 75


def _spawn_descendant(pid_file: Path) -> int:
    child = subprocess.Popen(  # noqa: S603 - bounded test fixture
        [sys.executable, str(Path(__file__).resolve()), "--mode", "sleep"]
    )
    pid_file.write_text(str(child.pid), encoding="ascii")
    return _sleep()


def _own_job() -> int:
    from core.research.ml.ds24.clean_v2_resources import WindowsWorkerJob

    job = WindowsWorkerJob(commit_limit_bytes=256 * MIB)
    child = job.launch(
        [sys.executable, str(Path(__file__).resolve()), "--mode", "sleep"],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    print(json.dumps({"child_pid": child.pid}), flush=True)
    return _sleep()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sleep", "allocate", "spawn-descendant", "own-job"),
    )
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    if args.mode == "sleep":
        return _sleep()
    if args.mode == "allocate":
        return _allocate_until_limited(args.chunk_mib)
    if args.mode == "spawn-descendant":
        if args.pid_file is None:
            raise ValueError("--pid-file is required")
        return _spawn_descendant(args.pid_file)
    return _own_job()


if __name__ == "__main__":
    raise SystemExit(main())
