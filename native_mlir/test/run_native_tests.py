from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_FLAGS = frozenset(
    {
        "--ng-constant-fold",
        "--ng-canonicalize",
        "--ng-fuse-linear-relu",
        "--ng-dce",
        "--convert-ng-to-linalg",
    }
)
MAX_TESTS = 64
MAX_TEST_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
PROCESS_TIMEOUT_SECONDS = 20


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NG dialect FileCheck tests")
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--filecheck", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    return parser.parse_args()


def _run_line(test_file: Path) -> str:
    for line in test_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("// RUN: "):
            return line.removeprefix("// RUN: ").strip()
    raise ValueError(f"{test_file}: missing RUN directive")


def _driver_flags(run_line: str) -> tuple[bool, tuple[str, ...]]:
    left_side = run_line.split("|", 1)[0].strip().split()
    expects_failure = bool(left_side and left_side[0] == "not")
    command = left_side[1:] if expects_failure else left_side
    if not command or command[0] != "ngopt-native-opt" or "%s" not in command:
        raise ValueError(f"unsupported RUN directive: {run_line}")
    flags = tuple(item for item in command[1:] if item not in {"%s", "2>&1"})
    if any(flag not in ALLOWED_FLAGS for flag in flags):
        raise ValueError(f"RUN directive contains a disallowed flag: {run_line}")
    return expects_failure, flags


def _execute_bounded(command: list[str], *, input_text: str | None = None) -> tuple[int, str]:
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            process.communicate(input_text, timeout=PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise RuntimeError(
                f"process exceeded {PROCESS_TIMEOUT_SECONDS}s: {command[0]}"
            ) from error
        size = output.tell()
        if size > MAX_OUTPUT_BYTES:
            raise RuntimeError(f"process output exceeded {MAX_OUTPUT_BYTES} bytes: {command[0]}")
        output.seek(0)
        return process.returncode, output.read().decode("utf-8", errors="replace")


def _run_test(driver: Path, filecheck: Path, test_file: Path) -> str | None:
    expects_failure, flags = _driver_flags(_run_line(test_file))
    returncode, checked_text = _execute_bounded([str(driver), *flags, str(test_file)])
    if (returncode == 0) == expects_failure:
        expectation = "failure" if expects_failure else "success"
        return f"{test_file.name}: expected {expectation}, got exit {returncode}\n{checked_text}"
    check_returncode, check_output = _execute_bounded(
        [str(filecheck), str(test_file)], input_text=checked_text
    )
    if check_returncode:
        return f"{test_file.name}: FileCheck failed\n{check_output}\n--- output ---\n{checked_text}"
    return None


def main() -> int:
    arguments = _arguments()
    tests = tuple(sorted(arguments.test_dir.glob("*.mlir")))
    if len(tests) > MAX_TESTS:
        raise ValueError(f"test count exceeds {MAX_TESTS}")
    oversized = tuple(test for test in tests if test.stat().st_size > MAX_TEST_BYTES)
    if oversized:
        raise ValueError(f"test file exceeds {MAX_TEST_BYTES} bytes: {oversized[0]}")
    failures = tuple(
        failure
        for test in tests
        if (failure := _run_test(arguments.driver, arguments.filecheck, test)) is not None
    )
    if failures:
        sys.stderr.write("\n\n".join(failures) + "\n")
        return 1
    print(f"PASS: {len(tests)} native MLIR tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
