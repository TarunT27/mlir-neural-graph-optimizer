from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAXIMUM_OUTPUT_BYTES = 64 * 1024
TIMEOUT_SECONDS = 10


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reproducer execution is disabled")
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        result = subprocess.run(
            [str(arguments.driver), "--run-reproducer", str(arguments.input)],
            capture_output=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write("driver timed out while rejecting --run-reproducer\n")
        return 1

    output = result.stdout + result.stderr
    if len(output) > MAXIMUM_OUTPUT_BYTES:
        sys.stderr.write("driver produced excessive output\n")
        return 1
    decoded = output.decode("utf-8", errors="replace")
    if result.returncode == 0:
        sys.stderr.write("driver unexpectedly accepted --run-reproducer\n")
        return 1
    if "--run-reproducer is disabled" not in decoded:
        sys.stderr.write(f"driver failed for the wrong reason:\n{decoded}")
        return 1
    print("PASS: reproducer execution is disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
