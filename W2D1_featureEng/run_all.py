"""
W2D1: Feature Engineering and Encoding

Purpose:
This program runs encoding.py, scaler.py, and
feature_selection.py in sequence.

It displays the output of each program and reports
whether the execution was successful.
"""


import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

PROGRAMS = [
    "encoding.py",
    "scaler.py",
    "feature_selection.py",
]


def run_program(program):
    """Run a Python program and report its status."""

    program_path = PROJECT_DIR / program

    if not program_path.exists():
        print(f"ERROR: {program} was not found.")
        return False

    print(f"\n{'=' * 50}")
    print(f"Running {program}")
    print("=" * 50)

    result = subprocess.run(
        [sys.executable, str(program_path)],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"FAILED: {program}")
        return False

    print(f"SUCCESS: {program}")
    return True


def main():
    failed_programs = []

    for program in PROGRAMS:
        if not run_program(program):
            failed_programs.append(program)

    print(f"\n{'=' * 50}")
    print("PIPELINE SUMMARY")
    print("=" * 50)

    if failed_programs:
        print("Programs requiring attention:")
        for program in failed_programs:
            print(f"- {program}")

        sys.exit(1)

    print("All programs completed successfully!")


if __name__ == "__main__":
    main()