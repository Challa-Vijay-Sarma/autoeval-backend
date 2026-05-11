#!/usr/bin/env python3
"""Run both trajectory evaluators back-to-back over a jobs_new/<run> directory.

Usage:
    python run_all.py jobs_new/2026-05-08__16-23-19/
    python run_all.py jobs_new/2026-05-08__16-23-19/ --skip-golden
    python run_all.py jobs_new/2026-05-08__16-23-19/ --skip-failure
    python run_all.py jobs_new/2026-05-08__16-23-19/ --model gpt-5.1 --limit 5

Runs golden_trajectory.py first (evaluates every trajectory.json against the
ByteDance GT rubric), then failure_trajectory.py (classifies failed trials
into the 9-category failure taxonomy). Each child script is invoked as a
subprocess so its output streams live to the terminal and a crash in one
does not kill the other.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden_trajectory.py"
FAILURE = HERE / "failure_trajectory.py"


def run(cmd: list[str], label: str) -> int:
    print(f"\n========== {label} ==========")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd)
    print(f"========== {label} exit={proc.returncode} ==========")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run golden + failure evaluators in sequence.")
    ap.add_argument("base_dir", type=Path, help="Path to jobs_new/<run> directory.")
    ap.add_argument("--model", default=None, help="Override OpenAI model for both scripts.")
    ap.add_argument("--limit", type=int, default=None, help="Limit failed trials processed by failure_trajectory.py.")
    ap.add_argument("--skip-golden", action="store_true", help="Skip golden_trajectory.py.")
    ap.add_argument("--skip-failure", action="store_true", help="Skip failure_trajectory.py.")
    args = ap.parse_args()

    if not args.base_dir.is_dir():
        sys.stderr.write(f"ERROR: not a directory: {args.base_dir}\n")
        sys.exit(2)

    exit_codes: dict[str, int] = {}

    if not args.skip_golden:
        cmd = [sys.executable, str(GOLDEN), str(args.base_dir)]
        if args.model:
            cmd += ["--model", args.model]
        exit_codes["golden"] = run(cmd, "GOLDEN TRAJECTORY")

    if not args.skip_failure:
        cmd = [sys.executable, str(FAILURE), str(args.base_dir)]
        if args.model:
            cmd += ["--model", args.model]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        exit_codes["failure"] = run(cmd, "FAILURE TRAJECTORY")

    print("\n========== SUMMARY ==========")
    for name, code in exit_codes.items():
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {name}: {status}")

    if any(c != 0 for c in exit_codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
