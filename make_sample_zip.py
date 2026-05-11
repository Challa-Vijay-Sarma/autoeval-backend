#!/usr/bin/env python3
"""Build a tiny sample.zip for testing the platform end-to-end.

Picks one passed trial and one failed/no-reward trial from jobs_new/<run>/
and packs them into Golden_trajectories/ and Failure_trajectories/.

Usage:
    python make_sample_zip.py jobs_new/2026-05-08__16-23-19/
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path


def trial_passed(trial_dir: Path) -> bool:
    reward = trial_dir / "verifier" / "reward.txt"
    return reward.exists() and reward.read_text().strip() == "1"


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python make_sample_zip.py jobs_new/<run>/\n")
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        sys.stderr.write(f"not a directory: {run_dir}\n")
        sys.exit(2)

    trials = [p for p in run_dir.iterdir() if p.is_dir() and (p / "agent" / "trajectory.json").exists()]
    golden = next((t for t in trials if trial_passed(t)), None)
    failure = next((t for t in trials if not trial_passed(t)), None)

    if not golden and not failure:
        sys.stderr.write("no trials with agent/trajectory.json found\n")
        sys.exit(2)

    out_zip = Path("sample.zip").resolve()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for trial, category in [(golden, "Golden_trajectories"), (failure, "Failure_trajectories")]:
            if not trial:
                continue
            for path in trial.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(trial)
                    arc = f"sample/{category}/{trial.name}/{rel}"
                    zf.write(path, arc)
            print(f"  packed {trial.name} -> {category}/")

    print(f"\nWrote {out_zip}  ({out_zip.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
