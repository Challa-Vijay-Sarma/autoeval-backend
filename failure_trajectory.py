#!/usr/bin/env python3
"""Failure Trajectory Analyzer.

Walks a jobs_new/<run> directory, identifies failed trials (verifier/reward
!= "1" or exception present), and classifies each failure with an OpenAI
chat-completions model into one of nine failure categories.

Usage:
    python failure_trajectory.py jobs_new/2026-05-08__16-23-19/
    python failure_trajectory.py jobs_new/2026-05-08__16-23-19/ --limit 1
    python failure_trajectory.py jobs_new/2026-05-08__16-23-19/ --model gpt-5.1

Environment:
    OPENAI_API_KEY must be set (loaded from .env automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    sys.stderr.write("ERROR: python-dotenv is not installed. pip install -r requirements.txt\n")
    sys.exit(2)

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write("ERROR: openai is not installed. pip install -r requirements.txt\n")
    sys.exit(2)

try:
    import pandas as pd
except ImportError:
    sys.stderr.write("ERROR: pandas is not installed. pip install -r requirements.txt\n")
    sys.exit(2)


MAX_WORKERS = 6
RETRIES = 5
BASE_SLEEP = 2
BATCH_SIZE = 50
DEFAULT_MODEL = "gpt-5.1"


def task_hint(task: str) -> str:
    mapping = {
        "dependency_lockfile_issues": "Fix dependency or package issues",
        "dns-resolution-chain-debugging": "Debug DNS resolution chain",
        "sqlite-fs-indexer-lockswap": "Fix concurrency or locking issues",
    }
    return mapping.get(task, "General debugging task")


def extract_failures_only(text: str) -> str:
    match = re.search(
        r"=+ FAILURES =+(.*?)(=+ PASSES =+|=+ short test summary info =+)",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:1000]
    return ""


def read_file_safe(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_trial_meta(trial_dir: Path) -> dict[str, Any]:
    """Pull agent name + benchmark/task name from result.json or config.json."""
    meta = {"agent": trial_dir.parent.name, "benchmark": trial_dir.name}
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            r = json.loads(result_path.read_text(encoding="utf-8"))
            ai = r.get("agent_info") or {}
            if ai.get("name"):
                meta["agent"] = ai["name"]
            if r.get("task_name"):
                meta["benchmark"] = r["task_name"]
        except (json.JSONDecodeError, OSError):
            pass
    return meta


def load_trials(base_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_file in base_dir.rglob("result.json"):
        try:
            res = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            res = {}

        trial_dir = result_file.parent
        if trial_dir == base_dir:
            continue
        meta = load_trial_meta(trial_dir)

        trajectory: dict[str, Any] = {}
        traj_path = trial_dir / "agent" / "trajectory.json"
        if traj_path.exists():
            try:
                trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                trajectory = {}

        verifier_dir = trial_dir / "verifier"
        status = "unknown"
        reward_file = verifier_dir / "reward.txt"
        if reward_file.exists():
            val = reward_file.read_text(encoding="utf-8", errors="replace").strip()
            status = "pass" if val == "1" else "fail"
        elif res.get("exception_info"):
            status = "fail"

        failure_output = ""
        stdout_file = verifier_dir / "test-stdout.txt"
        if stdout_file.exists():
            failure_output = extract_failures_only(
                stdout_file.read_text(encoding="utf-8", errors="replace")
            )

        rows.append({
            "agent": meta["agent"],
            "benchmark": meta["benchmark"],
            "trial_id": trial_dir.name,
            "trial_dir": str(trial_dir),
            "log": read_file_safe(trial_dir / "trial.log"),
            "exception": read_file_safe(trial_dir / "exception.txt") or (res.get("exception_info") or {}).get("exception_message", ""),
            "trajectory": trajectory,
            "status": status,
            "failure_output": failure_output,
        })
    return pd.DataFrame(rows)


def extract_user_input(traj_steps: list[dict[str, Any]], max_len: int = 600) -> str:
    user_msgs = [s.get("message", "") for s in traj_steps if s.get("source") == "user"]
    if not user_msgs:
        return ""
    text = user_msgs[0]
    sections = []
    for p in (
        r"Task Description:(.*?)(?:\n\n|\Z)",
        r"Expected outputs:(.*?)(?:\n\n|\Z)",
        r"Rules:(.*?)(?:\n\n|\Z)",
    ):
        m = re.search(p, text, re.DOTALL)
        if m:
            sections.append(m.group(0).strip())
    if not sections:
        return text[:max_len]
    return "\n\n".join(sections)[:max_len]


def build_context(row: dict[str, Any]) -> dict[str, Any]:
    traj = row.get("trajectory") or {}
    traj_steps = traj.get("steps", []) if isinstance(traj, dict) else []

    user_input = extract_user_input(traj_steps)
    agent_steps = [s for s in traj_steps if s.get("source") == "agent"]

    actions, observations, thoughts = [], [], []
    for s in agent_steps[:5]:
        msg = s.get("message", "")
        if msg:
            thoughts.append(msg[:300])
        for t in s.get("tool_calls", []) or []:
            cmd = (t.get("arguments") or {}).get("keystrokes", "")
            if cmd:
                actions.append(cmd.strip())
        for r in (s.get("observation") or {}).get("results", []) or []:
            content = r.get("content", "")
            if content:
                observations.append(content[:300])

    early_failure = not actions and not observations
    return {
        "agent": row["agent"],
        "task_name": row["benchmark"],
        "task_hint": task_hint(row["benchmark"]),
        "user_input": user_input,
        "failure_output": row.get("failure_output", ""),
        "thoughts": thoughts[:3],
        "actions": actions[:5],
        "observations": observations[:5],
        "exception": row["exception"],
        "early_failure": early_failure,
    }


FAILURE_PROMPT_TEMPLATE = """
You are analyzing an AI agent failure.

Classify the failure into ONE primary category from below:

1. Disobey Specification
2. Step Repetition
3. Unaware of termination conditions
4. Reasoning-Action Mismatch
5. Context Loss
6. Task Derailment
7. Premature termination
8. No or incorrect Verification
9. Weak Verification

Definitions:

- Disobey Specification: Contradicts explicit task instructions or constraints
- Step Repetition: Repeats same action/step without strategy change
- Unaware of termination: Continues after success or futility
- Reasoning-Action Mismatch: Claims contradict actual logs/actions
- Context Loss: Forgets or contradicts prior context
- Task Derailment: Deviates from main objective
- Premature termination: Stops before completing task properly
- No/incorrect Verification: Skips or bypasses required checks
- Weak Verification: Uses incomplete or incorrect validation

Now analyze:

Agent: {agent}
Task: {task_name}
Task Description: {task_hint}

User Input:
{user_input}

Actions:
{actions}

Observations:
{observations}

Test Failure Output:
{failure_output}

Exception:
{exception}

Early Failure:
{early_failure}

Return ONLY JSON:
{{
  "failure_category": "...",
  "secondary_category": "... or null",
  "root_cause": "...",
  "reason": "...",
  "fix": "..."
}}
"""


def build_context_from_trajectory(
    trajectory: dict[str, Any],
    *,
    agent_name: str,
    task: str,
    failure_output: str = "",
    exception: str = "",
) -> dict[str, Any]:
    """Library entrypoint: build the LLM context from an in-memory trajectory dict."""
    return build_context({
        "agent": agent_name,
        "benchmark": task,
        "trajectory": trajectory,
        "failure_output": failure_output,
        "exception": exception,
    })


def classify_failure(client: OpenAI, context: dict[str, Any], model: str) -> dict[str, Any]:
    """Library alias for explain_failure — same behavior, clearer name."""
    return explain_failure(client, context, model)


def explain_failure(client: OpenAI, context: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = FAILURE_PROMPT_TEMPLATE.format(
        agent=context.get("agent", ""),
        task_name=context.get("task_name", ""),
        task_hint=context.get("task_hint", ""),
        user_input=context.get("user_input", ""),
        actions=context.get("actions", []),
        observations=context.get("observations", []),
        failure_output=context.get("failure_output", ""),
        exception=context.get("exception", ""),
        early_failure=context.get("early_failure", False),
    )
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            last_err = e
            time.sleep(BASE_SLEEP * (2 ** attempt))
    return {
        "failure_category": "failed",
        "secondary_category": None,
        "root_cause": "",
        "reason": f"API failed: {last_err}" if last_err else "API failed",
        "fix": "",
    }


def run_parallel(client: OpenAI, df: pd.DataFrame, model: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(df)

    def process(i: int, row: pd.Series) -> tuple[int, dict[str, Any]]:
        return i, explain_failure(client, row["context"], model)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process, i, row) for i, (_, row) in enumerate(df.iterrows())
        ]
        for f in as_completed(futures):
            i, res = f.result()
            results[i] = res
    return [r if r is not None else {} for r in results]


def write_per_trial_json(df: pd.DataFrame, outputs: list[dict[str, Any]]) -> None:
    for (_, row), out in zip(df.iterrows(), outputs):
        trial_dir = Path(row["trial_dir"])
        (trial_dir / "eval_failure.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify failed-trial trajectories with OpenAI.")
    ap.add_argument("base_dir", type=Path, help="Path to jobs_new/<run> directory.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N failed trials.")
    args = ap.parse_args()

    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("ERROR: OPENAI_API_KEY is not set in the environment or .env.\n")
        sys.exit(2)

    if not args.base_dir.is_dir():
        sys.stderr.write(f"ERROR: not a directory: {args.base_dir}\n")
        sys.exit(2)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Scanning {args.base_dir} ...")
    df = load_trials(args.base_dir)
    print(f"Loaded {len(df)} trial records.")

    failure_analysis = df[df["status"] == "fail"].copy().reset_index(drop=True)
    print(f"Failed trials: {len(failure_analysis)}")
    if args.limit is not None:
        failure_analysis = failure_analysis.head(args.limit).copy().reset_index(drop=True)
        print(f"Limited to first {len(failure_analysis)}")

    if failure_analysis.empty:
        print("No failed trials to analyze.")
        return

    failure_analysis["context"] = failure_analysis.apply(build_context, axis=1)

    all_outputs: list[dict[str, Any]] = []
    summary_path = args.base_dir / "failure_summary.xlsx"

    for start in range(0, len(failure_analysis), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = failure_analysis.iloc[start:end]
        print(f"Processing batch {start} -> {end}")
        batch_outputs = run_parallel(client, batch, args.model)
        all_outputs.extend(batch_outputs)
        write_per_trial_json(batch, batch_outputs)

        temp_df = failure_analysis.iloc[: len(all_outputs)].copy()
        temp_df["llm_output"] = all_outputs
        _write_summary(temp_df, summary_path)

    failure_analysis["llm_output"] = all_outputs
    failure_analysis["failure_type"] = failure_analysis["llm_output"].apply(
        lambda x: x.get("failure_category")
    )
    failure_analysis["reason"] = failure_analysis["llm_output"].apply(lambda x: x.get("reason"))
    failure_analysis["root_cause"] = failure_analysis["llm_output"].apply(lambda x: x.get("root_cause"))
    failure_analysis["fix"] = failure_analysis["llm_output"].apply(lambda x: x.get("fix"))

    _write_summary(failure_analysis, summary_path)
    print(f"\nDONE. Wrote summary: {summary_path}")


def _write_summary(df: pd.DataFrame, out_path: Path) -> None:
    columns = [
        "agent", "benchmark", "trial_id", "log", "exception", "trajectory",
        "status", "failure_output", "context", "llm_output",
        "failure_type", "reason", "root_cause", "fix",
    ]
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    # Serialize dict/list cells so Excel can hold them
    for col in ("trajectory", "context", "llm_output"):
        out[col] = out[col].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        )
    out[columns].to_excel(out_path, index=False)


if __name__ == "__main__":
    main()
