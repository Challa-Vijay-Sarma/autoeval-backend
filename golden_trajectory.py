#!/usr/bin/env python3
"""Golden Trajectory Evaluator (OpenAI port).

Classifies an agent trajectory against the ByteDance Golden Trajectory
rubric (GT1-GT5 + SC1-SC4) using an OpenAI chat-completions model.

Usage:
    # Single trajectory
    python golden_trajectory.py path/to/trajectory.json
    python golden_trajectory.py path/to/trajectory.json --pretty
    python golden_trajectory.py path/to/trajectory.json --raw

    # Batch over a jobs_new/<run>/ directory; walks */agent/trajectory.json
    python golden_trajectory.py jobs_new/2026-05-08__16-23-19/

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
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    sys.stderr.write(
        "ERROR: python-dotenv is not installed. pip install -r requirements.txt\n"
    )
    sys.exit(2)

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write(
        "ERROR: openai is not installed. pip install -r requirements.txt\n"
    )
    sys.exit(2)


HERE = Path(__file__).resolve().parent
KB_PATH = HERE / "knowledge_base.md"
SP_PATH = HERE / "system_prompt.md"

DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_TOKENS = 16000
RETRIES = 5
BASE_SLEEP = 2

ALLOWED_GT = {"GT1", "GT2", "GT3", "GT4", "GT5"}
ALLOWED_VERDICTS = {"PASS", "FAIL", "WEAK PASS", "NA"}
ALLOWED_FINAL = {"ACCEPT as Golden Trajectory", "REJECT", "BORDERLINE"}


def read_text(path: Path) -> str:
    if not path.exists():
        sys.stderr.write(f"ERROR: required file not found: {path}\n")
        sys.exit(2)
    return path.read_text(encoding="utf-8")


def load_trajectory(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.stderr.write(f"ERROR: trajectory not found: {path}\n")
        sys.exit(2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"ERROR: trajectory is not valid JSON: {e}\n")
        sys.exit(2)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from the model's reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start : i + 1])
    raise ValueError("no top-level JSON object found in model reply")


def validate_result(obj: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    def _need(path: str, parent: dict, key: str) -> Any:
        if key not in parent:
            warnings.append(f"missing field: {path}")
            return None
        return parent[key]

    _need("agent", obj, "agent")
    _need("model", obj, "model")
    _need("step_count", obj, "step_count")
    _need("tool_call_count", obj, "tool_call_count")

    gt = _need("gt_class", obj, "gt_class") or {}
    label = gt.get("label")
    if label not in ALLOWED_GT:
        warnings.append(f"gt_class.label not in {sorted(ALLOWED_GT)}: {label!r}")

    sc = _need("success_criteria", obj, "success_criteria") or {}
    for name in ("SC1", "SC2", "SC3", "SC4"):
        entry = sc.get(name)
        if not isinstance(entry, dict):
            warnings.append(f"success_criteria.{name} missing or malformed")
            continue
        v = entry.get("verdict")
        if v not in ALLOWED_VERDICTS:
            warnings.append(
                f"success_criteria.{name}.verdict not in {sorted(ALLOWED_VERDICTS)}: {v!r}"
            )

    hr = _need("hard_requirements", obj, "hard_requirements") or {}
    for key in ("min_10_tool_calls", "planning_thinking_present", "cross_file_or_multi_module"):
        if not isinstance(hr.get(key), bool):
            warnings.append(f"hard_requirements.{key} must be a boolean")

    final = obj.get("verdict")
    if final not in ALLOWED_FINAL:
        warnings.append(f"verdict not in {sorted(ALLOWED_FINAL)}: {final!r}")

    return warnings


def count_agent_steps_and_tool_calls(traj: dict[str, Any]) -> tuple[int, int]:
    steps = traj.get("steps", [])
    agent_steps = 0
    tool_calls = 0
    for s in steps:
        if s.get("source") == "agent":
            agent_steps += 1
            tool_calls += len(s.get("tool_calls", []) or [])
    return agent_steps, tool_calls


def format_pretty(result: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"Agent:        {result.get('agent')}")
    out.append(f"Model:        {result.get('model')}")
    out.append(f"Steps:        {result.get('step_count')}")
    out.append(f"Tool calls:   {result.get('tool_call_count')}")
    out.append("")

    gt = result.get("gt_class") or {}
    out.append(f"GT Class:     {gt.get('label')}")
    if gt.get("justification"):
        out.append(f"  why: {gt['justification']}")
    for pm in gt.get("partial_matches") or []:
        out.append(f"  partial {pm.get('label')}: {pm.get('note')}")
    out.append("")

    sc = result.get("success_criteria") or {}
    for name in ("SC1", "SC2", "SC3", "SC4"):
        entry = sc.get(name) or {}
        v = entry.get("verdict", "?")
        r = entry.get("reason", "")
        out.append(f"{name}: {v}")
        if r:
            out.append(f"  {r}")
    out.append("")

    hr = result.get("hard_requirements") or {}
    out.append("Hard requirements:")
    out.append(f"  >=10 tool calls:                {hr.get('min_10_tool_calls')}")
    out.append(f"  planning/thinking present:      {hr.get('planning_thinking_present')}")
    out.append(f"  cross-file / multi-module:      {hr.get('cross_file_or_multi_module')}")
    if hr.get("notes"):
        out.append(f"  notes: {hr['notes']}")
    out.append("")

    out.append(f"Verdict:      {result.get('verdict')}")
    if result.get("verdict_reason"):
        out.append(f"  {result['verdict_reason']}")

    return "\n".join(out)


def build_user_prompt(kb: str, traj_json_str: str, measured: tuple[int, int]) -> str:
    agent_steps, tool_calls = measured
    return (
        "# Knowledge Base (authoritative definitions)\n\n"
        f"{kb}\n\n"
        "# Trajectory under evaluation\n\n"
        "The trajectory JSON follows. Your pre-computed measurements (use these\n"
        "to fill step_count and tool_call_count; they were counted by the\n"
        "wrapper script, not inferred by you):\n\n"
        f"- step_count (agent-action steps): {agent_steps}\n"
        f"- tool_call_count (sum over tool_calls arrays): {tool_calls}\n\n"
        "```json\n"
        f"{traj_json_str}\n"
        "```\n\n"
        "Now produce ONLY the JSON object specified in the system prompt. No\n"
        "prose before or after. Ground every claim in specific step indices\n"
        "from the trajectory above.\n"
    )


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(BASE_SLEEP * (2 ** attempt))
    raise RuntimeError(f"OpenAI call failed after {RETRIES} attempts: {last_err}")


def evaluate_trajectory_dict(
    client: OpenAI,
    traj: dict[str, Any],
    system_prompt: str,
    kb: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Evaluate an in-memory trajectory dict. Platform path (no filesystem)."""
    measured = count_agent_steps_and_tool_calls(traj)
    traj_json_str = json.dumps(traj, indent=2, ensure_ascii=False)
    user_prompt = build_user_prompt(kb, traj_json_str, measured)
    raw_text = call_model(client, model, system_prompt, user_prompt, max_tokens)
    if not raw_text.strip():
        raise RuntimeError("model returned empty response")
    return extract_json_object(raw_text)


def evaluate_one(
    client: OpenAI,
    traj_path: Path,
    system_prompt: str,
    kb: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Evaluate a trajectory loaded from disk. CLI path."""
    traj = load_trajectory(traj_path)
    return evaluate_trajectory_dict(client, traj, system_prompt, kb, model, max_tokens)


def load_trial_meta(trial_dir: Path) -> dict[str, Any]:
    """Pull task / agent / model metadata from result.json or config.json."""
    meta: dict[str, Any] = {"task": "", "agent": "", "model": "", "trial_name": trial_dir.name}
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            r = json.loads(result_path.read_text(encoding="utf-8"))
            meta["task"] = r.get("task_name") or ""
            meta["trial_name"] = r.get("trial_name") or trial_dir.name
            ai = r.get("agent_info") or {}
            meta["agent"] = ai.get("name") or ""
            mi = ai.get("model_info") or {}
            meta["model"] = mi.get("name") or ""
        except (json.JSONDecodeError, OSError):
            pass
    if not meta["agent"] or not meta["model"]:
        config_path = trial_dir / "config.json"
        if config_path.exists():
            try:
                c = json.loads(config_path.read_text(encoding="utf-8"))
                ag = c.get("agent") or {}
                meta["agent"] = meta["agent"] or ag.get("name") or ""
                meta["model"] = meta["model"] or ag.get("model_name") or ""
            except (json.JSONDecodeError, OSError):
                pass
    return meta


def format_sc_summary(result: dict[str, Any]) -> str:
    sc = result.get("success_criteria") or {}
    parts = []
    for name in ("SC1", "SC2", "SC3", "SC4"):
        v = (sc.get(name) or {}).get("verdict", "?")
        parts.append(f"{name}={v}")
    return "; ".join(parts)


def summary_row(trial_dir: Path, result: dict[str, Any], traj_path: Path) -> dict[str, Any]:
    meta = load_trial_meta(trial_dir)
    gt = result.get("gt_class") or {}
    task_name = (meta["task"].split("/")[-1] if meta["task"] else trial_dir.name)
    return {
        "Task": meta["task"] or task_name,
        "Agent": meta["agent"] or result.get("agent") or "",
        "Model": meta["model"] or result.get("model") or "",
        "GT Class(AI)": gt.get("label") or "",
        "Success Criteria (AI)": format_sc_summary(result),
        "GT Class(Human)": "",
        "Success Criteria (Human)": "",
        "HITL Remarks": "",
        "Task name": task_name,
        "Trajectory name": str(traj_path),
    }


def trial_passed(trial_dir: Path) -> bool:
    """A trial counts as passed iff verifier/reward.txt exists and equals '1'."""
    reward_file = trial_dir / "verifier" / "reward.txt"
    if not reward_file.exists():
        return False
    try:
        return reward_file.read_text(encoding="utf-8", errors="replace").strip() == "1"
    except OSError:
        return False


def run_batch(
    client: OpenAI,
    base_dir: Path,
    system_prompt: str,
    kb: str,
    model: str,
    max_tokens: int,
) -> None:
    import pandas as pd

    all_traj_paths = sorted(base_dir.rglob("agent/trajectory.json"))
    if not all_traj_paths:
        sys.stderr.write(f"ERROR: no agent/trajectory.json files under {base_dir}\n")
        sys.exit(2)

    traj_paths = [p for p in all_traj_paths if trial_passed(p.parent.parent)]
    skipped = len(all_traj_paths) - len(traj_paths)
    if not traj_paths:
        sys.stderr.write(
            f"ERROR: no passed trials under {base_dir} "
            f"(checked {len(all_traj_paths)} trajectories; none have verifier/reward.txt == '1').\n"
        )
        sys.exit(2)
    print(f"Found {len(all_traj_paths)} trajectories; evaluating {len(traj_paths)} passed, skipping {skipped} non-passed.")
    rows: list[dict[str, Any]] = []
    columns = [
        "Task", "Agent", "Model",
        "GT Class(AI)", "Success Criteria (AI)",
        "GT Class(Human)", "Success Criteria (Human)",
        "HITL Remarks", "Task name", "Trajectory name",
    ]
    for i, traj_path in enumerate(traj_paths, 1):
        trial_dir = traj_path.parent.parent
        trial_id = trial_dir.name
        print(f"[{i}/{len(traj_paths)}] {trial_id}")
        try:
            result = evaluate_one(client, traj_path, system_prompt, kb, model, max_tokens)
        except Exception as e:
            sys.stderr.write(f"  FAILED: {e}\n")
            meta = load_trial_meta(trial_dir)
            rows.append({
                "Task": meta["task"], "Agent": meta["agent"], "Model": meta["model"],
                "GT Class(AI)": "ERROR", "Success Criteria (AI)": str(e),
                "GT Class(Human)": "", "Success Criteria (Human)": "",
                "HITL Remarks": "", "Task name": trial_id, "Trajectory name": str(traj_path),
            })
            continue
        warnings = validate_result(result)
        for w in warnings:
            sys.stderr.write(f"  WARN: {w}\n")

        out_path = traj_path.parent / "eval_golden.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(summary_row(trial_dir, result, traj_path))

    df = pd.DataFrame(rows, columns=columns)
    csv_path = base_dir / "golden_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote summary CSV: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate agent trajectories against the ByteDance GT rubric using OpenAI."
    )
    ap.add_argument(
        "input",
        type=Path,
        help="Path to a trajectory.json (single mode) OR a directory to walk (batch mode).",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Response max_tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    ap.add_argument("--raw", action="store_true", help="Single mode: print raw JSON only.")
    ap.add_argument("--pretty", action="store_true", help="Single mode: print pretty summary only.")
    args = ap.parse_args()

    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        sys.stderr.write("ERROR: OPENAI_API_KEY is not set in the environment or .env.\n")
        sys.exit(2)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = read_text(SP_PATH)
    kb = read_text(KB_PATH)

    if args.input.is_dir():
        run_batch(client, args.input, system_prompt, kb, args.model, args.max_tokens)
        return

    result = evaluate_one(client, args.input, system_prompt, kb, args.model, args.max_tokens)
    warnings = validate_result(result)
    for w in warnings:
        sys.stderr.write(f"WARN: {w}\n")

    out_path = args.input.parent / "eval_golden.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.raw:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.pretty:
        print(format_pretty(result))
        return
    print(format_pretty(result))
    print()
    print("--- raw JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
