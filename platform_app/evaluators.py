"""Thin wrappers over the existing trajectory evaluators so the platform
doesn't depend on the CLI scripts' file-walking logic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path so we can import the existing CLI modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

import golden_trajectory as gt
import failure_trajectory as ft


# ----------------------------------------------------------------------------
# Prompt asset loading (cached)
# ----------------------------------------------------------------------------
_KB: str | None = None
_SP: str | None = None


def load_prompts() -> tuple[str, str]:
    global _KB, _SP
    if _KB is None:
        _KB = (PROJECT_ROOT / "knowledge_base.md").read_text(encoding="utf-8")
    if _SP is None:
        _SP = (PROJECT_ROOT / "system_prompt.md").read_text(encoding="utf-8")
    return _SP, _KB


# ----------------------------------------------------------------------------
# Golden
# ----------------------------------------------------------------------------
def evaluate_golden(
    client: OpenAI,
    trajectory: dict[str, Any],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Run the GT-rubric evaluator on an in-memory trajectory dict."""
    sp, kb = load_prompts()
    return gt.evaluate_trajectory_dict(client, trajectory, sp, kb, model, max_tokens)


def golden_summary_row(
    result: dict[str, Any],
    *,
    task: str,
    agent: str,
    model: str,
    trajectory_name: str,
    explorer_path: str = "",
) -> dict[str, Any]:
    gt_block = result.get("gt_class") or {}
    task_name = task.split("/")[-1] if task else trajectory_name
    return {
        "Task": task or task_name,
        "Agent": agent or result.get("agent") or "",
        "Model": model or result.get("model") or "",
        "GT Class(AI)": gt_block.get("label") or "",
        "GT Justification(AI)": gt_block.get("justification") or "",
        "Success Criteria (AI)": gt.format_sc_summary(result),
        "GT Class(Human)": "",
        "Success Criteria (Human)": "",
        "HITL Remarks": "",
        "Task name": task_name,
        "Trajectory name": trajectory_name,
        "Explorer HTML": explorer_path,
    }


# ----------------------------------------------------------------------------
# Failure
# ----------------------------------------------------------------------------
def evaluate_failure(
    client: OpenAI,
    trajectory: dict[str, Any],
    model: str,
    *,
    agent: str,
    task: str,
    failure_output: str = "",
    exception: str = "",
) -> dict[str, Any]:
    """Run the failure classifier on an in-memory trajectory dict."""
    context = ft.build_context_from_trajectory(
        trajectory,
        agent_name=agent,
        task=task,
        failure_output=failure_output,
        exception=exception,
    )
    result = ft.classify_failure(client, context, model)
    return {"context": context, "result": result}


def failure_summary_row(
    eval_out: dict[str, Any],
    *,
    agent: str,
    task: str,
    episode_name: str,
    status: str,
    gt_class: str = "",
    gt_justification: str = "",
    explorer_path: str = "",
) -> dict[str, Any]:
    """Compact row that matches the columns of failure_summary.csv,
    plus the GT-class label + 1-sentence justification we run on failures, plus
    the relative path of the per-episode explorer HTML inside results.zip."""
    res = eval_out.get("result", {}) or {}
    return {
        "agent": agent,
        "benchmark": task,
        "trial_id": episode_name,
        "status": status,
        "GT Class(AI)": gt_class,
        "GT Justification(AI)": gt_justification,
        "failure_type": res.get("failure_category", ""),
        "reason": res.get("reason", ""),
        "root_cause": res.get("root_cause", ""),
        "fix": res.get("fix", ""),
        "Explorer HTML": explorer_path,
    }


# ----------------------------------------------------------------------------
# OpenAI client factory
# ----------------------------------------------------------------------------
def make_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)
