#!/usr/bin/env python3
"""Failure Trajectory Analyzer.

Walks a jobs_new/<run> directory, identifies failed trials (verifier/reward
!= "1" or exception present), and classifies each failure with an OpenAI
chat-completions model into one of eight failure categories using a
Framing + Decision Procedure rubric.

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


def humanize_slug(slug: str) -> str:
    """Turn 'dependency_lockfile_issues' or 'dns-resolution-chain-debugging' into a
    readable label. Replaces _ and - with spaces and title-cases the result.
    Returned as a fallback when the trajectory has no structured user_input."""
    cleaned = re.sub(r"[_-]+", " ", slug).strip()
    return cleaned.title() if cleaned else slug


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
        # Humanized slug used in the prompt as a fallback when `user_input` is
        # empty. The authoritative task description lives in `user_input`, which
        # `extract_user_input` parses from the trajectory's first user message.
        "task_label": humanize_slug(row["benchmark"]),
        "user_input": user_input,
        "failure_output": row.get("failure_output", ""),
        "thoughts": thoughts[:3],
        "actions": actions[:5],
        "observations": observations[:5],
        "exception": row["exception"],
        "early_failure": early_failure,
    }


FAILURE_PROMPT_TEMPLATE = """
You are analyzing a single AI agent failure. Apply the rubric procedurally:
for each of the eight categories below, walk its Decision Procedure step by
step and decide MATCH or NO MATCH. After scoring all eight, apply the
DISAMBIGUATION rules to pick exactly one primary category. Return only the
final classification plus a single concise reason that supports the MATCH —
no per-category reasoning, no quotes, no confidence score.

CATEGORIES

1. Disobey Task Specification
2. Step Repetition
3. Unaware of Termination Conditions
4. Context Loss
5. Premature Termination
6. No or Irrelevant Verification
7. Reasoning-Action Mismatch
8. Weak Verification

RUBRIC

Disobey Task Specification. Framing. Disobey task specification concerns material contradictions to explicit directives in the task, including both hard directives ("must," "required," "shall,"
explicit prohibitions) and soft directives ("should," "recommended," "aim to"). Pure response-format/schema violations are excluded. Violations include ignoring or replacing required methods, constraints, sources of truth, or required output locations. Using the wrong source of truth
counts even if the result appears plausible. Transient violations fully reversed before completion
are ignored. Acceptable substitutions due to environment constraints are allowed if demonstrably
equivalent via strong proof (tool-native introspection, passing the eval/check script, checksum/bytewise equality, or independent cross-check). Soft-guidance departures only count when they clearly
undermine the task's stated intent or expected behavior.
Decision Procedure.
Step 1. Locate directives. Identify hard or soft directives from task/system instructions: required
methods/sources, success criteria, required output paths, or prohibitions/recommendations.
If none present: NO MATCH
Step 2. Check for contradiction. Determine if the agent ignored or replaced at least one directive,
for example:
- Using a placeholder instead of the required implementation
- Performing a forbidden operation
- Using the wrong source of truth/metric
- Altering/fabricating data instead of recovering it
- Failing to measure/verify a mandated numeric constraint
- Failing to produce required artifact at the specified path
- Using Tool Y when "use exactly Tool X" and X is available
Exceptions: Extra copies elsewhere are acceptable if correct artifact exists at
required path; "use X if available; otherwise Y" permits Y unless explicitly forbidden.
Step 3. Assess materiality.
- Response-format/schema issues only: NO MATCH
- Shortfalls (i.e. numeric) despite attempting mandated method: NO MATCH
- Wrong source of truth or missing required output: material
- Soft-guidance violation that undermines task intent: material
Step 4. Check for correction. If the agent fully corrected/reversed the violation before completion
such that final outcome satisfies all directives: NO MATCH
Step 5. Decide. If Steps 2-3 satisfied and Step 4 not satisfied: MATCH; otherwise: NO MATCH

Step Repetition. Framing. Step Repetition occurs when the agent re-executes the same phase
(same sub-goal, same tool/effect, same target) with semantically or conceptually identical actions.
A material change meaningfully alters strategy, algorithm, mode, or information state; superficial
edits (formatting, parameter tweaks that do not change mode, refactors preserving the same method)
are not material. Regenerating artifacts implementing the same underlying method counts as repetition. Switching tools, changing algorithms, or introducing meaningfully different inputs counts as
progress. Repeated initiations of the same phase that never complete (abort-loops) are an explicit
subtype.
Decision Procedure.
Step 1. Verify preconditions. Confirm phases/sub-goals are identifiable and multiple distinct agent
blocks/turns exist.
- If either missing: NO MATCH
Step 2. Collect signals. Extract: phase grouping, distinct tool-call blocks (with block-level
deduplication), outcome classification (success/error/interrupted), parameters/flags/code identity,
intent statements.
Step 3. Apply block-level deduplication. Each tool-call block counts as at most one attempt per
unique action. Do not count multiple lines, echoed commands, or parallel outputs within
the same block as separate attempts.
Step 4. Trigger check. Within a single phase, do two or more semantically/conceptually identical
actions occur across distinct blocks?
- If no: NO MATCH
Step 5. Assess identity.
- Semantic identity: Same tool, same effective operation, same target inputs/paths, same
  effective flags/arguments (ignoring whitespace/verbosity tweaks)
- Conceptual identity: Same underlying method/algorithm and inputs, even if code/scripts differ
- Not identical: Different I/O routing, different targets, parameters altering operation mode
Step 6. Classify category.
- Post-error-blind-retry
- Post-success repetition
- Verification repetition (exceeds N=2 for simple read-only probes)
- Abort-loop repetition (repeated initiations without outcomes)
Step 7. Apply exclusions.
- Material changes (different algorithm/mode, different inputs/targets, switching tools): not repetition
- Diagnostic-only changes (logging flags, 2>&1): do not break identity
- Up to 2 identical performance-tuning re-runs: allowed
- First retry after incomplete/interrupted attempt: not counted
Step 8. Decide. Repetition count >= 2 within any single phase: MATCH; otherwise: NO MATCH

Unaware of Termination Conditions. Framing. Failure-to-stop occurs when an agent continues
acting beyond a reasonable stopping point - after success has been achieved, after futility is established, or when the agent prematurely declares completion before success conditions are met. It
captures unnecessary continuation, lack of halting after confirmed futility, and premature finalization.
Decision Procedure.
Step 1. Verify preconditions. A stopping point must be identifiable:
- Explicit success criteria, OR
- Strong implicit success evidence (verifier pass, validated artifact), OR
- Futility (two consecutive identical failures with no progress), OR
- Explicit finalization instructions
- If none present: NO MATCH
Step 2. Identify triggering condition. One of the following must occur:
- Success condition satisfied
- Futility established
- Agent explicitly claims/signals completion
Step 3. Classify violation type.
- C1 (After Success): Agent continues the completed subgoal for >=2 additional turns (1
  verification turn allowed)
- C2 (After Futility): After two consecutive identical failures, agent continues same failing approach
- P1 (Premature Completion): Agent claims completion before success conditions are met
Step 4. Apply exclusions.
- One verification turn after success: allowed
- Hygiene-only turns (renaming, formatting): not counted
- Meaningful strategy change: resets futility counter
- Tool-call echoes within single turn: not counted
Step 5. Decide.
- C1 with >=2 redo turns after success: MATCH
- C2 with >=1 further attempt without strategy change: MATCH
- P1 with explicit completion before required criteria: MATCH
- Otherwise: NO MATCH

Context Loss. Framing. History loss occurs when the agent forgets or contradicts relevant recent context. Two major forms exist: (1) state-memory loss - forgetting concrete state (files created, errors resolved, configs applied); (2) context-memory loss - forgetting semantic commitments
(instructions, constraints, plans, prior reasoning). A match occurs when later actions/claims are
incompatible with previously established state or context within the same window.
Decision Procedure.
Step 1. Verify preconditions. Identify a recent contiguous window without major resets containing
at least one established:
- State (environmental fact: file created, dependency installed, error fixed), OR
- Context (semantic commitment, plan, instruction, constraint, prior reasoning)
- If neither exists: NO MATCH
Step 2. Identify contradiction. Look for later behavior that:
- Acts as if earlier state/context never occurred
- Reverts to older assumptions
- Re-asks answered questions or redoes completed steps
- Ignores earlier constraints, instructions, or reasoning
Step 3. Classify violation type.
- State Contradiction: Agent behaves as if state updates never happened (recreating resources,
  using stale outputs)
- Context Contradiction: Agent forgets semantic context (ignoring constraints, switching
  tasks after clarification, contradicting own reasoning)
Step 4. Apply exclusions.
- Acknowledged uncertainty or legitimate recovery attempts
- Harmless re-checks
- Explicit environment resets
- Pure formatting issues with no context reliance
- Contradictions within same tool block
Step 5. Decide. Contradiction present AND no exclusion applies: MATCH; otherwise: NO MATCH

Premature Termination. Framing. Premature termination occurs when an agent declares completion or presents a final answer before meeting explicit task objectives or providing required verifications/critical data that were obtainable or already obtained but not delivered via required output
channels. The focus is on whether necessary, explicitly specified information for task success was
exchanged or verified. Involuntary endings (timeouts, crashes) are excluded. A concrete, actionable
handoff that enables continuation avoids a match.
Decision Procedure.
Step 1. Identify objectives. Extract explicit objectives and required verifications/data from task
prompt, success criteria, or built-in checks. Do not infer implicit checks.
- If none identifiable: NO MATCH
Step 2. Confirm agent-declared ending. Did the agent explicitly declare completion or present
final outputs as if done? Exclude involuntary endings.
- If no: NO MATCH
Step 3. Check for unmet necessities. Are any explicit objectives, required verifications, or critical
data missing from required output channel(s)? Include items obtained but not delivered via
required channels.
- If no: NO MATCH
Step 4. Evaluate handoff. Did the agent:
(a) Clearly flag infeasibility/incompleteness with concrete, actionable handoff (exact commands, file paths, parameters), OR
(b) Provide sufficient instructions enabling continuation?
- If yes: NO MATCH
Step 5. Decide. Agent ended with explicit necessities missing AND claimed success or presented
proxies as final: MATCH; otherwise: NO MATCH

No or Irrelevant Verification. Framing. This rubric flags missing or irrelevant verification of required properties, with completion defined as calling mark_task_complete(). It distinguishes
core functional properties (correctness of behavior/edits, success metrics, minimality when central)
from peripheral structural properties (format, filenames, ordering, mere existence). When core
properties exist, the agent must verify at least one with an observed, substantive result before completion. Self-assertions ("looks good") are insufficient. Verification of non-conforming artifacts
cannot satisfy core verification. Failing core results ignored at completion are a match.
Decision Procedure.
Step 1. Identify required properties. From task/evaluator, classify properties:
- Core functional: Correctness, required metrics, minimality when central
- Peripheral structural: Format, filenames, ordering, mere existence
- If no required properties: NO MATCH
Step 2. Check completion status. Did agent call mark_task_complete()?
- If no: go to Step 5
Step 3. Check for explicit admission. Does agent explicitly acknowledge inability or that required
properties are not met?
- If yes: NO MATCH
Step 4. Assess core verification at completion. Consider only checks on the required final deliverable under method constraints. Self-assertions do not count.
- No substantive result for any core property: MATCH
- Failing core result ignored: MATCH
- Core property verified with observed compliant result: NO MATCH
- Only peripheral properties exist: any observed relevant result: NO MATCH; none/self-assertion only: MATCH
Step 5. No completion: check for verifier bypass. Was a designated verifier for core properties
provided/available? If so, and no substantive authoritative result ever appears: MATCH;
otherwise: NO MATCH

Reasoning-Action Mismatch. Framing. Reasoning-action mismatch evaluates whether an
agent's stated reasoning/conclusions and binding commitments from the task/interface contract align
with observable actions and outputs. Decision principle: if a claim/requirement says X but behavior/artifacts show not-X, that's a mismatch. Environment/tool failures only count when the agent
proceeds as if success occurred or asserts outcomes contrary to evidence.
Decision Procedure.
Step 1. Identify commitments/claims. From transcript and task instructions, identify: required
output format/protocol, declared method/provenance, success/verification assertions.
- If none present: NO MATCH
Step 2. Pre-check: JSON-only filter. If all contradictions are JSON format/structure issues and
none blocked core deliverables/validations: NO MATCH
Step 3. Collect behavioral evidence. Examine: executed commands/tool calls, generated artifacts/files, logs/errors, timestamps/order, final responses/formatting.
Step 4. Compare commitments to evidence. Apply materiality rule - count only if:
- Repeated: 2+ independent instances of same contradiction class, OR
- Blocking: Violates required spec/contract or prevents completion/tooling
Step 5. Apply clarifications.
- Judge claims against evidence available when claim was made
- Later failures create mismatch only if agent repeats success claim, proceeds as if successful, or relies on earlier claim without reconciliation
- Core requirements: items affecting acceptance/grading (required files/paths, metrics,
  verification artifacts, mandatory methods)
- Ignore benign deviations (trailing newline, harmless comments)
Step 6. Decide. Clear contradiction that is repeated or blocking: MATCH; otherwise: NO MATCH

Weak Verification. Framing. Weak verification and data fabrication are distinct but related failure
modes. Verification is weak when checks do not cover properties that must hold for true correctness.
Fabrication occurs when the agent creates or alters data that should have been measured, recovered,
or derived from specified sources, and treats it as authentic. Relying solely on an authoritative
official evaluator is sufficient unless additional explicit essentials are stated. Declared limitations
can mitigate narrower verification if claims are correspondingly narrowed.
Decision Procedure.
Step 1. Identify data fabrication. Did the agent generate or alter outputs/source data that should
have been recovered/derived from existing artifacts, and treat result as genuine? Include
modifying evaluation target/environment to make checks pass.
Require reliance: Count only if fabricated artifact is used to satisfy requirement, pass check,
or serve as deliverable.
- Set data fabrication flag
Step 2. Identify verification actions. Does transcript show agent using checks to judge progress/completion (tests, assertions, comparisons, metrics, end-to-end trials)?
- If none: weak verification = false
Step 3. Extract essentials.
- Explicit requirements: Critical properties for correctness from task/instructions
- Authoritative evaluator: If designated, treat its checks as essentials
- Implied prerequisites: Prerequisites whose failure makes explicit requirements impossible
Step 4. Compare coverage to essentials
Step 5. Check decisiveness and mitigation
Step 6. Decide output
Tie-breakers and exclusions: authoritative evaluator rules; absence of verification; brittleness exceptions; inconclusive-only exceptions; verified constraints; strong coverage.

DISAMBIGUATION

When more than one category's Decision Procedure produces MATCH, choose by this
priority (highest first):

1. Disobey Task Specification - wins when the agent ignored an EXPLICIT, named
   directive from `User Input` (e.g. "must use library X", "do not modify file Y").
   Beats all others if the directive violation directly caused the failure.

2. Premature Termination - wins over Unaware of Termination Conditions whenever the
   agent declared completion BEFORE meeting the success criteria. Reserve
   "Unaware of Termination Conditions" for C1 (continued after a success
   observation) and C2 (continued after futility) only.

3. Weak Verification - wins over No or Irrelevant Verification when the agent DID
   run some verification but coverage was insufficient. "No or Irrelevant" is
   reserved for the case where no substantive check ran at all.

4. Reasoning-Action Mismatch - applies only when the contradiction is between the
   agent's OWN stated reasoning and its actions/observations. If the
   contradiction is between the task's instructions in `User Input` and the
   agent's actions, classify as Disobey Task Specification instead.

5. Step Repetition - only as the primary category if repetition is the proximate
   cause of failure. If repetition merely accompanied a deeper issue (e.g.,
   context loss), classify by the deeper cause and list Step Repetition as
   `secondary_category`.

If two categories still tie after these rules, pick the one whose Framing more
closely matches the evidence and record the loser in `secondary_category`.

EVIDENCE

Agent: {agent}
Task: {task_name} ({task_label})

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

OUTPUT

Return ONLY JSON, with "failure_category" set to exactly one of the eight names
above (verbatim). "reason" is a single, concise explanation (<=2 sentences)
of the MATCH that was selected — no walkthrough of the other categories.

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
        task_label=context.get("task_label", "") or humanize_slug(context.get("task_name", "")),
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
