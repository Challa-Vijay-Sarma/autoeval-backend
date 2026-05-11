# Knowledge Base — ByteDance Golden Trajectory Rubric

This file is the **single source of truth** for the evaluator. Every GT class
and Success Criterion definition below is transcribed verbatim from the
ByteDance specification PDFs in `Knowledge Source/`. The evaluator MUST NOT
invent criteria that are not stated here.

## 1. Golden Trajectory (GT) Classes

> Source: `ByteDance Trajectory Report.pdf`, page 2.

- **GT1 — High-Quality Tool Use CoT**
  Features a clear Chain of Thought (CoT) reasoning process at each step of
  tool invocation.

- **GT2 — Long-Horizon Planning**
  Includes explicit initial planning and dynamic replanning as the task
  evolves.

- **GT3 — Error Identification and Recovery**
  Features self-correction, rollback, and retry reasoning chains to handle
  failures.

- **GT4 — Multi-Tool Collaborative Orchestration**
  Involves chained tool usage and complex dependency reasoning.

- **GT5 — Code Understanding and Contextual Reasoning**
  Covers cross-file and cross-module semantic comprehension.

### Extended GT definitions (from `Requirenments.pdf`)

- **GT1 Tool Use CoT**: Each tool invocation is preceded and followed by
  explicit Thinking segments that explain why the tool is used, the expected
  outcome, and how the returned results will be utilized.

- **GT2 Long-Horizon Planning Traj with Thinking**: Tasks involve ≥ 10
  tool calls (the source PDF says "steps" but treats "step" and "tool
  call" as synonyms — see section 3). The model performs explicit
  planning at the start, dynamically adjusts the plan during execution
  based on feedback (Replanning), and concludes with a summary and
  verification.

- **GT3 Error Identification and Recovery**: The trajectory includes
  failures such as tool errors, runtime exceptions, or unexpected outputs.
  The model identifies the root cause and applies appropriate recovery
  strategies (retry, backtrack, change approach).

- **GT4 Multi-Tool Collaborative Orchestration**: A single task involves
  coordinated use of **≥ 3 different tools**, with **data dependencies**
  between them. The model correctly understands and manages context passing
  across tools.

- **GT5 Code Understanding and Contextual Reasoning**: Involves semantic
  understanding across files/modules. The model explores the codebase to
  infer dependencies, interface design, and business logic.

## 2. Success Criteria (SC)

> Source: `ByteDance Trajectory Report.pdf`, page 2.

- **SC1 — Correct Behavior**
  The final output is correct and passes all tests or human verification.

- **SC2 — Transparent Reasoning**
  Clear "Thinking" blocks are provided at every key decision point.

- **SC3 — Sound Strategy**
  Tool selection and ordering align with expert human practices.

- **SC4 — Adaptability**
  The agent demonstrates reasonable responses to errors and unexpected
  situations.

## 3. Hard Requirements for a Golden Trajectory

> Source: `Requirenments.pdf`, page 1.

- Minimum length: **≥ 10 tool calls**. The source uses "steps" and
  "tool calls" interchangeably, so the unit to count is **individual
  tool invocations**, not trajectory step_ids. ATIF-v1.6 trajectories
  often batch several tool calls under a single `step_id`, so a
  trajectory with fewer than 10 `step_id`s can still satisfy this rule
  if its total `tool_calls[]` count across all steps is ≥ 10. Count
  only meaningful tool calls — repeated `mark_task_complete` retries
  and other no-op invocations do not contribute to the floor.
- Planning explicitness: Must include Thinking segments for
  Planning / Replanning.
- Task complexity: Tasks involving cross-file operations, multi-module
  modifications, and requiring environment exploration.
- Thinking Model: Preferably generated using an **Extended Thinking**
  model.

## 4. What Counts as Each GT Class (strict rules)

These are direct implications of the definitions above — use them to decide
the single dominant GT class when multiple classes partially match.

- **GT1 fits when** the defining signal is *per-tool-call* CoT: every tool
  invocation is wrapped in an explicit "why → call → how I'll use the
  result" reasoning segment. No requirement of replanning, error recovery,
  or multi-tool chaining beyond that discipline.

- **GT2 fits when** the defining signal is a **plan produced up front** and
  then **revised** as feedback arrives. An explicit task tracker, phase
  list, or plan artifact that is updated across the run is a strong GT2
  marker. A trajectory that is linear with no replanning is NOT GT2 even
  if it has many steps.

- **GT3 fits when** the defining signal is **at least one real failure**
  (tool error, wrong output, failing test, runtime exception) that the
  agent diagnoses and recovers from with a visible strategy change
  (retry / rollback / alternative approach). If the run was error-free,
  GT3 is NOT a fit.

- **GT4 fits when** the defining signal is **≥ 3 distinct tool families**
  whose outputs **feed each other**. Running the same shell in a loop is
  NOT GT4. Piping `find | grep | head`, or chaining `openssl → jq →
  sqlite3 → awk` with data passed between them, IS GT4.

- **GT5 fits when** the defining signal is the agent reading and
  reasoning across **multiple files / modules** to understand structure
  before acting. Trace-across-files, "read this to know how to edit that"
  patterns are GT5. Working inside a single file is NOT GT5.

## 4a. Classification Discriminators for Borderline Cases

Before committing to a GT class, apply the following tie-break questions.
These help distinguish the **dominant signal** from the surface silhouette
of the trajectory. Do not pattern-match on "has a traceback = GT3" or
"reads many files = GT5". Apply the questions below instead.

### GT3 vs GT5: is the error localising the bug, or just a symptom?

A trajectory with a traceback is NOT automatically GT3. Ask, in order:

1. **Does the error message name the file/function/line where the fix
   eventually goes?** If yes, and reading only that file would have been
   sufficient to understand and apply the fix, the dominant signal is
   GT3.
2. **Is the code at the error location actually wrong?** If the code
   that raised the error is itself correct (a budget check, an
   assertion, a schema validator, a downstream consumer), the error is
   a **symptom**, not the bug. The real bug lives elsewhere.
3. **Is each file's code locally correct, with the defect existing only
   as an emergent interaction across files?** If yes, the dominant
   signal is GT5.
4. **Did the agent have to read files outside the error location to
   understand the cause?** If yes, and the fix location differs from
   where understanding came from, the dominant signal is GT5.

Rule of thumb: **error localises the bug → GT3. Error localises a
symptom, real cause is a cross-module semantic interaction → GT5.**

Concrete examples:
- Stack trace says `IndexError: list index out of range` at the line
  that computes `final[-1]`, and the fix is a guard at that line → GT3.
- Stack trace says `OperationBudgetExceeded` in a budget check, but the
  code raising it is correct and the real cause is a per-iteration
  cache-invalidation call in a different module that defeats caching in
  a third module → GT5. The error localises the symptom, not the cause.

### GT1 vs GT4: narrated tool use vs chained tool use with data dependencies

- **GT1** is per-tool-call CoT discipline. Many tool calls, each with
  its own rationale, but the calls do not necessarily pass data between
  each other. Running `ls`, then `cat`, then `python` independently with
  good analysis is GT1 even when it is well reasoned.
- **GT4** requires the outputs of one tool family to **feed the next**,
  across at least 3 distinct tool families. A SHA256 computed with
  `sha256sum` compared against a value extracted via `jq` from a config,
  joined with a `sqlite3` row count, with each value flowing into the
  next step or into a cross-consistency check, IS GT4. Well-narrated
  but independent commands are NOT GT4.

### GT2 vs GT3: planned phases vs reactive debugging

- **GT2** requires an explicit **initial plan artifact** (phase list,
  task tracker entries, multi-phase contract) AND revision of that
  plan in response to feedback. Linear execution, even if long, is NOT
  GT2.
- **GT3** requires a **real failure** whose diagnosis changes the
  approach. A trajectory with a planned phase list and no failure is
  GT2, not GT3. A trajectory that responds to failures without a prior
  plan artifact is GT3, not GT2.

### GT5 vs GT1: reading many files vs understanding across files

- Reading multiple files to locate a single known-good fix target is
  GT1-shaped. The reads are tool calls with good CoT, but the
  understanding of each file is independent.
- **GT5** requires that the understanding of one file **informs the
  action in another**. If the agent reads file A to learn a contract,
  schema, or interface, then edits file B to respect that contract,
  that is GT5. Broad exploration reads with no cross-file inference
  is not GT5.

## 4b. Decision procedure when two GT classes both seem to fit

When two classes look plausible:

1. **Identify the single piece of evidence that, if removed, would most
   weaken the trajectory's case for being a Golden Trajectory at all.**
   That piece of evidence defines the dominant class.
2. **Counterfactual test:** imagine the trajectory without the evidence
   for one class and ask whether it would still be a Golden Trajectory.
   - If SC1 could still pass without the error (imagine no traceback
     was ever raised), then GT3 is not load-bearing. Pick the class
     whose signal survives.
   - If SC1 could still pass with the entire fix in one file, then GT5
     is not load-bearing.
   - If the tool chain could be replaced by a single script, then GT4
     is not load-bearing.
   - If the task could be completed without a plan artifact being
     written or revised, then GT2 is not load-bearing.
3. **Default when all signals are weak:** GT1, because per-call CoT
   discipline is the baseline competence every other class builds on.
4. **Never downgrade to GT1 from a class that genuinely fits just
   because the trajectory is also well-narrated.** GT1 is the fallback,
   not the consolation prize.

## 5. SC Evaluation Rules

- **SC1 = PASS** only if the trajectory's outputs actually satisfy the
  task's success criteria. If tests fail, artifacts are missing, or the
  final state does not meet the task's schema, SC1 = FAIL regardless of
  effort. If the trajectory does not contain verifier output, infer from
  whether the required artifacts were produced with correct content.
- **SC2 = PASS** only if Thinking / analysis / plan text appears at
  effectively every decision point, not just occasionally. "Mostly"
  qualifies as PASS; "intermittent" is WEAK PASS; "absent at key points"
  is FAIL.
- **SC3 = PASS** only if the tool choices and their ordering look like
  what a competent human would do. Flags for SC3 FAIL: using `sed` for
  multi-line edits, reading instead of writing when the task asks for
  edits, fabricating inputs, skipping verification.
- **SC4 = PASS** only if the agent actually handled at least one
  unexpected situation reasonably (a tool error, a failing test, a
  wrong intermediate result, an environment surprise that required
  changing approach). When real adversity was handled, this is PASS.
- **SC4 = NA when no real adversity was encountered.** If the task ran
  end to end without any genuine unexpected situation, the correct
  verdict is **NA**, not FAIL and not WEAK PASS. The agent cannot be
  graded on adaptability it was never given a chance to show. State
  explicitly "NA, no adversity encountered" in the reason.
- **SC4 = FAIL only when adversity occurred and the agent failed to
  adapt.** Stuck-in-a-loop behaviour, ignoring an error, or
  oscillating between states without resolution is FAIL. Absence of
  adversity is never FAIL.
- Trivial tool-availability recoveries (e.g. `hexdump` missing →
  fallback to `od`) are WEAK PASS signals at most, not full PASS.

## 6. Output Grammar the Evaluator Must Follow

When reporting, always use exactly these tokens:

- Verdicts: `PASS`, `FAIL`, `WEAK PASS`, `NA`.
- GT class: exactly one of `GT1`, `GT2`, `GT3`, `GT4`, `GT5`.

Never introduce new verdict labels (no "PARTIAL", no "MIXED", etc.).
