---
name: technical-solution
description: Produce a requested technical solution or the working plan for a controlled technical change under WF-0002. Preserve established direction, requirement provenance, user surface and persistence ownership, and proceed within explicit implementation authorization. Ordinary implementation does not require this skill; discussion and review requests stay read-only.
---

# Technical Solution

## Goal

Establish or reuse the compact direction contract: outcome, boundary, this task's execution type, invariants, ownership, completion rule and material open decisions. Load `../align-solution-direction/SKILL.md` only when a material direction choice cannot be resolved from available facts and user instructions. Preserve settled decisions; pause dependent implementation only for a blocking choice, missing evidence or authority.

For a requested solution or review, map the outcome to current support, required change, user surface, owner, persistence impact, evidence and material open decisions. For already authorized implementation, establish the necessary working plan and continue through implementation and verification. Add UI, API, data, operations, security or migration detail only when affected; a plan does not require a separate presentation or approval round.

Use this skill only for a requested solution or a controlled technical change under WF-0002. Do not invoke it for a change classified as normal by `WORKFLOW_CONTRACT.md`; those changes use the user request, targeted fact discovery, source, types and tests directly.

This skill is Development-only. A coding-ready solution may define scoped white-box verification and name a possible candidate output, but must not include an independent system-test campaign or deployment sequence as implementation steps. Route an explicitly requested SystemTest objective to its skill, which separately chooses control strength and persistence; Deployment always receives its own durable controlled task and plan. Do not call either a project or session phase transition.

An explicit implementation request authorizes its necessary planning and coding within scope once the coding-ready gate below is satisfied. Do not require separate direction approval when existing authorization and facts settle it. A solution-only request authorizes the requested design deliverable, not coding. If a blocking fact or observable acceptance surface is missing, pause dependent implementation and continue independent authorized work.

## Load

Reuse current root `AGENTS.md`, `文档/TASK_CONTROL.md` and matched project context; load the actual durable formal owner and missing facts needed by the task. Read `references/solution-template.md` when delivering a requested solution or detailed handoff; a temporary controlled implementation can keep its compact working plan in the task. Read `文档/工作流/WORKFLOW_CONTRACT.md` and the main Workflow when changing the workspace or preparing durable recovery; read-only discussion does not enter a Workflow. Do not create a placeholder CurrentDesign when another owner or source and tests already carry the facts; pause a required artifact write if no truthful durable owner can be identified.

## Build the solution

### Requested solution or review

Use a decision table when comparing choices; a simple decision can use concise prose. Cover the relevant fields:

| Outcome / decision | Current support | Required change | User surface | Owner | Persistence impact | Evidence | Open decision |
|---|---|---|---|---|---|---|---|

- Describe only affected dimensions. For example, add a short UI/API/data/security/operations note when it changes the decision; do not require a fixed layer checklist for every outcome.
- Group closely related surfaces and actions by business capability; do not repeat every component, field, endpoint, test class, or historical detail in the main plan.
- State whether the outcome changes an existing frontend/client surface, needs a new user entrypoint, remains internal, or is unreachable. State `no persistence change`, `reuse existing state`, `schema migration`, or `new persisted state` instead of leaving data ownership implicit.
- State the recommended order and no more than the prerequisite that affects a decision.
- Distinguish already authorized work, a requested design deliverable, and work blocked by a concrete product/ownership decision.
- State schema, migration, public-contract, operational, or compatibility impact only when the outcome reaches that boundary. Name concrete assets only when verified.
- Turn each unresolved product rule into two or three mutually exclusive, labelled choices and mark one as recommended. Do not write vague requests such as “confirm the rule”, “clarify the model”, or “determine ownership” without saying what the reviewer can choose.
- Give a short explicit list of rejected/non-goal legacy behavior so review does not reopen it accidentally.

When a user correction changes the direction contract, regenerate every dependent section and remove incompatible scope, completion, plan, skill, or document wording. Do not preserve the old assumption as a caveat or add a second rule beside it.

Use exact paths, routes, contracts, tables, jobs, and owners only to substantiate a decision or when a coding-ready follow-up is requested. Keep the main plan concise.

### Working plan for authorized implementation or requested detail

Cover these items in substance, using only the structure needed by the task:

1. Goal and Scope
2. Facts and Sources
3. Plan and Change Boundary
4. Agent Constraints
5. Success Criteria
6. Stop Implementation Conditions
7. Verification
8. Open Questions and Non-Goals

Use exact paths, routes, contracts, tables, jobs, and owners when verified. Separate confirmed requirements, verified facts, and working assumptions. Keep each working assumption visible and reversible; never promote it into an invariant, success criterion, non-goal, prohibition, ownership decision, compatibility promise, or stop condition. Explicit bounded delegation supplies authority only within that boundary; silence, repetition, or prior-plan inclusion does not. For phased work, state prerequisites, this phase's stopping boundary, downstream handoff, and explicit exclusions.

## Coding-ready gate

Do not hand the plan to coding unless:

- it preserves the direction contract without redefining its outcome, invariants, ownership, or completion rule;
- every retained constraint identifies whether its authority is a confirmed requirement or a verified fact, and no working assumption is treated as mandatory;
- the allowed change surface and prohibited dependencies are explicit;
- critical behavior is grounded in source, measured state, or an authoritative contract;
- success is observable through named UI actions, tests, APIs, CLI output, artifacts, or database checks;
- validation commands/checks are concrete and proportionate to risk;
- Development white-box completion evidence and repository state are reported separately; SystemTest or Deployment appears only when the user requested or the workflow actually established those independent tasks;
- missing facts, unavailable secrets/access, unauthorized expansion, and unresolved product choices are stop conditions;
- non-goals prevent scope drift.

Agent constraints must directly state what to read, what may/must not change, required validation, secret limits, and behavior that must remain stable.

Before coding any controlled technical change, establish its A→B delta, agent constraints, observable success criteria and stop conditions. For temporary execution, keep that plan in the current task and do not create persistence artifacts. For durable recovery, register the unfinished task and save exactly one `Kind: ChangePlan`, `Workflow: WF-0002` document under the project's or workspace's `推进中/` directory. `Draft` does not authorize coding. A persisted ChangePlan must depend on at least one actual durable formal owner and state the delta that owner will receive when its facts change; CurrentDesign is required only for capability-design changes.

## Closeout

Keep non-blocking questions separate from stop conditions. Once required verification passes, broaden or repeat checks only for new changes, failures or concrete unresolved concerns. Register only user-authorized executable work that needs durable recovery in `文档/TASK_CONTROL.md`; controlled strength alone is insufficient. Do not keep a Development task or ChangePlan active solely because Git closeout or another explicitly requested task remains after implementation and scoped white-box completion has passed. A requested SystemTest objective is a separate task and gets an activity plan only when durable and controlled; Deployment always gets its own durable task and plan. Add `WORK_CANDIDATES.md` only when verified work leaves an independent, evidence-backed, uncommitted outcome; do not type it or auto-promote it. Before reporting completion, execute WF-0004 conditionally and report repository state separately.
