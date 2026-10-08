---
name: evolve-document-driven-workflow
description: Evolve workspace governance from a requested workflow improvement or a durable future-facing agent/process correction such as 以后必须、一律、统一、不要再. Distinguish discussion from authorized implementation by user intent and scope, not message count. Keep one-off product requirements local.
---

# Evolve Document-Driven Workflow

## Goal

Turn the user's durable corrections into the smallest verified improvement to how this workspace is operated. Keep the workflow adapted to this user's established decisions as they evolve over time, while preserving current product and code ownership.

Use this as a recurring method loop, not a one-time redesign:

`observe -> identify wrong assumption -> choose generality -> reconcile owners -> verify -> learn from the next task`

Do not solve the concrete product problem in this skill. Establish the desired rule, scope, owner and observable outcome from the request and verified facts. Load `../align-solution-direction/SKILL.md` only for a material unresolved choice about direction; use `../technical-solution/SKILL.md` for a requested technical solution or a controlled technical change under WF-0002. Authorized governance edits use `../document-governance/SKILL.md`.

## Trigger Gate

Run the loop when either condition is true:

- the user explicitly asks to evolve, review, optimize, or create a skill/method from the document-driven workflow or prior conversation;
- the user gives a future-facing rule about agent or process behavior using durable scope such as “以后”, “必须”, “一律”, “统一”, “不要再”, or “沉淀成 skill”.

Do not trigger merely because a product requirement says a field, API, page, test, or deployment “must” behave a certain way. Keep one-off acceptance and local implementation constraints in the current task and their actual product/code owner. If durable scope is genuinely ambiguous and would broaden authority, stop for that decision; otherwise choose the narrowest local scope.

Choose the response from the current request and established authorization:

- Discussion, evaluation, challenge, an explicit no-edit request, or a general observation without a concrete requested change stays read-only. Return the grounded proposal or review and end that task without persisted activity state.
- An explicit implementation request, including a concrete future-facing correction directed at the workflow, authorizes the required governance edit when the rule and scope are clear. Evaluate, implement and verify within that scope in the same Development task, including on the first request. A proposal alone does not authorize its implementation.
- Discoverable facts and routine implementation choices do not require another user instruction. If a material choice remains unresolved or a change needs additional authority, pause only the dependent work, explain that exact decision, and continue independent authorized work. Existing authorization remains valid within its scope.

Use WorkflowContract for control strength and persistence; governance edits remain controlled. Preserve the root AGENTS engineering and safety gates, and explain the old constraint, new constraint and risk for an authorized baseline adjustment. Do not create a workflow phase, approval state, task or plan merely because a discussion precedes implementation.

## Load

1. Read the latest user instruction and only the earlier turns needed to recover its correction chain: rejected assumptions, repeated failures, accepted revisions, and current desired behavior.
2. Reuse already-read current context: root `AGENTS.md`, `文档/TASK_CONTROL.md`, the actual governance owners and only relevant skills/checker tests. Read `文档/工作流/WORKFLOW_CONTRACT.md` when it is a review target or the request authorizes governance changes; a read-only proposal does not select or enter a Workflow.
3. Read `WORK_CANDIDATES.md` under `文档/` only when the correction concerns known future work, promotion, or completeness answers.
4. Treat conversation as authority for desired agent behavior, not as product/code fact. Do not use Archive as a current rule source or scan unrelated documents.

## Run the Method Loop

### 1. Build an in-memory correction contract

Capture the latest instruction, concrete examples, explicitly rejected behavior, expected future behavior, and apparent scope. Latest explicit corrections supersede incompatible earlier preferences. Do not persist this ledger or create a user-profile document.

### 2. Find the generating assumption

Explain what default caused the repeated failure. Classify it as one of: direction/scope, execution type, control strength, persistence, knowledge ownership, validation/completion, interaction/output, or repository closeout. Fix the cause rather than adding an exception for the latest example.

### 3. Choose the narrowest supported generality

Evaluate `case -> work class -> project -> workspace`. Select the lowest level that covers all accepted examples. An explicit future-facing user rule can establish broader scope; repetition without such scope is evidence to investigate, not automatic authority to globalize.

### 4. Inspect all three governance surfaces

Always evaluate each surface, but edit only actual owners:

- **Method surface**: relevant skills and their `agents/openai.yaml` routing.
- **Activity surface**: `TASK_CONTROL.md`, `WORK_CANDIDATES.md`, activity-plan lifecycle, completion and Git separation.
- **Workflow surface**: root/project AGENTS, WorkflowContract/WF, StructureContract, CurrentDesign and machine checker/tests.

Record “no change” when a surface already enforces the corrected rule. Never create a task, candidate, plan, or document merely to prove the review happened.

### 5. Map the rule to one owner

- Agent entry or mandatory router: AGENTS.
- Reusable specialist method: skill.
- Authorization, task classification, lifecycle or Action semantics: WorkflowContract/WF.
- Document Kind and placement: StructureContract.
- Current control-plane mechanics: the sole StructureContract, WorkflowContract, checker implementation, and focused checker tests.
- Recoverable authorized work: TASK_CONTROL; evidence-backed uncommitted outcome: WORK_CANDIDATES.
- Product behavior or implementation: its ProductContract, CurrentDesign, source and tests, outside this method.

### 6. Review only the decisions that need it

For a discussion or requested proposal, state the old assumption, proposed invariant, boundary, actual owners, positive and negative cases, expected impact and proportionate verification. Keep a simple rule correction concise; a full technical-solution format is conditional on that skill's trigger.

Use `../challenge-solution/SKILL.md` only when the user requests a challenge or a named material uncertainty about the proposed outcome, boundary, ownership or completion rule warrants it. Supply a fixed proposal and keep that review read-only. A review verdict grants no authority; a review-only request ends after its result, while a review within authorized implementation returns to that task if no blocking decision remains. Do not invoke review because this is the first request or recursively review the review.

### 7. Complete authorized implementation

For an implementation request, establish the authorized delta, constraints, observable success criteria and stop conditions, then use `document-governance` to replace the conflicting rule at its owner. Update only direct routers, dependent skills and necessary machine guards. Choose implementation order from the dependencies and evidence. If a correction changes the approach within the authorized outcome, update the working plan and continue; a new outcome, boundary or user-owned decision follows the trigger gate above.

Do not preserve both old and new rules as caveats, replicate risk matrices, or rewrite unrelated governance.

### 8. Verify behavior, not wording

Verify a positive case that must proceed and a nearby negative case that must remain read-only or local. Run the modified skill's `quick_validate.py`, relevant checker tests, `npm run check:docs`, targeted conflict scans and scoped whitespace checks. Structural validation does not prove improved model behavior. Once required checks pass, broaden or repeat them only for new changes, failures or concrete unresolved concerns. Do not run product SystemTest or Deployment unless separately requested.

### 9. Close and keep learning

For an implementation task, use WF-0004. A temporary controlled evolution has no TASK_CONTROL row or ChangePlan. Persist only the resulting invariant at its owner; never save proposals, challenge reports, chat transcripts, correction ledgers, generic preference profiles, or review reports. On a later contradiction, re-enter this method and update the same rule rather than adding a parallel mechanism.

## Completion Rule

A discussion or review task is complete when its requested result and any necessary next decision have been returned; no workspace or lifecycle state remains open. An implementation task is complete when the authorized instruction is enforced at the right owner, all three surfaces were evaluated, stale conflicting behavior was removed, positive and negative evidence pass, and task/document lifecycle is clean. Repository commit or push remains a separate status.
