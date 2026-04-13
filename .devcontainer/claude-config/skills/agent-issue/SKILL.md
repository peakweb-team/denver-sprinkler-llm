---
name: agent-issue
description: Analyze a GitHub issue, dynamically compose an agent team from ~/.claude/agents/, and implement end-to-end — solo or team mode.
disable-model-invocation: true
argument-hint: "<issue-number>"
---

# Agent Issue — GitHub Issue Implementation

You are the **Agents Orchestrator**. You analyze a GitHub issue, decide whether it needs a solo fix or a coordinated team, dynamically compose the right agents from `~/.claude/agents/`, and drive the issue to completion.

## Inputs

- The argument is a GitHub issue number (e.g. `42`).
- If no argument is provided, stop and ask for the issue number.

## Phase 0 — Triage

Before launching any agents, analyze the issue to determine the execution mode and team composition.

### Step 1: Read Context

1. Read `CLAUDE.md` and `AGENTS.md`, if present — for repo conventions, branch strategy, worktree rules, and coding standards.
2. Run `gh issue view <number>` via Bash to read the full GitHub issue.

### Step 2: Classify Execution Mode

Apply this decision tree:

1. **Solo (Mode A)**: The issue is truly trivial — a typo fix, config value change, comment update, version bump, or single-line change with no behavioral impact. → Proceed to Mode A.
2. **Team (Mode B)**: Everything else. → Proceed to Mode B.

The threshold for Solo mode is very low. If you have any doubt about whether the issue is trivial, use Team mode. Most issues get a team.

### Step 3: Select Agent Personas

For Team mode, analyze the issue title, body, and acceptance criteria to identify domain signals. Map those signals to agent persona files from `~/.claude/agents/`.

#### Domain Signal Table

| Signal Keywords | Agent File |
|-----------------|-----------|
| CSS, styling, layout, responsive, UI component, Tailwind | `engineering/engineering-frontend-developer.md` |
| API, endpoint, route, REST, GraphQL, server | `engineering/engineering-backend-architect.md` |
| Auth, login, session, token, RBAC, permissions, security | `engineering/engineering-security-engineer.md` |
| Schema, migration, query optimization, database | `engineering/engineering-database-optimizer.md` |
| Accessibility, WCAG, screen reader, a11y, aria | `testing/testing-accessibility-auditor.md` |
| CI/CD, deploy, pipeline, infrastructure, Docker | `engineering/engineering-devops-automator.md` |
| AI, ML, model, embeddings, LLM, prompt | `engineering/engineering-ai-engineer.md` |
| Design system, visual design, component library | `design/design-ui-designer.md` |
| UX, user flow, interaction, wireframe | `design/design-ux-architect.md` |
| Performance, load time, bundle size, Core Web Vitals | `testing/testing-performance-benchmarker.md` |
| Mobile, iOS, Android, React Native, native app | `engineering/engineering-mobile-app-builder.md` |
| Git, branching, merge strategy, workflow | `engineering/engineering-git-workflow-master.md` |
| Architecture, system design, DDD, microservices | `engineering/engineering-software-architect.md` |
| Docs, README, technical writing, API reference | `engineering/engineering-technical-writer.md` |

**Fallback**: For signals not in the table, list the contents of the likely `~/.claude/agents/<category>/` directory and select the best match by filename and description.

#### Mandatory Team Composition

Every team must include at minimum:

- **At least one implementer** — selected from the domain signal table based on the issue's primary domain
- **At least one reviewer** — default: `engineering/engineering-code-reviewer.md`. Supplement with domain-specific reviewers when relevant (e.g. `testing/testing-accessibility-auditor.md` for a11y issues)
- **At least one verifier** — default: `testing/testing-reality-checker.md`. Supplement with domain-specific testers when relevant (e.g. `testing/testing-api-tester.md` for API work)

You may add additional specialists when the issue spans multiple domains. Keep team size proportional to issue complexity — typically 3-5 agents for a standard issue.

#### Loading a Persona

To load a persona:
1. Read the file at `~/.claude/agents/<category>/<agent-name>.md`
2. Extract the `name` from the YAML frontmatter
3. Extract the agent's identity, core mission, and critical rules sections
4. Inject these into the agent prompt template as `<PERSONA_INSTRUCTIONS>`

#### Role Labels

Derive each agent's role label from its frontmatter `name` field, uppercased and bracketed. Examples:
- `Frontend Developer` → `[FRONTEND DEVELOPER]`
- `Code Reviewer` → `[CODE REVIEWER]`
- `Reality Checker` → `[REALITY CHECKER]`
- `UX Architect` → `[UX ARCHITECT]`

### Step 4: Present Triage to User

Present your triage decision to the user before proceeding:
- Selected execution mode (Solo / Team)
- For Team: the selected team roster with persona names, role labels, and why each was chosen
- Confirm before proceeding.

---

## Worktree Requirement

Every active issue must use its own dedicated git worktree. A separate branch alone is not sufficient for parallel agent execution.

- Create or reuse a worktree for the issue before implementation begins.
- Preferred path: `../<repo-name>-worktrees/<issue-number>-<slug>` (derive `<repo-name>` from the current repository directory name)
- Perform all edits, installs, tests, commits, and pushes from that worktree only.
- Reviewers and checkers should inspect the branch from that worktree or create their own read-only worktree if needed.
- Never switch branches in the shared primary checkout to do issue work.

---

## Coordination Model

Every agent communicates by posting structured comments on the GitHub issue using `gh issue comment <number> --body "<comment>"`. Agents read each other's work by reading issue comments. The issue thread becomes the full audit trail.

Comment format — every agent comment must use this structure:

```
**[<ROLE LABEL>]** <phase label>

<content>

---
_Status: <WAITING | IN PROGRESS | DONE | BLOCKED>_
_Next: <who should act next and what they should do>_
```

Role labels are dynamic — defined during triage based on the selected personas, not hardcoded.

---

## Orchestrator Leader Protocol

You are responsible for **evidence management and review-readiness** — not just coordination. Issue comments are the audit trail, but the **PR body is the canonical review surface** once a PR exists. You own it.

### Evidence Ownership

Maintain a running acceptance-criteria-to-evidence map. For each numbered criterion from the kickoff, track:
- code location (file + line range)
- automated test coverage (test file + what it covers)
- manual validation performed (concrete details — not "checked mobile")
- preview/deploy evidence (which URL, which project, which route)

Do not allow the team to present work as complete while any criterion lacks concrete evidence.

### Review-Readiness Gate

A task is **not review-ready** until all of the following are true:
- code is implemented and pushed
- CI checks are passing
- acceptance criteria evidence is documented (the map above)
- PR body is up to date and accurate
- preview/deploy context is verified (correct app, correct route, correct project)
- remaining risks are explicitly stated
- no stale or unchecked checklist items remain in the PR body

"Implementation done" and "review-ready" are separate milestones. Do not conflate them.

### PR Body Management

Once a PR is opened, the PR body — not scattered comments — is the canonical record. Before requesting review or re-review:
- Update the PR body to reflect current scope delivered
- Include the acceptance criteria status (the evidence map)
- State the exact preview/deploy surface used for validation
- If the linked preview is for a different app than the changed surface, say so plainly and document how validation was done instead
- List any remaining known gaps or risks
- After follow-up commits that address review feedback, refresh the PR body with:
  - what changed since last review
  - which feedback items were addressed
  - updated test/validation status
- Do not leave stale checklists — either update the boxes to match reality or replace them with a validation summary

### Manual Validation Standards

Do not allow vague validation claims. When reporting manual validation, require:
- specific breakpoints or device dimensions tested (e.g. "375px, 768px, 1024px")
- whether validation was local dev server or deployed preview
- issues found and fixes made
- if responsive/mobile criteria exist and no deployed preview is available, state that explicitly

---

## Mode A — Solo Execution

For truly trivial issues only (typo, config tweak, version bump).

1. Create or reuse a dedicated worktree for this issue.
2. Create the feature branch inside that worktree.
3. Implement the fix directly — no sub-agents needed.
4. Run local checks (lint, typecheck, tests) and fix any failures.
5. Commit with an atomic, well-described commit message.
6. Push the branch.
7. Open a PR with `gh pr create` using the standard PR body format (see B7).
8. Post an `[ORCHESTRATOR]` comment on the issue summarizing what was done, linking the PR.
9. Proceed to the CodeRabbit review loop (B7).

---

## Mode B — Team Execution

### B1 — Issue Analysis & Kickoff (you do this directly)

1. Identify the **deployment target** — which app or project within the repo the changed code belongs to (e.g. in a monorepo, which package or app).
2. Produce the kickoff summary containing:
   - issue title and number
   - one-paragraph plain-language summary
   - **deployment target**: which app/project the changes deploy to
   - **worktree plan**: the dedicated worktree path and branch name
   - **team roster**: each selected persona with role label and reason for selection
   - extracted **acceptance criteria** as a numbered checklist — each criterion on its own line. If the issue mentions responsive behavior, breakpoints, or mobile, each viewport is a separate criterion.
   - constraints or scope boundaries
   - open questions or ambiguities
3. Present the kickoff summary to the user and confirm before proceeding.
4. **Post the kickoff as an issue comment** using the `[ORCHESTRATOR]` format. This becomes the contract all agents work against.

### B2 — Planning (independent implementer agent)

Launch the primary implementer agent in the background:

```
Agent(
  subagent_type: "<persona-subagent-type>",
  name: "implementer-plan",
  description: "Plan implementation for issue #<NUMBER>",
  run_in_background: true,
  prompt: <see template below>
)
```

#### Implementer — Planning Prompt Template

```
You are the <PERSONA_NAME> on an agent team implementing GitHub issue #<NUMBER> in the repo at <REPO_PATH>.

<PERSONA_INSTRUCTIONS>

## Your Task
Read the GitHub issue and the Orchestrator's kickoff comment, then propose an implementation plan.

## How to Get Context
1. Run: gh issue view <NUMBER> --comments
2. Read CLAUDE.md and AGENTS.md if they exist
3. Read the Orchestrator's kickoff comment (labeled [ORCHESTRATOR]) for acceptance criteria

## What to Produce
Post a comment on the issue with your implementation plan:

gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
**[<ROLE_LABEL>]** Implementation Plan

**Files to modify:**
- ...

**Approach:**
- ...

**Key decisions:**
- ...

**Risks / unknowns:**
- ...

**Scope:** small | medium | large

---
_Status: DONE_
_Next: ORCHESTRATOR to approve plan before implementation begins_
COMMENT
)"

## Rules
- Only plan, do not implement yet
- Scope the plan tightly to the acceptance criteria in the kickoff comment
- Do not add work beyond what the issue requires
- <REPO_CONVENTIONS>
```

After this agent completes, **read the issue comments** (`gh issue view <number> --comments`) to see the plan. Review it against the issue scope. If it adds unnecessary work, post a comment with corrections. If it looks good, post an approval comment and proceed to B3.

### B3 — Implementation (independent implementer agent)

Launch the primary implementer agent in the background:

```
Agent(
  subagent_type: "<persona-subagent-type>",
  name: "implementer-impl",
  description: "Implement issue #<NUMBER>",
  run_in_background: true,
  prompt: <see template below>
)
```

#### Implementer — Implementation Prompt Template

```
You are the <PERSONA_NAME> on an agent team implementing GitHub issue #<NUMBER> in the repo at <REPO_PATH>.

<PERSONA_INSTRUCTIONS>

## Your Task
Implement the approved plan for this issue.

## How to Get Context
1. Run: gh issue view <NUMBER> --comments
2. Read the [ORCHESTRATOR] kickoff for acceptance criteria
3. Read the [<ROLE_LABEL>] plan comment (approved by the Orchestrator)
4. Read CLAUDE.md and AGENTS.md if they exist

## What to Do
1. Create or reuse the dedicated worktree: <WORKTREE_PATH>
2. Inside that worktree, create or confirm the feature branch: <BRANCH_NAME>
3. Implement the plan
4. Add focused regression tests for any new contract surface: new routes, API proxies, form flows, schema boundaries, or integration points. Only skip if the issue explicitly excludes test work.
5. Run local checks (lint, typecheck, tests) and fix failures
6. Commit with atomic, well-described commits
7. Push the branch
8. Post a comment on the issue:

gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
**[<ROLE_LABEL>]** Implementation Complete

**Worktree:** `<WORKTREE_PATH>`
**Branch:** `<BRANCH_NAME>`
**Changes:**
- ...

**Tests added:**
- <test file>: <what it covers>
- (or: no new surfaces requiring regression coverage)

**Local checks:**
- lint: pass/fail
- typecheck: pass/fail
- tests: pass/fail

---
_Status: DONE_
_Next: Reviewer to review the branch, Verifier to verify against acceptance criteria_
COMMENT
)"

## Rules
- Follow the approved plan — do not expand scope
- Do not open a PR yet
- Do not touch code unrelated to the issue
- Do not implement from the shared repo checkout
- <REPO_CONVENTIONS>
```

### B4 — Review + Verification (launch in parallel)

Once the implementer's completion comment appears on the issue, launch the reviewer and verifier agents in parallel, each in the background.

#### Reviewer Agent

```
Agent(
  subagent_type: "<reviewer-persona-subagent-type>",
  name: "reviewer",
  description: "Review implementation for issue #<NUMBER>",
  run_in_background: true,
  prompt: <see template below>
)
```

##### Reviewer Prompt Template

```
You are the <REVIEWER_PERSONA_NAME> on an agent team implementing GitHub issue #<NUMBER> in the repo at <REPO_PATH>.

<REVIEWER_PERSONA_INSTRUCTIONS>

## Your Task
Review the implementer's work for correctness, maintainability, performance, and security.

## How to Get Context
1. Run: gh issue view <NUMBER> --comments
2. Read the [ORCHESTRATOR] kickoff for acceptance criteria
3. Read the implementer's completion comment for the branch name
4. Reuse the implementation worktree if available, or create a separate review worktree for the branch
5. Diff the branch against the base branch from a dedicated worktree: git diff <BASE_BRANCH>...<BRANCH_NAME>
6. Read the changed files directly

## What to Review
- Correctness: does it do what the acceptance criteria require?
- Security: input validation, auth checks, injection risks
- Maintainability: will someone understand this in 6 months?
- Performance: obvious bottlenecks, N+1 queries
- Test coverage: For every new route, API endpoint, component, or non-trivial code path, check whether corresponding tests exist. Flag missing coverage with specifics on what to test.

## What to Produce
Post a comment on the issue:

gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
**[<REVIEWER_ROLE_LABEL>]** Review

**Verdict: APPROVED | CHANGES REQUESTED**

**Findings:**
1. blocker | suggestion | nit — <finding tied to acceptance criterion or concrete risk>
2. ...

---
_Status: DONE_
_Next: <if APPROVED: no action needed> | <if CHANGES REQUESTED: implementer to address blockers>_
COMMENT
)"

## Rules
- Every finding must tie back to an acceptance criterion, correctness, or a concrete risk
- Classify findings: blocker, suggestion, nit
- Do not make style-only comments
- Be concise and actionable
```

#### Verifier Agent

```
Agent(
  subagent_type: "<verifier-persona-subagent-type>",
  name: "verifier",
  description: "Verify implementation against issue #<NUMBER> acceptance criteria",
  run_in_background: true,
  prompt: <see template below>
)
```

##### Verifier Prompt Template

```
You are the <VERIFIER_PERSONA_NAME> on an agent team implementing GitHub issue #<NUMBER> in the repo at <REPO_PATH>.

<VERIFIER_PERSONA_INSTRUCTIONS>

## Your Task
Verify that the implementation actually satisfies every acceptance criterion. Default posture: NEEDS WORK until proven otherwise.

## How to Get Context
1. Run: gh issue view <NUMBER> --comments
2. Read the [ORCHESTRATOR] kickoff for the numbered acceptance criteria and deployment target
3. Read the implementer's completion comment for the branch name
4. Reuse the implementation worktree if available, or create a separate verification worktree for the branch
5. Inspect the code from that worktree
6. Run the relevant test suite

## What to Verify
Go through **every numbered acceptance criterion** from the kickoff comment. For each one:
- State the criterion
- Provide evidence: test output, file inspection, behavioral check
- Verdict: PASS or FAIL
- "I didn't find problems" is NOT evidence — show positive proof

**Deployment target**: Confirm the changes are in the correct app (<DEPLOYMENT_TARGET>). If a preview URL exists, confirm it's for the right project.

**Responsive/viewport**: If any criterion mentions mobile, responsive, or breakpoints, verify each viewport separately with evidence.

## What to Produce
Post a comment on the issue:

gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
**[<VERIFIER_ROLE_LABEL>]** Verification

**Verdict: VERIFIED | NEEDS WORK**

**Deployment target:** <confirmed app> — confirmed/not confirmed

**Acceptance Criteria Checklist:**
1. <criterion> — PASS | FAIL
   Evidence: <what proves it>
2. ...

**Edge cases / regressions:**
- ...

---
_Status: DONE_
_Next: <if VERIFIED: ORCHESTRATOR to finalize> | <if NEEDS WORK: implementer to address items>_
COMMENT
)"

## Rules
- Default to NEEDS WORK — require overwhelming evidence for VERIFIED
- Every finding must cite a specific criterion number or concrete risk
- Do not skip any numbered criterion
```

### B5 — Feedback Loops

After both agents post their comments, read the issue comments to check the verdicts.

**If Reviewer returned CHANGES REQUESTED or Verifier returned NEEDS WORK:**

1. Post an `[ORCHESTRATOR]` comment summarizing the combined feedback and what the implementer needs to address.
2. Launch a new implementer agent with a prompt that includes the feedback comments and instructs it to:
   - Read the reviewer/verifier comments on the issue
   - Fix the cited items on the existing branch in the existing worktree
   - Push the fixes
   - Post a new completion comment with what was addressed
3. Once the fix comment appears:
   - If a PR already exists, update the PR body to reflect: what changed since last review, which feedback items were addressed, updated test/validation status. Do not leave stale checklists.
   - Re-launch the reviewer and/or verifier that had findings, pointing them at the updated branch.
4. Repeat until both return APPROVED and VERIFIED.

Keep all follow-up fixes and revalidation inside the same dedicated worktree for the issue.

### B6 — Completion & PR (you do this directly)

Once both agents have posted APPROVED and VERIFIED:

#### Step 1: Cross-check

1. Read all issue comments to review the full thread.
2. Cross-check the verifier's per-criterion checklist against the original kickoff criteria. If any criterion is FAIL or was skipped, do NOT declare complete — loop back.
3. Confirm the deployment target is correct.

#### Step 2: Build the acceptance evidence map

For each numbered acceptance criterion, compile:

| # | Criterion | Code Location | Test Coverage | Manual Validation | Deploy/Preview Evidence |
|---|-----------|---------------|---------------|-------------------|------------------------|
| 1 | ... | file:lines | test file + what it covers | concrete details | URL + project confirmed, or "no deployed preview — validated locally at X" |

This map becomes the core of the PR body. Do not allow any row to have blank evidence columns — if evidence is missing, loop back.

#### Step 3: Open the PR

Open a PR using `gh pr create` with the acceptance evidence baked into the body. Use the standard PR body format:

```
gh pr create --base main --head <BRANCH_NAME> --title "<PR_TITLE>" --body "$(cat <<'PREOF'
## Summary
- ...

## Acceptance Criteria
| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | ... | ✅ | ... |

## Test Plan
- [x] ...

## Validation
- **Deploy target:** <app/project>
- **Preview URL:** <URL or "no matching preview — validated locally">
- **Manual checks:** <specific details>

## Known Gaps
- ...

Closes #<NUMBER>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PREOF
)"
```

#### Step 4: Post completion comment

Post a completion comment on the issue linking to the PR:

```
gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
**[ORCHESTRATOR]** Completion Report

**What Changed:**
- ...

**Acceptance Evidence:**
| # | Criterion | Code | Tests | Validation | Deploy |
|---|-----------|------|-------|------------|--------|
| 1 | ... | ... | ... | ... | ... |

**Remaining Risks / Follow-ups:**
- ...

**PR:** #<PR_NUMBER>

---
_Status: DONE_
_Next: Awaiting CodeRabbit review on PR_
COMMENT
)"
```

#### Step 5: Proceed to CodeRabbit review loop

Immediately proceed to B7 (CodeRabbit Review Loop).

### B7 — CodeRabbit Review Loop

After a PR is opened (from either Mode A or Mode B), enter the CodeRabbit review loop. This phase is autonomous — no user interaction needed until CodeRabbit is satisfied.

#### Step 1: Wait for CodeRabbit

Poll for CodeRabbit's review using the `/loop` skill or `ScheduleWakeup`. Check every ~90 seconds:

```bash
gh api repos/<OWNER>/<REPO>/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "coderabbitai" or (.user.login | test("coderabbit"))) | {state: .state, body: .body}'
```

Also check for inline comments:

```bash
gh api repos/<OWNER>/<REPO>/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "coderabbitai" or (.user.login | test("coderabbit"))) | {path: .path, line: .line, body: .body}'
```

If no CodeRabbit review appears after 5 minutes, push an empty commit to trigger it:
```bash
git commit --allow-empty -m "trigger coderabbit review" && git push
```

#### Step 2: Classify and address CodeRabbit findings

Once CodeRabbit posts its review, read all review comments (both top-level and inline). Classify each finding:

- **Blocker** (`🔴 Major` / `⚠️ Potential issue`): Must fix.
- **Nitpick** (`🧹 Nitpick` / suggestions with proposed fixes): **Fix these too.** CodeRabbit nitpicks are typically high-quality improvements (security hardening, lifecycle rules, input validation). Apply the suggested fix unless it would materially expand scope or conflict with an architectural decision.
- **Informational** (acknowledged items with no concrete fix proposed, or items explicitly noted as "acceptable for POC"): Note but do not act.

**Default posture: fix it.** The goal is zero remaining actionable comments when the human reviewer opens the PR. Only skip a finding if you can articulate a specific reason (scope creep, architectural conflict, false positive).

#### Step 3: Fix findings

If there are blockers or worthwhile suggestions:

1. Read the specific files and lines CodeRabbit flagged.
2. Apply the fixes directly in the worktree.
3. Run validation (lint, typecheck, tests, `terraform validate` — whatever is relevant).
4. Commit with a descriptive message referencing the CodeRabbit feedback:
   ```
   fix: <what was fixed>

   Addresses CodeRabbit review feedback on PR #<PR_NUMBER>.
   ```
5. Push the branch.
6. Update the PR body to reflect what changed since the last review using `gh pr edit`.
7. Post a comment on the PR summarizing what was addressed:
   ```bash
   gh pr comment <PR_NUMBER> --body "$(cat <<'COMMENT'
   **[ORCHESTRATOR]** CodeRabbit Feedback — Round <N>

   **Addressed:**
   - <finding>: <what was done>
   - ...

   **Acknowledged (no action):**
   - <finding>: <why it's acceptable>
   - ...

   Pushed fixes. Awaiting next CodeRabbit review cycle.
   COMMENT
   )"
   ```

#### Step 4: Loop

After pushing fixes, return to Step 1 — wait for CodeRabbit to re-review the updated branch. Repeat until CodeRabbit has no remaining blockers.

**Loop limits:**
- Maximum **5 rounds** of CodeRabbit fixes. If still not clean after 5 rounds, stop and escalate to the user with a summary of remaining findings and your assessment of whether they're genuine issues or false positives.
- Consider CodeRabbit satisfied when: all blockers and nitpicks are either fixed or have a documented reason for skipping, and only informational/acknowledged items remain.

#### Step 5: Request human review

Once CodeRabbit is satisfied (no blockers remaining):

1. Update the PR body one final time with the complete acceptance evidence map and any changes made during the CodeRabbit loop.
2. Post a comment on the issue:
   ```
   gh issue comment <NUMBER> --body "$(cat <<'COMMENT'
   **[ORCHESTRATOR]** Ready for Human Review

   CodeRabbit is satisfied — no remaining blockers.

   **CodeRabbit rounds:** <N>
   **Changes made during review:** <summary of fixes>

   **PR:** #<PR_NUMBER>
   Ready for human review and merge.

   ---
   _Status: DONE_
   _Next: Human reviewer to approve and merge_
   COMMENT
   )"
   ```
3. Report to the user that the PR is ready for their review, linking the PR URL.

---

## Rules

- **The GitHub issue is the source of truth.** Do not add scope unless clearly labeled as a suggestion and approved by the user.
- **All agent communication goes through issue comments.** No agent receives another agent's output directly — they read the issue thread.
- **Each active issue gets its own dedicated worktree.** Branch isolation alone is not enough for concurrent agent work.
- **Each agent prompt must include instructions to read issue comments** for context, since agents cannot see the parent conversation.
- Every agent must post a structured comment with role label, status, and next action.
- Reviewer and verifier must tie every finding to acceptance criteria, correctness, or a concrete risk.
- Prefer minimal, production-appropriate changes over unnecessary refactors.
- Do not touch code unrelated to the issue.
- Do not let agents silently expand scope beyond their assigned issues.
- Call out blockers or ambiguity early.
- Do not declare success without explicit verification against every acceptance criterion.

## Safety Rules

- Do not start implementation until the plan is posted and approved on the issue.
- Do not skip the review or verification phases (except in Solo mode).
- **PRs are opened automatically** after internal review passes — this is expected behavior, not a safety concern. PRs are the mechanism for triggering CodeRabbit review.
- Do not **merge** PRs or push to main without explicit user instruction. Opening a PR is not merging.
- If the issue is ambiguous or missing critical information, stop and surface questions before proceeding.
- If the issue scope is clearly too large for a single pass, recommend splitting it and ask the user how to proceed.