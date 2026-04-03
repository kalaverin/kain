# Active Tasks

<!-- Protocol: ~/.config/kimi/skills/task-protocol/SKILL.md (commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7, modified: 2026-04-07T12:03:05Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/task-protocol/SKILL.md
     Protocol commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7
     Protocol modified: 2026-04-07T12:03:05Z -->
---
name: task-protocol
description: Protocol for maintaining .agent/tasks.md task tracking and work item management
---

# Task Protocol

Task tracking system for managing active, blocked, and completed work items across AI agent sessions.

## Task identity (scope + global ID)

Every task has:

- **`scope`** — one English word, **lowercase**, Latin alphabet only: the *kind* of work (feature, bugfix, chore, audit, …).
- **`T<n>`** — **global** task identifier: literal **`T`** plus a decimal integer **`n`** with **no leading zeros** (`T1`, `T12`, `T1042`). **One sequence per repository:** backlog, active, completed, and cancelled entries all share the same counter. **`n` never decreases and is never reused** (cancelled tasks keep their `T<n>` in the file).

**Canonical heading (H2):**

```markdown
## [PRIORITY] <scope> T<n> — Short title
```

For completed work, keep **`[DONE]`** as the priority slot:

```markdown
## [DONE] <scope> T<n> — Short title
```

**User and agent references:** refer to a task by **`T<n>`** (e.g. `task T12`, `unblock T7`).

### Task `Status` values (placement-aware)

| `Status` | Typical section | Notes |
|----------|-----------------|--------|
| `NOT_STARTED` | Active or Backlog | Queued; may sit in either section per repo convention |
| `IN_PROGRESS` | **Active Tasks** only | Primary WIP row |
| `BLOCKED` | Active (or Backlog if not yet started) | Use **`Blockers:`** field |
| `INTERRUPTED` | **Active Tasks** only | Session ended mid-task |
| `POSTPONED` | **Backlog only** | **Hard rule:** any row with `Status: POSTPONED` **must** appear **under `# Backlog`** (after that heading). **Never** under `# Active Tasks`. Resume by moving to Active (or keep in Backlog) and set `NOT_STARTED` / `IN_PROGRESS` as appropriate |

Other terminal states: `DONE` (Completed section), `CANCELLED` (Cancelled section or noted inline).

### Allowed `scope` values (closed list)

Expand this list only by an explicit project decision (e.g. row in `.agent/decisions.md`).

| `scope` | Use for |
|---------|---------|
| `feat` | New capability or materially new behavior |
| `fix` | Bug fix, regression, incorrect behavior |
| `chore` | Small cleanup, polish, trivial maintenance without contract change |
| `audit` | Consistency review, alignment, repo/process audit follow-ups |
| `docs` | Primarily documentation / agent-facing prose |
| `refactor` | Same external behavior, structure or code reorganized |
| `infra` | CI, scripts, environment, template sync mechanics |
| `research` | Exploration without committing to a deliverable in the same pass |
| `meta` | Task system, templates, hub rules about how agents work |
| `test` | Failing tests, test coverage gaps, test infrastructure fixes |

### Allocating the next `T<n>`

1. Read **all** of `.agent/tasks.md` (every section: Backlog, Active, Completed, Cancelled, **Archive** if present).
2. Collect every **`T` + digits** that appears in an **H2 task heading** matching `## [` … `] <scope> T<n> —` **or in Archive table rows** matching `| T\d+ |` (pattern `\bT\d+\b` in those lines is sufficient).
3. Let **`max`** be the maximum numeric part. **Next id is `T{max + 1}`** (or **`T1`** if none exist).
4. **Never** reuse a number. **Do not** renumber existing tasks except during a documented, one-off migration.

## File Structure

```
repo-root/
└── .agent/
    └── tasks.md           # Active and completed task tracking
```

## Read on Session Start

1. **Read .agent/tasks.md** — check for:
   - `[IN_PROGRESS]` tasks — continue work
   - `[BLOCKED]` tasks — check if blockers resolved
   - `[INTERRUPTED]` tasks — restore context first
   - **`# Backlog`** rows with `Status: POSTPONED` — intentionally deprioritized; do not treat as active WIP
   - `[HIGH]` / `[CRITICAL]` tasks — prioritize these
   - Task **`T<n>`** if the user refers to a number
   - **`# Archive`** section — check archived IDs (still count for max T)

2. Acknowledge: "Tasks loaded: [X active, Y blocked, Z interrupted, W postponed in backlog, V archived], max T…"

## Task Lifecycle

### 1. Creating a New Task

When user requests work or you identify a new task:

1. Allocate **`T<n>`** per **Allocating the next `T<n>`** above.
2. Choose **`scope`** from the allowed list.
3. Write:

```markdown
## [PRIORITY] <scope> T<n> — Short title
Status: NOT_STARTED
Created: YYYY-MM-DDTHH:mm:ssZ
Last worked: YYYY-MM-DDTHH:mm:ssZ
Next step: (concrete first action)
Notes: (optional context)
```
(UTC only; format per `~/.config/kimi/prompts/settings.md` § Timestamps.)

Priority levels (active / not done): `CRITICAL` > `HIGH` > `MEDIUM` > `LOW`

### 2. Starting Work

Change status and update timestamp:

```markdown
Status: IN_PROGRESS
Last worked: YYYY-MM-DDTHH:mm:ssZ
```
(use current UTC instant per `~/.config/kimi/prompts/settings.md` § Timestamps)

### 3. Blocking / Unblocking

If blocked:
```markdown
Status: BLOCKED
Blockers: (specific what's needed to unblock; cite other tasks as T<id> if useful)
```

When unblocked:
```markdown
Status: IN_PROGRESS
Blockers: (cleared: what changed)
```

### 4. Task Interruption

If session ends mid-task:
```markdown
Status: INTERRUPTED
Last worked: YYYY-MM-DDTHH:mm:ssZ
Notes: (current state, what's next, any temp files/branches)
```
(per `~/.config/kimi/prompts/settings.md` § Timestamps)

### 5. Postponing (Backlog-only)

When work is deliberately deferred (not blocked—just out of scope for now):

1. Move the task **heading + body** so the row lives **under `# Backlog`** (never under `# Active Tasks`).
2. Set:

```markdown
Status: POSTPONED
Last worked: YYYY-MM-DDTHH:mm:ssZ
Postponed: (why / until when / trigger to revisit)
Next step: (first action when un-postponed)
```

3. **Do not** use `POSTPONED` for tasks that remain under **Active**—that placement is invalid; use `BLOCKED` or move to Backlog first.

**Resume from POSTPONED:** move row to `# Active Tasks` if it becomes WIP, then set `Status: NOT_STARTED` or `IN_PROGRESS` and clear or shorten **Postponed:** once no longer deferred.

### 6. Completing a Task

Move to `# Completed Tasks` section. **Keep `T<n>` and `scope` in the heading.**

```markdown
## [DONE] <scope> T<n> — Short title
Completed: YYYY-MM-DDTHH:mm:ssZ
Summary: (what was accomplished, key outcomes)
```

### 7. Cancelling / splitting

- **Cancelled:** move to a `# Cancelled Tasks` section (or keep with `Status: CANCELLED`) — **same `T<n>`**, note reason; **do not** reuse `n`.
- **Split:** new work gets a **new** `T`; in the original task's **Notes**: `split → T<new>`.

### 8. Archiving completed/cancelled tasks (user-commanded only)

When user explicitly requests to archive/cleanup/compress/compact completed and cancelled tasks (trigger phrases: "archive tasks", "clean up completed", "clear task log", "compress finished tasks", "archive done", "compact tasks", "archive completed"):

1. **Collect** all tasks from `# Completed Tasks` and `# Cancelled Tasks` sections
2. **Sort** tasks: first by **Date** (ascending), then by **ID** (ascending, numeric) — oldest first, stable within same date
3. **Convert** to **compressed table format** (ID, Date, Scope, Status, Title) in sorted order
4. **Append** to `# Archive` section at end of file (create if not exists)
5. **Preserve** all `T<n>` identifiers — archived IDs **still count** for next allocation
6. **Clear** Completed/Cancelled sections (leave headers, remove full task bodies)

**Archive table format:**

```markdown
# Archive

<!-- Compressed history — T<n> preserved for ID continuity. Never remove rows. -->

| ID | Date | Scope | Status | Title |
|----|------|-------|--------|-------|
| T1 | 2026-03-20 | feat | DONE | Initialize project documentation |
| T3 | 2026-03-24 | fix | CANCELLED | Login timeout bug |
| T5 | 2026-03-24 | infra | DONE | Set up CI/CD pipeline |
| T7 | 2026-03-25 | test | DONE | Fix flaky auth test |
```

**Critical rules:**
- Archive is **append-only** — never edit or delete archived rows
- When allocating next `T<n>`, scan **Archive section too** — include all `| T\d+ |` matches
- Archive grows indefinitely; old rows remain forever
- **User command required** — do not auto-archive without explicit request
- After archiving, report: "Archived N completed/cancelled tasks to # Archive. Max T<n> unchanged (still T<max>)."

## Save Work / Persist state (this protocol only)

When a **checkpoint** runs (triggers and order: **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*), this protocol's **sole** responsibility is **`.agent/tasks.md`**.

1. Set each **active** task's `Status` to match reality (`DONE`, `IN_PROGRESS`, `INTERRUPTED`, `BLOCKED`, `NOT_STARTED`, `POSTPONED`, etc.). If any row has `Status: POSTPONED`, **confirm** it sits **under `# Backlog`**; relocate the row if not.
2. Update **`Last worked`** to the current UTC instant in **`YYYY-MM-DDTHH:mm:ssZ`** (per `~/.config/kimi/prompts/settings.md` § Timestamps) for every task you touched during this checkpoint.
3. If work stops mid-task: use `INTERRUPTED` and expand **Notes** with the next concrete step and WIP context (branch, failing paths, temp artifacts).
4. **Do not** change **`T<n>`** or `scope` in headings unless correcting a documented error.

This step runs **after** **memory-protocol** and **decision-protocol** in the template order; then **knowledge-protocol**, then **style-protocol**, then **glossary-protocol** when its **Write gate** applies — do not skip knowledge unless the skip rule applies.

## Task Hygiene Rules

- **Respect .gitignore** — Never create tasks that involve modifying files in `.gitignore`
  (except `.agent/` directory). Always check .gitignore first.
- **One task at a time** — avoid multiple IN_PROGRESS tasks
- **POSTPONED lives in Backlog** — never an Active Tasks row; violates protocol → move under `# Backlog` on sight
- **Specific next steps** — vague "continue work" is not helpful
- **Clear blockers** — state exactly what's needed to unblock; reference **`T<id>`** for sibling tasks
- **Archive completed** — move to Completed Tasks, don't delete
- **Archive on user command only** — never auto-archive; wait for explicit trigger phrase
- **Stable IDs** — `T<n>` is immutable for the life of that task row; titles may be edited

## Cross-Session Continuity

- On resume: read `[INTERRUPTED]` tasks first
- Restore context from Notes before accepting new work
- Update status from INTERRUPTED → IN_PROGRESS when resuming

## Example Task Entries

### Active Task
```markdown
## [HIGH] feat T24 — Implement user authentication
Status: IN_PROGRESS
Created: 2026-03-25T09:00:00Z
Last worked: 2026-03-27T14:00:00Z
Next step: Add JWT token validation middleware
Notes: OAuth flow working; blocked by T23 (API keys). Branch `feat/auth`.
```

### Blocked Task
```markdown
## [MEDIUM] refactor T31 — Optimize database queries
Status: BLOCKED
Created: 2026-03-26T10:00:00Z
Last worked: 2026-03-26T16:00:00Z
Blockers: Waiting for DBA to provide query execution plan
Next step: Review execution plan and identify indexes
```

### Interrupted Task
```markdown
## [HIGH] fix T28 — Refactor API error handling
Status: INTERRUPTED
Created: 2026-03-20T11:00:00Z
Last worked: 2026-03-27T18:00:00Z
Next step: Update test suite for changed signatures
Notes: 12/15 endpoints fixed. Tests failing in `tests/test_api_v2.py`. Branch `fix/api-errors`.
```

### Completed Task
```markdown
## [DONE] infra T22 — Set up CI/CD pipeline
Completed: 2026-03-24T12:00:00Z
Summary: Configured GitHub Actions with test, lint, and deploy jobs. Pipeline runs on PR and merge to main.
```

### Postponed (Backlog)
```markdown
## [LOW] research T33 — Spike on alternative SDK
Status: POSTPONED
Created: 2026-03-20T09:00:00Z
Last worked: 2026-03-27T10:00:00Z
Postponed: Await product decision on vendor; revisit next quarter
Next step: Re-read vendor docs and prototype auth flow
Notes: (optional)
```
**(Placement)** this block must appear only under **`# Backlog`**, not under **`# Active Tasks`**.
<!-- END OF PROTOCOL COPY -->

## [HIGH] Initialize project documentation
Status: DONE
Created: 2026-04-03
Last worked: 2026-04-03
Next step: Review .agent/intro.md for accuracy
Notes: Initial setup by repo-init-agent

## [HIGH] Populate knowledge.md with codebase analysis
Status: DONE
Created: 2026-04-03
Last worked: 2026-04-03
Next step: Keep knowledge.md synchronized as codebase evolves
Notes: Initial knowledge capture per knowledge-protocol

<!-- 
Task format for new tasks:
## [PRIORITY] Task Name
Status: [NOT_STARTED/IN_PROGRESS/BLOCKED/DONE/INTERRUPTED]
Created: YYYY-MM-DD
Last worked: YYYY-MM-DD
Blockers: (if BLOCKED)
Next step: (concrete action)
Notes: (optional context)
-->

# Completed Tasks

## [DONE] Initialize project documentation
Completed: 2026-04-03
Summary: Created .agent/intro.md, AGENTS.md symlink, .agent/history.md, .agent/tasks.md, .agent/decisions.md, and .agent/knowledge.md with initial repository analysis.

## [DONE] Populate knowledge.md with codebase analysis
Completed: 2026-04-03
Summary: Analyzed src/kain/ structure, mapped dependency graph, documented class hierarchy, and recorded initial gotchas and insights.

<!-- 
Move tasks here when done:
## [DONE] Task Name
Completed: YYYY-MM-DD
Summary: (what was accomplished)
-->
