# Decision Log

<!-- Protocol: ~/.config/kimi/skills/decision-protocol/SKILL.md (commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7, modified: 2026-04-07T12:03:05Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/decision-protocol/SKILL.md
     Protocol commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7
     Protocol modified: 2026-04-07T12:03:05Z -->
---
name: decision-protocol
description: Protocol for maintaining .agent/decisions.md architectural decision log
---

# Decision Protocol

Architectural Decision Record (ADR) system for tracking decisions, their rationale, and status across AI agent sessions.

## File Structure

```
repo-root/
└── .agent/
    └── decisions.md       # Decision log with status
```

## Read on Session Start

1. **Read .agent/decisions.md** — check for:
   - `ACTIVE` decisions — current constraints/guidelines
   - `PENDING` decisions — awaiting input
   - `SUPERSEDED` decisions — understand history

2. Acknowledge: "Decisions loaded: [X active, Y pending]"

## Decision Lifecycle

### 1. Recording a New Decision

When an architectural/approach decision is made:

```markdown
| YYYY-MM-DDTHH:mm:ssZ | Decision text | Why this choice | ACTIVE | - |
```

Use a **full UTC instant** per `~/.config/kimi/prompts/settings.md` § Timestamps.

Add to the table in chronological order (newest last).

### 2. Save Work / Persist state (this protocol only)

When a **checkpoint** runs (triggers and order: **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*), this protocol's **sole** responsibility is **`.agent/decisions.md`**.

1. For each **new** architectural or approach decision from this session: add a row to the decision table with **rationale** and status **ACTIVE** (or **PENDING** if blocked).
2. If a decision **supersedes** an older one: mark the old row **SUPERSEDED** and add the replacement row per **§ 4. Superseding Decisions** below.
3. Keep narrative under **Decisions:** in `.agent/history.md` (memory-protocol) aligned with the table — the table is the long-term source of truth.

After finishing this file, continue the checkpoint with **task-protocol**, then **knowledge-protocol**, then **style-protocol**, then **glossary-protocol** (conditional per that skill) if not already done; do not skip **memory-protocol** — it runs first in the template order.

### 3. Pending Decisions

When a decision needs to be made but is blocked:

```markdown
## Pending Decisions

- Use PostgreSQL or MySQL? (raised: 2026-03-25T12:00:00Z, context: Need to evaluate JSON support and performance for our use case)
```

Move to table with status `PENDING` if awaiting external input.

### 4. Superseding Decisions

When a decision changes:

1. Mark old decision:
```markdown
| 2026-03-25T14:00:00Z | ~~Use Redis for sessions~~ | Overkill for MVP | SUPERSEDED | Use in-memory cache |
```

2. Add new decision:
```markdown
| 2026-03-27T09:15:00Z | Use in-memory cache for sessions | Simpler, sufficient for current scale | ACTIVE | - |
```

3. In `.agent/history.md` session, note: "Superseded: [old] → [new]"

## Decision Hygiene Rules

- **Respect .gitignore** — Decisions should not involve modifying files in `.gitignore`
  (except `.agent/` directory). Check .gitignore before making file-related decisions.
- **Every significant decision gets recorded** — if it affects future work, log it
- **Rationale is mandatory** — future you/others need to understand WHY
- **Link related decisions** — use "Superseded By" column
- **Review PENDING regularly** — don't let decisions stall indefinitely
- **Mark stale decisions** — use ~~strikethrough~~ for SUPERSEDED

## What to Record

### Record These
- Technology choices (framework, database, library)
- Architecture patterns (microservices vs monolith, REST vs GraphQL)
- Coding conventions (enforced via this decision)
- Tool choices (linter, formatter, CI/CD)
- Approach choices (sync vs async, eager vs lazy loading)

### Don't Record
- Obvious choices (use git for version control)
- Temporary workarounds with TODOs in code
- Personal preferences without project impact

## Cross-Session Continuity

- Reference ACTIVE decisions when making similar choices
- Check if PENDING decisions are now resolvable
- When questioning a decision, check rationale first

## Example Decision Table

```markdown
| Date | Decision | Rationale | Status | Superseded By |
|------|----------|-----------|--------|---------------|
| 2026-03-20T10:00:00Z | Use FastAPI for API framework | Modern async support, auto-generated docs, type hints | ACTIVE | - |
| 2026-03-22T11:00:00Z | Use SQLAlchemy 2.0 with async | Official async support, type-safe queries | ACTIVE | - |
| 2026-03-25T14:00:00Z | ~~Use raw SQL for complex queries~~ | Hard to maintain, no type safety | SUPERSEDED | Use SQLAlchemy Core |
| 2026-03-27T15:00:00Z | Use SQLAlchemy Core for complex queries | Balance of control and type safety | ACTIVE | - |
| 2026-03-28T16:00:00Z | Use pytest with async fixtures | Standard for FastAPI testing, good async support | ACTIVE | - |
```

## Integration with Session History

When decisions are made in a session, include in `.agent/history.md`:

```markdown
**Decisions:**
- Use PostgreSQL over MySQL (better JSON support for our document storage needs)
- Adopt repository pattern for data access (easier testing, clearer boundaries)
```

These should then be added to `.agent/decisions.md` for long-term tracking.
<!-- END OF PROTOCOL COPY -->

| Date | Decision | Rationale | Status | Superseded By |
|------|----------|-----------|--------|---------------|

<!--
Format:
| YYYY-MM-DD | Decision text | Why this choice | ACTIVE/SUPERSEDED/PENDING | (if superseded) |
-->

## Pending Decisions

<!--
Decisions that need to be made:
- Question? (raised: YYYY-MM-DD, context: link or brief description)
-->

## Decision Templates

When superseding a decision:
1. Mark old decision as SUPERSEDED
2. Add link to new decision in "Superseded By"
3. Add new decision row with ACTIVE status

Example:
```markdown
| 2026-03-25 | ~~Use Redis for sessions~~ | Overkill for MVP | SUPERSEDED | Use in-memory cache |
| 2026-03-27 | Use in-memory cache for sessions | Simpler, sufficient for current scale | ACTIVE | - |
```
