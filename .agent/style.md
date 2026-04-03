# Code style ledger

<!-- Protocol: ~/.config/kimi/skills/style-protocol/SKILL.md (commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7, modified: 2026-04-07T12:03:05Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/style-protocol/SKILL.md
     Protocol commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7
     Protocol modified: 2026-04-07T12:03:05Z -->
---
name: style-protocol
description: Protocol for maintaining .agent/style.md empirical code-style ledger (refs + observations)
---

# Style Protocol

Ledger of **empirical** project code style: what to **mirror** when writing/editing project source (naming, comment habits, error shapes, test layout, etc.). **Not** a dump of linter config — formatters enforce mechanics; this file captures **patterns agents should copy** from real neighbors, with **repo-relative** `Refs`.

## Style note identity

Each note has:

- **`S<n>`** — global id: literal **`S`** + decimal integer, **no leading zeros** (`S1`, `S12`). **One sequence per repository.** **`n` never decreases and is never reused** when a note is completed or cancelled (same immutability spirit as `T<n>` in task-protocol).

**Canonical heading (H2):**

```markdown
## S<n> — Short title
```

**Body fields** (order flexible; include all that apply):

```markdown
Status: ACTIVE | DONE | CANCELLED
Created: YYYY-MM-DDTHH:mm:ssZ
Last touched: YYYY-MM-DDTHH:mm:ssZ
Refs: `path/under/repo/a.ext`, `path/b.ext`
Observation:
- telegraphic bullets — what to mimic
Applies_to: optional `glob/` or path prefix this note is about
```

For **DONE**, add `Completed: …` and short `Summary:` when moving to **# Completed style notes**.

### Allocating the next `S<n>`

1. Read **all** of `.agent/style.md` (**# Active style notes**, **# Completed style notes**, any stray `## S` lines).
2. Collect every **`S` + digits** in an H2 line matching `## S<n> —`.
3. **`max`** = maximum numeric part; **next id** = `S{max + 1}` (or **`S1`** if none).
4. **Never** reuse a number.

## File structure

```
repo-root/
└── .agent/
    └── style.md           # Empirical code-style ledger
```

## Read on Session Start

1. **When you will edit, add, or review project source code** (any non-`.agent/` path under the repo): read **`.agent/style.md`** before writing patches — at least **# Reference anchors** and **all `Status: ACTIVE` notes** whose `Applies_to` matches the area you touch (if absent, read all ACTIVE notes; use **Refs** as secondary anchors).
2. **Still read neighboring source files** in the same package/directory — the ledger **supplements** local mirroring, does not replace it.
3. Acknowledge briefly: "Style ledger: N active notes, M anchors" (optional).

## Write triggers (mandatory when coding)

**During any session with substantive code work:**

- After you notice a **repeatable** pattern not already captured (or under-captured), **append** a new **`S<n>`** or extend an existing note's **Observation** / **Refs** (update **`Last touched`**).
- Prefer **Refs** with **2+ files** when the pattern is structural; one file is OK for rare isolated idioms.
- **Do not** log trivia formatters fix alone unless the team's **pre-format** habit matters (e.g. import grouping conventions not auto-fixed).

## Completing / cancelling

- **DONE:** move the whole `## S<n> — …` block to **`# Completed style notes`**; set `Status: DONE`, add **`Completed`** + **`Summary`**.
- **CANCELLED:** move to Completed (or keep in Active with `Status: CANCELLED` + reason in **Observation**) — **do not** reuse **`n`**.

## Save Work / Persist state (this protocol only)

When a **checkpoint** runs (triggers and order: **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*), this protocol's **sole** responsibility is **`.agent/style.md`**.

1. Ensure **ACTIVE** notes reflect what you learned this session (**Observation** / **Refs** / **Applies_to** / **`Last touched`**).
2. Move finished notes to **Completed** per above.
3. **Do not** delete historical **DONE** notes — ledger is append-only except corrections.

This step runs **after** **knowledge-protocol** in the template order (fifth file). **glossary-protocol** (sixth) runs **after** this step **only when** a glossary write was authorized or pending edits exist.

## Hygiene

- **Repo-relative `Refs`** only — no secrets, no machine-specific paths.
- **Respect .gitignore** — do not point **Refs** at ignored paths unless the user explicitly works there.
- **English** in **Observation** / **Summary** / titles per hub **§ On-disk language**.

## Example

```markdown
## S1 — HTTP handler error JSON shape
Status: ACTIVE
Created: 2026-04-03T20:00:00Z
Last touched: 2026-04-03T20:15:00Z
Refs: `src/api/handlers/foo.ts`, `src/api/handlers/bar.ts`
Observation:
- errors return `{ error: string, code: string }` not thrown bare strings
- log line prefix `[api]` before `logger.error`
Applies_to: `src/api/handlers/`
```
<!-- END OF PROTOCOL COPY -->

## Reference anchors

<!--
Representative files agents should skim when entering an area (2-8 paths). Example:
- `src/main.rs` — binary entry
- `tests/integration/foo.rs` — integration test layout
-->

# Active style notes

<!--
## S1 — Short title
Status: ACTIVE
Created: 2026-04-07T13:18:30Z
Last touched: 2026-04-07T13:18:30Z
Refs: `path/under/repo/file.ext`
Observation:
- what to mimic (telegraphic EN)
Applies_to: optional `path/prefix/` or glob
-->

# Completed style notes

<!--
When done:
## S1 — Short title
Status: DONE
Created: YYYY-MM-DDTHH:mm:ssZ
Last touched: YYYY-MM-DDTHH:mm:ssZ
Completed: YYYY-MM-DDTHH:mm:ssZ
Summary: why retired / superseded
Refs: `...`
Observation:
- ...
-->
