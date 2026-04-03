# Glossary

<!-- Protocol: ~/.config/kimi/skills/glossary-protocol/SKILL.md (commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7, modified: 2026-04-07T12:03:05Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/glossary-protocol/SKILL.md
     Protocol commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7
     Protocol modified: 2026-04-07T12:03:05Z -->
---
name: glossary-protocol
description: Protocol for .agent/glossary.md — user-defined project terms and model interpretation (write only on explicit ask)
---

# Glossary Protocol

Per-repository **lexicon**: what phrases and names **mean for the agent** when the user uses them (scope of "read the repo", what counts as a "unit", etc.). **Not** general documentation — entries are **binding interpretive rules** for this workspace.


## Entry identity

Each entry uses:

- **`G<n>`** — global id: literal **`G`** + decimal integer (**`G1`**, **`G12`**). **One sequence per repository.** **`n` never decreases** and **is never reused** for a different meaning (supersede old row instead).

**Canonical heading (H2):**

```markdown
## G<n> — Short label (English)
```

**Body fields:**

```markdown
Status: ACTIVE | SUPERSEDED
User phrase(s): "verbatim user wording allowed in quotes" — optional EN gloss after
Model meaning:
- telegraphic bullets — what the agent shall do / infer / skip
Notes: optional scope, anti-examples, links to repo paths
Created: YYYY-MM-DDTHH:mm:ssZ
Updated: YYYY-MM-DDTHH:mm:ssZ
Supersedes: (optional G<id>)
Superseded by: (optional G<id> when Status SUPERSEDED)
```

### Allocating the next `G<n>`

1. Read **all** of `.agent/glossary.md` (every `## G` heading).
2. Match `## G<number> —`; **`max`** = largest **number**; next id = **`G{max+1}`** or **`G1`** if none.
3. **Never** reuse **n** for a new concept.

## File structure

```
repo-root/
└── .agent/
    └── glossary.md        # User-authorized terminology + model interpretation
```

## Read on session start

1. Read **`.agent/glossary.md` in full** **before** interpreting vague repo-wide instructions (e.g. "read everything", "whole repo", "all code") or domain terms that may appear in entries.
2. When **User phrase(s)** matches (including cross-language intent, not only exact string), apply **Model meaning** for this session.
3. Acknowledge optionally: "Glossary: N active entries".

## Write gate (**mandatory**)

**Do not** add, rename, merge, or materially edit glossary entries **unless** the user **explicitly** authorizes a glossary write in the current turn or a turn you are implementing. **Casual chat, examples, or your own inference are not enough.**

Treat as **authorized** when the user clearly intends to **record / define / fix** a project term, e.g. (non-exhaustive):

- RU: «запиши в глоссарий», «добавь в глоссарий», «зафиксируй понятие», «новое определение:», «зафиксируй термин»
- EN: `add to glossary`, `record in glossary`, `define for glossary`, `put this in the glossary`, `glossary entry:`

If ambiguous — **ask once** whether to persist; **do not** write until confirmed.

**Updates** to an existing **G<n>** (narrower/wider meaning) require the same explicit authorization.

## Save Work / Persist state (this protocol only)

When a **checkpoint** runs (see **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*):

1. **If** this session had **no** glossary **Write gate** opens **and** you made no in-session edit to `.agent/glossary.md` → **leave file unchanged** (step still "done" as no-op).
2. **If** the user opened the gate: ensure new or updated **`G<n>`** blocks are complete (**Model meaning**, **Created**/**Updated** UTC **`Z`**, **Status**); mark **SUPERSEDED** + **`Superseded by`** when replacing a prior definition.
3. **Never** delete historical **SUPERSEDED** rows without user direction — append-only evolution.

This step runs **after** **style-protocol** when included in the template order (**sixth**), **only when** a glossary write was authorized or pending edits exist.

## Hygiene

- Prefer **actionable** **Model meaning** (what to read, what to skip, where "meaning" lives vs boilerplate).
- Link repo areas with **repo-relative** paths in **Notes** when useful.
- **Respect .gitignore** — definitions must not require reading ignored secrets.

## Example

```markdown
## G1 — Read whole repository
Status: ACTIVE
User phrase(s): "read the whole repo", "прочитай весь репозиторий"
Model meaning:
- locate where **semantic** work lives (domain code, main docs); **do not** re-scan trivial config (e.g. linter-only files) on every pass unless task is lint/config
- prefer architecture + entry points + glossary over exhaustive file-by-file re-read
Created: 2026-04-03T20:00:00Z
Updated: 2026-04-03T20:00:00Z
```
<!-- END OF PROTOCOL COPY -->

<!--
## G1 — Short English label
Status: ACTIVE
User phrase(s): "example phrase"
Model meaning:
- what the agent does when user uses this phrase
Notes: optional
Created: 2026-04-07T13:18:30Z
Updated: 2026-04-07T13:18:30Z
-->
