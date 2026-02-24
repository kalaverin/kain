# Work History — 2026-04-07T13:18:30Z

<!-- Protocol: ~/.config/kimi/skills/memory-protocol/SKILL.md (commit: d7b218d442ccc2d8229bbc567649af803b333b28, modified: 2026-04-03T19:39:37Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/memory-protocol/SKILL.md
     Protocol commit: d7b218d442ccc2d8229bbc567649af803b333b28
     Protocol modified: 2026-04-03T19:39:37Z -->
---
name: memory-protocol
description: Protocol for maintaining .agent/history.md session logs and cross-session continuity
---

# Memory Protocol

Cross-session memory system for AI agents working on long-running projects.

## File Structure

```
repo-root/
├── AGENTS.md              # repo root: symlink → .agent/intro.md (canonical; not reverse)
└── .agent/
    ├── decisions.md       # Decision log with status
    ├── history.md         # Work history (append-only)
    ├── style.md           # Empirical code-style ledger (S<n>)
    ├── glossary.md        # User-defined terms → model meaning (G<n>)
    └── tasks.md           # Current and queued tasks
```

## Related Protocols

- **Hub settings** — `~/.config/kimi/prompts/settings.md`: read **in full** before this protocol in hub-driven work; **supreme** — **nothing** overrides it. Session bodies you append: **§ On-disk language** (English telegraphic).
- **task-protocol** — How to manage .agent/tasks.md
- **style-protocol** — How to maintain .agent/style.md
- **glossary-protocol** — How to maintain .agent/glossary.md
- **decision-protocol** — How to maintain .agent/decisions.md
- **knowledge-protocol** — How to maintain .agent/knowledge.md

## Read on Session Start

1. **Read `AGENTS.md` at repo root** or **`.agent/intro.md`** (same file; root entry is symlink → intro) — understand project context
2. **Read .gitignore** — note all ignored files/patterns (always ignore these, except `.agent/`)
3. **Smart read .agent/history.md** — see below for efficient reading
4. **Read .agent/tasks.md** — check active/queued tasks (see task-protocol)
5. **Read .agent/decisions.md** — understand active decisions (see decision-protocol)
6. Acknowledge: "Memory loaded: [N sessions, last session header, X active tasks]" — headers use UTC `Z` per `~/.config/kimi/prompts/settings.md` § Timestamps

## Smart Memory Reading

To avoid loading excessive context:

1. **Always read** first 50 lines (initialization + project summary)
2. **Read last 5 sessions** fully
3. **For older sessions** — read only:
   - Session header (date/time)
   - **Completed** section (brief)
   - **Decisions** section (with status)
   - Skip detailed discoveries and file lists
4. **Skip entirely** sessions marked with `[ARCHIVED]` in header

## Write on Session End (or after non-trivial task)

Also run this append at every **Persist state** checkpoint (explicit save, pivot phrases, or non-trivial work done — full list under **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*). Same checkpoint order: **history → decisions → tasks → knowledge**; this protocol is **first**.

**Append at the END of Sessions section** (before `## Archive` or before the 
`<!-- Agent appends new sessions -->` comment):

Session headers MUST follow **Timestamps** in `~/.config/kimi/prompts/settings.md` (`YYYY-MM-DDTHH:mm:ssZ`), e.g. `### [2026-04-03T19:41:14Z]`.

```markdown
### [YYYY-MM-DDTHH:mm:ssZ]
**Completed:** <what was actually done>
**Discovered:** <non-obvious findings, gotchas, env quirks>
**Decisions:** <architectural/approach decisions made and why>
**Open:** <unresolved questions, blocked items>
**Modified files:** <list of changed files>
```

**Order:** Sessions are stored chronologically from OLDEST to NEWEST (new sessions 
added at the END of the Sessions section).

## Important Rules

- **Respect .gitignore** — Never read, modify, or suggest changes to files listed in 
  `.gitignore`, except `.agent/` directory which must always be tracked and maintained
- **Never overwrite** existing entries in history.md — append only
- **Auto-archive**: If .agent/history.md exceeds 200 lines, move old sessions to Archive

## Save Work / Persist state (this protocol only)

When a **checkpoint** runs (triggers and order: **`~/.config/kimi/templates/agent.md`** or **`.agent/intro.md`** § SKILLS → *Persist state*), this protocol's **sole** responsibility is **`.agent/history.md`**.

1. Follow **Write on Session End (or after non-trivial task)** above: append **one** new session entry at the end of `## Sessions`.
2. In the entry's **Decisions:** lines, summarize outcomes at **session** level only. Full ADR rows and statuses belong in **decision-protocol** during the same checkpoint, not duplicated as prose here.

After finishing this file, continue the same checkpoint with **decision-protocol**, then **task-protocol**, then **knowledge-protocol**, then **style-protocol**, then **glossary-protocol** (the latter only when its **Write gate** applied — see that skill) — do not stop early.

## Memory Hygiene Rules

- **Never overwrite** existing entries — append only
- **Auto-archive**: If .agent/history.md exceeds 200 lines:
  1. Create `## Archive` section at end if not exists
  2. Move sessions older than 10 most recent to Archive
  3. Summarize archived session to 3-4 lines max
- **Stale marking**: Mark superseded decisions with ~~strikethrough~~
- **Do NOT store**: temp debug output, trivial one-liners, obvious facts
- **Do store**: non-obvious findings, architectural decisions, gotchas

## Session Summary (Auto-Generated)

Every 10 sessions, prepend to .agent/history.md:

```markdown
## Session Summary (as of YYYY-MM-DDTHH:mm:ssZ)
- **Total sessions:** 47
- **Active decisions:** 5
- **Last focus:** API refactoring
- **Key patterns:** FastAPI, SQLAlchemy, pytest
- **Common gotchas:**
  - Database needs re-creation between tests
```

## Cross-Session Continuity

- If task was interrupted: mark in `.agent/tasks.md` as `[INTERRUPTED]` with last known state (see task-protocol)
- On resume: read `[INTERRUPTED]` tasks first, restore context before accepting new tasks

## Example Session Entry

```markdown
### [2026-03-27T14:30:00Z]
**Completed:**
- Implemented user authentication middleware
- Added JWT token validation
- Created login/logout endpoints

**Discovered:**
- FastAPI dependency injection requires `Depends()` on every protected route
- The test database needs to be re-created between test runs (no rollback support)

**Decisions:**
- Use `python-jose` instead of `PyJWT` for better algorithm support
- Store refresh tokens in Redis with 7-day TTL

**Open:**
- How to handle token refresh in WebSocket connections?
- Rate limiting strategy for login attempts

**Modified files:**
- `app/auth/middleware.py`
- `app/auth/router.py`
- `tests/test_auth.py`
```
<!-- END OF PROTOCOL COPY -->

You MUST append a session entry after completing any work

## Initialization
- Repo scaffolded: 2026-04-03
- Current commit: a094e2a3564fa75d2b91c85542a3b567e889e0c3
- `.agent/intro.md`: created (AGENTS.md symlink)
- Initialized by: repo-init-agent

## Session Summary
<!-- Generated every 10 sessions -->
- Total sessions: 19
- Active decisions: 0
- Last focus: Type annotations and test coverage across all modules
- Key patterns: Pure stdlib utility library; modern descriptor package in `properties/`; legacy `descriptors.py` still exported

## Sessions

<!-- Agent appends new sessions here -->
<!-- Format:
### [YYYY-MM-DD HH:MM]
**Completed:** <what was actually done>
**Discovered:** <non-obvious findings, gotchas, env quirks>
**Decisions:** <architectural/approach decisions made and why>
**Open:** <unresolved questions, blocked items>
**Modified files:** <list of changed files>
-->

### [2026-04-02 18:35]
**Completed:**
- Wrote comprehensive unit tests for `kain.descriptors` (`@pin`, `@pin.native`, `@pin.cls`, `@pin.any`, `@pin.pre`, `@pin.post`, `@class_property`, `@mixed_property`)
- Total 32 tests covering caching behavior, async/future caching, TTL expiration, manual set/delete, class vs instance access, inheritance with child-aware class passing, and mixed-level caching
- Updated tests after user fixed the `Who(..., addr=True)` bug in descriptors — `@pin` now correctly raises `ContextFaultError`/`ReadOnlyError` instead of `TypeError`
- All 32 tests pass (`pytest`)

**Discovered:**
- `pin.pre` / `pin.post` still create empty `__instance_memoized__` / `__class_memoized__` dicts even when the value is not cached; only the specific key is omitted
- `class_property` passes the accessed class (e.g. `Bar`), not the owner class (`Foo`), because `ClassProperty.get_node` is dead code for plain (non-cached) descriptors
- `mixed_property` has a known behavior where falsy instances (`__bool__` returning `False`) receive the class instead of themselves due to `instance or klass` — external libraries depend on this
- `pin.cls` shares a single class-level cache for each concrete class in inheritance; child classes get their own memoization dict

**Decisions:**
- Did not modify `src/kain/descriptors.py` per user constraint — tests reflect current (sometimes quirky) behavior exactly
- Documented known edge-case behaviors directly in test docstrings and comments for future maintainers

**Open:**
- None

**Modified files:**
- `tests/test_descriptors.py` (new, 32 tests)
- `MEMORY.md` (updated)

### [2026-04-02 19:00]
**Completed:**
- Created extended test suite `tests/test_descriptors_extended.py` with 58 additional tests
- Covered previously missing public API: `cache()` function with all limit variants
- Added comprehensive `with_parent` tests for all property types (`@pin`, `@pin.native`, `@pin.cls`, `@pin.any`, `@pin.pre`, `@pin.post`, `@class_property`, `@mixed_property`)
- Added full coverage for `.by()` / `.expired_by()` custom cache invalidation callbacks across all pin variants
- Added TTL tests for all cached variants (not just `@pin.native`)
- Added exception hierarchy tests (`PropertyError`, `ContextFaultError`, `ReadOnlyError`, `AttributeException`)
- Added extensive edge case tests: `None` caching, falsy values, multiple inheritance, diamond inheritance, descriptor subclassing
- Added async edge case tests: exception handling, future caching behavior
- Added manual set/delete operation tests for all descriptor types
- Total: 90 tests (32 original + 58 new), all passing

**Discovered:**
- `cache()` function works as `lru_cache` wrapper: direct function call wraps immediately, numeric arg sets maxsize, `None` means unlimited
- `.by()` callback signature is `(self, node, timestamp=None)` — returns timestamp on set, bool on get
- `with_parent` raises `NotImplementedError` (not `RecursionError`) when parent descriptor not found via `get_attr`
- `pin.pre` `__set__` returns value without caching for instances (only caches on class access)
- `AbstractProperty.is_data` only available on `BaseProperty` subclasses, not on `pin` (which inherits from `InsteadProperty`)

**Decisions:**
- Kept tests in separate file (`test_descriptors_extended.py`) for clarity; can merge later if needed
- Tests follow existing pytest patterns from `test_descriptors.py`
- Documented callback signatures in test docstrings for future reference

**Open:**
- None

**Modified files:**
- `tests/test_descriptors_extended.py` (new, 58 tests)
- `MEMORY.md` (updated)

### [2026-04-02 19:35]
**Completed:**
- Created comprehensive test suite `tests/test_pin_here.py` with 20 tests for `.here` functionality
- Tests cover `@pin.cls.here`, `@pin.any.here`, `@pin.pre.here`, `@pin.post.here`
- Key insight: `.here` changes **cache storage location**, not the class passed to the function
  - Without `.here`: cache stored on the accessed class (each child has separate cache)
  - With `.here`: cache stored on the defining/parent class (all children share one cache)
- Added tests for cache location verification, shared cache behavior, inheritance chains
- Added tests comparing `.here` vs regular variants side-by-side
- Added edge case tests: TTL with `.here`, diamond inheritance, multiple inheritance, manual set/delete

**Discovered:**
- `pin.cls.here` stores cache in `Foo.__class_memoized__` even when accessed via `Bar` (child)
- `Bar.prop = 100` creates attribute in `Bar.__dict__`, shadowing the descriptor from `Foo`
- `del Foo.prop` removes the descriptor itself from `Foo`, breaking inheritance
- `pin.post.here` creates empty `__class_memoized__` dict (via `get_cache`) even though it doesn't cache on class
- All 20 new tests pass, total test count now 223

**Decisions:**
- Tests document actual `.here` behavior which is cache-location switching, not class-passing switching
- Clarified docstrings to explain the difference between expected (class passing) vs actual (cache storage) behavior

**Open:**
- None

**Modified files:**
- `tests/test_pin_here.py` (new, 20 tests)
- `MEMORY.md` (updated)

### [2026-04-02 19:48]
**Completed:**
- Added `__slots__` to all public property classes in `src/kain/descriptors.py` for memory efficiency
- Classes modified:
  - `AbstractProperty`: `__slots__ = ("function", "__dict__")` — stores the wrapped function; `__dict__` required for `cached_property` decorators (`name`, `header`)
  - `Cached`: `__slots__ = ("is_actual",)` — stores the optional cache expiration callback
  - All other classes (`InsteadProperty`, `BaseProperty`, `InheritedClass`, `ClassProperty`, `MixedProperty`, `ClassCachedProperty`, `MixedCachedProperty`, `PreCachedProperty`, `PostCachedProperty`, `pin`, `class_property`, `mixed_property`, `CustomCallbackMixin`): `__slots__ = ()` — no instance attributes, inherit from parents
- Fixed compatibility issues caused by `__slots__`:
  - `Cached.__init__`: Changed check from `getattr(Is.classOf(self), "is_actual", None)` to `"is_actual" in klass.__dict__` to avoid detecting inherited slot descriptor as method
  - Added `type(is_actual).__name__ != "member_descriptor"` checks in `Cached.call()` and `Cached.__set__()` to handle un-initialized slot values

**Discovered:**
- When using `__slots__`, accessing an unset slot attribute returns `member_descriptor`, not the default value or `AttributeError`
- `getattr(obj, "attr")` on a slotted attribute returns the descriptor object itself if not set, not `None` or default
- `functools.cached_property` requires `__dict__` to store cached values, so `AbstractProperty` needs `"__dict__"` in its slots

**Decisions:**
- Kept `__dict__` in `AbstractProperty.__slots__` to support `cached_property` usage (`name`, `header`, `title` properties)
- Used `type(x).__name__ == "member_descriptor"` as pragmatic check instead of importing `member_descriptor` type (which is C-level internal)

**Open:**
- None

**Modified files:**
- `src/kain/descriptors.py` — added `__slots__` to 15 classes, fixed 3 methods for slot compatibility

### [2026-04-02 20:12]
**Completed:**
- Added full type annotation support for `class_property` and `mixed_property` decorators
- Created `src/kain/py.typed` marker file (PEP 561) to enable type checking for the package
- Added `@overload` declarations to `class_property.__new__` and `mixed_property.__new__`:
  - Both decorators now properly preserve return type `R` through the descriptor protocol
  - Type checkers correctly infer `MyClass.prop -> int` when decorator is applied to `def prop(cls) -> int`
- Added `TYPE_CHECKING`-only `with_parent` attribute to both classes for `@class_property.with_parent` support

**Discovered:**
- Basedpyright/Pyright has excellent support for descriptor type inference — revealed types are correct
- Mypy has known limitations with generic descriptors and `__get__` signatures (false positives on parameter types)
- The key insight: `Callable[..., R]` in `@overload` is more flexible than `Callable[[type[T]], R]` for decorator typing
- `__new__` overloads work better than function-based decorator approach because they preserve the class identity for `with_parent` access

**Decisions:**
- Used `Callable[..., R]` in overloads to avoid strict parameter type checking issues
- Kept class-based implementation (not function wrapper) to preserve `with_parent` class method access
- Accepted mypy's false positives on `__get__` parameters — the revealed types are correct, and pyright works perfectly

**Open:**
- None

**Modified files:**
- `src/kain/descriptors.py` — added `@overload` to `class_property.__new__` and `mixed_property.__new__`, added `TYPE_CHECKING` block with `with_parent`
- `src/kain/py.typed` — new empty marker file

### [2026-04-02 20:35]
**Completed:**
- Added `@overload` declarations to `Cached.__new__` for `@pin.native` and `@pin.ttl()` support
- Added `@overload` declarations to `ClassCachedProperty.__new__`, `MixedCachedProperty.__new__`, `PreCachedProperty.__new__`, `PostCachedProperty.__new__`
- Updated `pin` class with `TYPE_CHECKING` block for type-only declarations of `native`, `cls`, `any`, `pre`, `post` attributes
- Fixed runtime compatibility: `__new__` overloads now accept `is_actual` parameter for TTL support via `partial(cls, is_actual=callback)`
- All 222 tests pass — runtime behavior fully preserved

**Discovered:**
- Type stubs for `pin.native`, `pin.cls`, etc. cannot use simple `ClassVar[type[...]]` because type checkers don't understand that `@pin.native` applied to a function should return `R`
- The `@overload` approach with `__new__(cls, func: Callable[..., R]) -> R` is the correct pattern for decorator typing
- `InheritedClass.make_from()` creates dynamic subclasses at runtime, making static typing of `pin.cls` challenging
- For full type inference of `@pin.native` → `R`, would need descriptor protocol support in type checkers beyond current capabilities

**Decisions:**
- Kept runtime implementation unchanged — only added type annotations and `@overload` stubs
- Used `TYPE_CHECKING` blocks to separate type-only declarations from runtime implementation

**Open:**
- Future: Explore `@pin.native` returning proper generic descriptor type instead of `Unknown`

**Modified files:**
- `src/kain/descriptors.py` — added `@overload` to `Cached.__new__`, `ClassCachedProperty.__new__`, `MixedCachedProperty.__new__`, `PreCachedProperty.__new__`, `PostCachedProperty.__new__`; added `TYPE_CHECKING` block to `pin` class

### [2026-04-02 21:00]
**Completed:**
- Added full type inference support for `@pin.cls.here`, `@pin.any.here`, `@pin.pre.here`, `@pin.post.here` variants
- Created Protocol classes (`_PinClsType`, `_PinAnyType`, `_PinPreType`, `_PinPostType` and their `.here` variants) to enable type checkers to understand decorator usage
- Each Protocol has `__call__` overloads that return `R` when used as decorator and the property class when called without arguments
- Added `ttl` attribute to each Protocol for `@pin.*.here.ttl(n)` support
- Updated `pin` class TYPE_CHECKING block to use Protocol types instead of `type[ClassCachedProperty[T, R]]`
- All 225 tests pass — runtime behavior fully preserved
- Basedpyright now correctly infers types:
  - `@pin.cls.here` → `(cls: TestClass) -> set[str]` (class access) / `() -> set[str]` (instance access)
  - `@pin.any.here` → `(self_or_cls: object) -> tuple[int, ...]` / `() -> tuple[int, ...]`
  - `@pin.pre.here` → `(self_or_cls: object) -> frozenset[int]` / `() -> frozenset[int]`
  - `@pin.post.here` → `(self_or_cls: object) -> bytes` / `() -> bytes`
  - TTL variants work correctly: `@pin.cls.here.ttl(3600)` → `(cls: TestClass) -> bytearray`

**Discovered:**
- `type[ClassCachedProperty[T, R]]` doesn't work for decorator typing because type checkers don't look at `__new__` overloads of the inner type
- Protocol with `__call__` overloads is the correct pattern for classes that act as decorators
- `ClassVar` is required on Protocol-typed attributes in generic classes to avoid "Access to generic instance variables is ambiguous" errors
- `_PinClsHereType` must reference `_PinClsHere` Protocol (not the class) to enable `.here.ttl()` chaining

**Decisions:**
- Used Protocol pattern instead of trying to make `type[...]` work with `__new__` overloads
- Created separate Protocol classes for each variant (`_PinClsType`, `_PinAnyType`, etc.) and their `.here` counterparts
- Kept runtime `InheritedClass.make_from()` implementation unchanged — Protocols are TYPE_CHECKING only
- Accepted that mypy has limitations with generic class variables ("Access to generic class variables is ambiguous") — basedpyright works correctly

**Open:**
- None

**Modified files:**
- `src/kain/descriptors.py` — added Protocol imports, created 8 Protocol classes for `@pin.*` and `@pin.*.here`, updated `pin` class TYPE_CHECKING block

### [2026-04-02 21:00]
**Completed:**
- Added full type annotation support for `@pin.cls.here`, `@pin.any.here`, `@pin.pre.here`, `@pin.post.here`
- Created type stubs `_PinClsDescriptor`, `_PinAnyDescriptor`, `_PinPreDescriptor`, `_PinPostDescriptor` with:
  - `__call__` overloads for decorator usage (returning `R`)
  - `here` attribute for chained access (`@pin.cls.here`)
  - `ttl` attribute for TTL support (`@pin.cls.here.ttl(n)`)
- Created comprehensive type inference test suite `tests/test_type_inference.py` with:
  - All `@pin` variants: `@pin`, `@pin.native`, `@pin.cls`, `@pin.any`, `@pin.pre`, `@pin.post`
  - All `@pin.ttl()` combinations
  - All `@pin.xxx.here` variants including with TTL
  - `@class_property` and `@mixed_property`
  - `cache()` function
  - `with_parent` decorator
  - Runtime assertions to verify functionality matches type annotations
- All 225 tests pass (222 original + 3 new type inference tests)

**Discovered:**
- Type checker can now resolve `@pin.cls.here` as `_PinClsDescriptor[Unknown, Unknown]`
- Type stubs for descriptors with `.here` chaining require descriptor classes with both `__call__` and attribute annotations
- `reveal_type()` in test file shows that `@pin.cls.here` decorated methods are correctly typed as callable with proper return types
- Basedpyright correctly infers types for: `pinned_cls_here` → `(cls: TestClass) -> set[str]`, etc.

**Decisions:**
- Used descriptor pattern with `__call__` for type stubs instead of `__new__` — better represents runtime behavior where `pin.cls` is accessed as class attribute then called as decorator
- Accepted "Unknown" generic parameters as limitation of current type system for dynamic descriptor creation
- Test file serves dual purpose: runtime verification and type checking validation

**Open:**
- None — type inference for all exported descriptors is now implemented

**Modified files:**
- `src/kain/descriptors.py` — added `_PinClsDescriptor`, `_PinAnyDescriptor`, `_PinPreDescriptor`, `_PinPostDescriptor` type stubs with `here` and `ttl` attributes
- `tests/test_type_inference.py` — new comprehensive type inference test suite (3 tests covering all descriptors)

### [2026-04-03 03:29]
**Completed:**
- Added exhaustive module-level, class-level, and method-level docstrings to every file in `src/kain/properties/`
- Documented all 11 exported entities and their relationships (`pin`, `bound_property`, `class_property`, `mixed_property`, and the full cached family)
- Wrote detailed explanations of key mechanisms:
  - `parent_call` and MRO traversal with the `bool → int` index trick
  - `extract_wrapped` as the inverse of descriptor wrapping
  - `CustomCallbackMixin`/`is_actual` TTL semantics
  - Differences between plain and parent-aware (`*.here`) cache storage locations
  - Instance-only vs class-only vs mixed descriptor protocols
- Verified all modified files compile successfully (`py_compile`)

**Discovered:**
- `invocation_context_check` in `src/kain/properties/primitives.py` is dead code — it is defined but never used by any class in the `properties` package (only used in `descriptors.py`)
- Bug in `src/kain/classes.py` `Singleton.__call__`: `cls.instance is not Missing` compares an instance `Nothing` against the *class* `Missing`; because `Nothing is Missing` is always `False`, the condition is always `True` and the singleton recreates its instance on every call
- `parent_call` relies on the subtle Python fact that `bool` subclasses `int`: `index=func.__name__ not in Is.classOf(node).__dict__` yields `0` (immediate parent) when overridden and `1` (grandparent) when inherited — this is intentional but extremely subtle
- `properties/cached/klass.py` fixed a latent bug in `ttl()` compared to `descriptors.py`: it uses `isinstance(value, float)` instead of a truthiness check, so a stored timestamp of `0.0` is no longer mistaken for falsy and replaced with `time()`

**Decisions:**
- Did not modify any runtime logic or fix the discovered bugs — only added documentation per user constraint
- Documented the discovered issues inline in docstrings so future agents/maintainers are aware of them

**Open:**
- Awaiting user decision on whether to fix the `Singleton` bug and the dead `invocation_context_check` code

**Modified files:**
- `src/kain/properties/__init__.py`
- `src/kain/properties/primitives.py`
- `src/kain/properties/class_property.py`
- `src/kain/properties/cached/__init__.py`
- `src/kain/properties/cached/klass.py`
- `src/kain/properties/cached/instance.py`
- `src/kain/properties/cached/mixed.py`
- `src/kain/properties/cached/pre.py`
- `src/kain/properties/cached/post.py`

### [2026-04-03 03:46]
**Completed:**
- Added complete type annotations to `src/kain/signals.py`:
  - `NeedRestart: bool`, `TracebackType | None` for tracebacks, precise `Callable` signatures for hooks and callbacks
  - Introduced `_OnChangeCallable` Protocol to accurately type the return value of `quit_at()`
  - Full annotations on `on_quit.__init__`, `schedule`, `add_hook`, `exceptions_hooks_proxy`, `threading_handler`, `signal_handler`, `get_mtime`, `get_selfpath`, `quit_at`
- Fixed 3 bugs in `signals.py`:
  1. `restore_original_handlers` now resets `SIGQUIT` (was incorrectly resetting `SIGHUP` which was never hooked)
  2. `threading_handler` now safely returns when `args.exc_type is None` (Python allows `None` in `threading.ExceptHookArgs`)
  3. Fixed infinite recursion in `exceptions_hooks_proxy`: `sys.excepthook is not self.exceptions_hooks_proxy` was always `True` because accessing a bound method creates a new object each time. Solved by storing a stable reference in `self._proxy` during `__init__`
- Created comprehensive test suite `tests/test_signals.py` with 23 tests covering:
  - Singleton behavior of `on_quit`
  - `schedule` / `teardown` / callback error handling / idempotency
  - `threading_handler` skips for `SystemExit` and `None`, proxies for other exceptions
  - `exceptions_hooks_proxy` hook invocation and teardown side-effects
  - `signal_handler` calls teardown and raises `SystemExit(1)`
  - `inject_hook` / `restore_original_handlers` correct signal reset (SIGQUIT, not SIGHUP)
  - `quit_at` baseline behavior, `sleep` polling, signal registration, `NeedRestart` trigger, `FileNotFoundError` handling
- User confirmed fix of `Singleton` bug in `src/kain/classes.py` (`cls.instance is Missing` → `cls.instance is Nothing`)
- All 23 tests pass (`pytest`)

**Discovered:**
- `sys.excepthook is not self.exceptions_hooks_proxy` is a classic Python bound-method identity trap: every attribute access on a bound method produces a new wrapper object, so `is` comparison always fails
- `threading.ExceptHookArgs` is a `structseq`; it must be constructed with a single tuple argument, not keyword arguments
- `RuntimeError('boom') == RuntimeError('boom')` can be `False` in CPython because exception equality is identity-based unless `__eq__` is overridden

**Decisions:**
- Kept `signal` as the parameter name in `quit_at` per user request (it shadows the `signal` module, but user explicitly said not to change it)
- Removed the implicit `can_continue = quit_at(signal=1)` module-level export per user direction; `__all__` now only exports `on_quit` and `quit_at`
- Used `self._proxy` reference pattern to fix recursion without changing public API

**Open:**
- None

**Modified files:**
- `src/kain/signals.py` — full type annotations + 3 bug fixes
- `tests/test_signals.py` — new comprehensive test suite (23 tests)
- `MEMORY.md` — updated

### [2026-04-03 04:35]
**Completed:**
- Created extended test suite `tests/test_internals_extended.py` with 185 tests for `kain.internals`
- Focus on typing variants and edge cases: `GenericAlias`, `UnionType`, `typing.Union`, `Optional`, `TypeVar`, `Protocol`, etc.
- Added parametrized tests for `is_subclass`, `is_collection`, `is_mapping`, `is_primitive`, `is_internal`, `object_name`, `pretty_module`, `unique`, `iter_inheritance`, `get_owner`, `get_attr`, `to_ascii`, `to_bytes`, `who_is`, and more
- Added docstrings to all public functions, constants, and dataclasses in `src/kain/internals.py`
- All 298 tests pass (113 original + 185 new)

**Discovered:**
- `is_subclass` handles `types.UnionType` (`int | str`) and `typing.Union[int, str]` correctly via `get_args` check, but does not handle `Literal[1, 2]` (raises `TypeError` from `issubclass`)
- `unique` with `key` de-duplicates by key value, so multiple items sharing the same key are collapsed to the first occurrence
- `pretty_module(os.path.join)` returns `posixpath` on macOS, not `os.path`, because `os.path.join` is defined in `posixpath` module
- `class_of(typing.Union[int, str])` returns `typing._UnionGenericAlias` in Python 3.12+, not `typing._SpecialForm`

**Decisions:**
- Kept new tests in a separate file for clarity
- Added concise PEP-257 style docstrings with occasional multi-line explanations where logic is non-trivial

**Open:**
- None

**Modified files:**
- `tests/test_internals_extended.py` (new, 185 tests)
- `src/kain/internals.py` (docstrings added)
- `MEMORY.md` (updated)

### [2026-04-03 04:55]
**Completed:**
- Created comprehensive unit tests for `kain.importer` module (`tests/test_importer.py`, 71 tests)
- Tests cover all public functions: `required`, `optional`, `add_path`, plus internal helpers
- Added tests for constants (`IGNORED_OBJECT_FIELDS`, `PACKAGES_MAP`)
- Tested import workflows: simple modules, nested attributes, module paths, error handling
- Tested path resolution: dot notation (`.`, `..`, `...`), relative paths, directory walking
- Tested `sys.path` manipulation with `add_path`
- Fixed bug in `add_path`: `logger.Args(**kw)` → `Who.Args(**kw)` (there is no `logger.Args`)
- Fixed function signature for `required()`: removed conflicting keyword-only args that interfered with `**kw` popping
- Added comprehensive docstrings to all functions in `src/kain/importer.py`:
  - Module-level docstring with usage examples
  - All constants documented with `#:` comments
  - Full docstrings for all functions (Args, Returns, Raises, Examples)

**Discovered:**
- `ModuleType("name")` creates a module with `__name__ = 'name'`, but iterating over `IGNORED_OBJECT_FIELDS` and setting them to `None` breaks `__import__` which requires `__name__` to be a string
- macOS `/tmp` resolves to `/private/tmp` via `Path.resolve()` — tests must use resolved paths for assertions
- `ismodule` is imported directly from `inspect`, so patching must be done at `kain.importer.ismodule` (where it's imported to), not at `inspect.ismodule`
- `get_path` with dot patterns (`.`, `..`, `...`) has specific behavior: single dot returns as-is, multiple dots navigate parent directories

**Decisions:**
- Used `ModuleType` with `__name__` preserved for mock modules in tests
- Tests for `get_path` use flexible assertions that work across macOS/Unix path resolution differences
- All 71 tests pass

**Modified files:**
- `tests/test_importer.py` (new, 71 tests)
- `src/kain/importer.py` (bug fixes + docstrings added)
- `MEMORY.md` (updated)

### [2026-04-03 05:05]
**Completed:**
- Added comprehensive type annotations to `src/kain/importer.py`
- Type annotations added:
  - Constants: `IGNORED_OBJECT_FIELDS: set[str]`, `PACKAGES_MAP: dict[str, str]`
  - `get_module() -> tuple[ModuleType, tuple[str, ...]]`
  - `get_child(path: str, parent: object, child: str) -> object`
  - `import_object()` with `@overload` variants for better inference
  - `cached_import(*args: object, **kw: object) -> object`
  - `required(path: str, *args: object, **kw: object) -> object`
  - `optional(path: str, *args: object, **kw: object) -> object`
  - `sort: Callable[..., list[object]]` (properly typed as callable)
  - `get_path(path: str | Path, root: str | Path | None = None) -> Path`
  - `add_path(path: str | Path, **kw: object) -> Path`
- Used `ModuleType` instead of generic `object` for module return types
- Used `Callable` from `collections.abc` for `sort` variable
- Fixed bug: restored missing `@cache` decorator on `get_module()`

**Discovered:**
- When adding type annotations, accidentally removed `@cache` decorator from `get_module` causing test failure
- `sort` variable needs explicit `Callable[..., list[object]]` type annotation since `optional()` returns `object`
- Using `@overload` for `import_object` allows type checkers to better infer return types based on argument patterns

**Decisions:**
- Kept `required()` and `optional()` using `**kw` pattern with `object` return type rather than strict generics, since actual return type depends on runtime import
- Used `object` as return type for import functions since the imported object could be anything (module, class, function, etc.)
- All 71 tests pass

**Modified files:**
- `src/kain/importer.py` (type annotations added)
- `MEMORY.md` (updated)

### [2026-04-03 05:30]
**Completed:**
- Wrote comprehensive unit tests for `kain.classes` (`Missing`, `Nothing`, `Singleton`) and `kain.monkey` (`Monkey.expect`, `Monkey.patch`, `Monkey.bind`, `Monkey.wrap`)
- Total 26 tests; all pass (`pytest`)
- Added module-level, class-level, and method-level docstrings to `classes.py` and `monkey.py`
- Added/updated type annotations in both modules; `basedmypy` reports 0 errors, `basedpyright` reports 0 errors

**Discovered:**
- `Monkey.patch` with a tuple `(node, name)` calls `required(node, name)`, which works only when `node` is a module or object with `hasattr(node, name)` — not for plain dicts
- `Monkey.wrap(original)` without an explicit `name` almost always fails because `Who.Name(node) != func.__name__` and `required(node, "wrapper")` raises `ImportError`

**Decisions:**
- Used `types.SimpleNamespace` and mock modules in tests to avoid mutating real global state
- Used `object` instead of `Any` where possible to reduce pyright/mypy noise, while keeping `Any` for the `expect` return type because `classmethod(...)` cannot be expressed as a plain `Callable`

**Open:**
- None

**Modified files:**
- `src/kain/classes.py`
- `src/kain/monkey.py`
- `tests/test_classes.py`
- `tests/test_monkey.py`

### [2026-04-03 05:45]
**Completed:**
- Fixed broken `make test` (598 tests now passing)
  - Added missing exports to `kain.properties.__init__.py`: `PropertyError`, `ContextFaultError`, `ReadOnlyError`, `AttributeException`, `BaseProperty`, `pin`
  - Added `is_data` cached_property to `BaseProperty` in `primitives.py` to support the descriptor protocol introspection tests
- Added comprehensive docstrings and inline comments to `src/kain/signals.py`:
  - Module-level docstring with example usage
  - Class docstrings for `on_quit` singleton
  - Method-level docstrings for all public and internal methods
  - Inline comments explaining the `NeedRestart` flag and closure structure in `quit_at`

**Discovered:**
- `tests/test_properties.py` was importing exception classes that weren't exported from `kain.properties`
- `is_data` attribute was missing from `BaseProperty` — it's used by tests to verify whether a descriptor is a data descriptor (defines `__set__` or `__delete__`)
- Coverage warnings about "Module app was never imported" are benign — the coverage config references a module that doesn't exist in this project

**Decisions:**
- Implemented `is_data` as a `cached_property` that introspects the class for `__set__` or `__delete__` rather than hardcoding it per subclass
- Added `pin` to `__all__` explicitly since it's referenced in tests

**Open:**
- None

**Modified files:**
- `src/kain/properties/__init__.py`
- `src/kain/properties/primitives.py`
- `src/kain/signals.py`

### [2026-04-03 06:15]
**Completed:**
- Analyzed `proxy_to` decorator in `src/kain/properties/proxy_to.py` and its dependencies (`Missing`, `Nothing`, `Is`, `Who`, `bound_property`)
- Created comprehensive test suite `tests/test_proxy_to.py` with 25 tests covering:
  - Basic string-pivot proxying via `bound_property` (caching, method proxying)
  - All bind modes: default (`bound_property`), `None` (direct copy), custom callable
  - Non-string (object) pivot mode
  - `safe=True/False` collision detection and private-attribute bypass
  - `default` parameter: raising vs fallback on missing pivot or missing attribute
  - `pre` parameter wrapping results in `partial`
  - Custom `getter` parameter (e.g., `itemgetter` for dict pivots)
  - Input validation: non-class decorator target, empty mapping list, missing pivot attribute
  - Warning log capture when fallback default is used
- Fixed bug in `proxy_to.py`: `Nothing` sentinel was redefined locally, breaking identity checks when callers passed `Nothing` from `kain.classes`. Now imports the canonical `Nothing` from `kain.classes`.
- All 25 tests pass (`pytest`)

**Discovered:**
- `proxy_to` has three bind modes: `bound_property` (default when last arg is string), direct copy (`None`), or custom descriptor factory
- When `bind=None`, the decorator copies the attribute reference directly from the class-level pivot descriptor; this means class-level access returns the bound method from the pivot instance
- `safe=True` only blocks public attributes (those not starting with `_`); private attributes bypass the collision check entirely
- `default` uses identity check against `Nothing`; any other `Missing()` instance would also be treated as a fallback value rather than "no default"
- `pre` wraps the final result in `functools.partial(pre, result)`, meaning the proxied attribute returns a callable that must be invoked to get the post-processed value

**Decisions:**
- Fixed the `Nothing` identity bug because it caused tests to fail when using the public `Nothing` export
- Kept all other runtime logic unchanged

**Open:**
- None

**Modified files:**
- `src/kain/properties/proxy_to.py` (bug fix: import `Nothing` from `kain.classes`)
- `tests/test_proxy_to.py` (new, 25 tests)
- `MEMORY.md` (updated)

### [2026-04-03 06:25]
**Completed:**
- Added comprehensive module-level, function-level, and inline comments to `src/kain/properties/proxy_to.py`
- Documented `proxy_to` signature, parameters, return value, exceptions, and usage examples in a detailed docstring
- Added inline comments explaining:
  - Bind strategy detection from the last positional argument
  - `__proxy_fields__` maintenance
  - Validation of the mapping list
  - Collision guard (`safe` mode) behaviour
  - Difference between string pivot (instance attribute) and object pivot (direct source)
  - Default/fallback logic and warning log emission
  - Direct copy mode vs descriptor binding mode
- Fixed internal spelling `class_wraper` → `class_wrapper` during the rewrite
- All 25 tests in `test_proxy_to.py` still pass

**Discovered:**
- No new runtime issues; the code behaviour matches the documented contract after the `Nothing` fix

**Decisions:**
- Kept all runtime logic identical except the benign rename of the internal closure
- Used Google-style docstring sections (Parameters, Returns, Raises, Examples) to match the rest of the codebase

**Open:**
- None

**Modified files:**
- `src/kain/properties/proxy_to.py` (docstrings + comments + minor rename)
- `MEMORY.md` (updated)

### [2026-04-03 06:40]
**Completed:**
- Added full type annotations to `src/kain/properties/proxy_to.py`:
  - Function signature typed with `*mapping: object`, `getter: Callable[[str], Callable[[Any], Any]]`, `default: Any`, `pre: Callable[[Any], Any] | None`, `safe: bool`
  - Return type `Callable[[type[Any]], type[Any]]`
  - Internal variables (`bind`, `fields`, `pivot`, `mapping_list`, `value`, `wrap`) fully annotated
  - Inner `wrapper` function annotated with `(name: str, node: Any) -> Any`
- Moved typing-only imports (`Callable`, `Any`) into `TYPE_CHECKING` blocks to satisfy `TC003`
- Added `# ruff: noqa: ANN401` because `proxy_to` is inherently dynamic and `Any` is the appropriate type for `default`, `node`, and return values
- Removed unused `Missing` import from `proxy_to.py` and `test_proxy_to.py`
- Fixed all ruff warnings in `tests/test_proxy_to.py` (line length, unused imports, missing annotations, SLF001, B018, ARG005)
- All 25 tests pass; ruff reports zero errors for both files

**Discovered:**
- `ruff` rule `TC003` requires stdlib typing imports to be in `TYPE_CHECKING` when `from __future__ import annotations` is present
- `ANN401` forbids `Any` in signatures, but for a generic proxy decorator it is unavoidable; module-level noqa is the pragmatic solution
- `B018` flags bare attribute access inside `pytest.raises` as a "useless expression"; assigning to `_` silences it without changing semantics

**Decisions:**
- Used `TYPE_CHECKING` blocks for `Callable` and `Any` in both the source file and the test file
- Kept `Any` for dynamic parameters rather than inventing overly complex generic overloads that wouldn't improve real-world UX

**Open:**
- None

**Modified files:**
- `src/kain/properties/proxy_to.py` (type annotations added)
- `tests/test_proxy_to.py` (style/typing fixes)
- `MEMORY.md` (updated)

### [2026-04-03]
**Completed:**
- Repository initialized
- .agent/intro.md + AGENTS.md symlink and .agent/history.md created
- .agent/ directory structure set up
- Initial knowledge.md populated with codebase analysis:
  - dependency_graph: mapped components and relationships
  - class_hierarchy: documented inheritance patterns
  - gotchas: noted initial environment quirks
  - insights: recorded initial architectural observations
- Migrated historical work sessions and knowledge from `agent.old/MEMORY.md` into `.agent/history.md` and `.agent/knowledge.md`

**Discovered:**
- Project is a pure-Python utility library with zero runtime dependencies
- Two parallel descriptor implementations exist: legacy `descriptors.py` and modern `properties/` package
- `Makefile` test target references `--cov app` but source lives in `src/kain/`
- Tests directory is populated (11 test files) contrary to old agent.old/AGENTS.md claim
- Development dependencies managed via `uv` with `uv.lock` present

**Decisions:**
- (none yet)

**Open:**
- (none yet)

**Modified files:**
- `.agent/intro.md` + `AGENTS.md` symlink — created project context
- `.agent/history.md` — created work history file, migrated old sessions
- `.agent/tasks.md` — created task tracking
- `.agent/decisions.md` — created decision log
- `.agent/knowledge.md` — populated with initial codebase knowledge, added historical gotchas and insights

### [2026-04-03T13:32:59+03:00]
**Completed:**
- Added comprehensive type annotations to `src/kain/properties/primitives.py` for IDE inference:
  - `Generic[T, R]` + `@overload` on `bound_property.__new__` and `__get__`
  - `@overload` variants for `cache()` decorator
  - Full typing for `parent_call`, `extract_wrapped`, `invocation_context_check`, and `BaseProperty`
- Created `tests/test_primitives.py` with 69 granular unit tests covering exceptions, `cache()`, `extract_wrapped()`, `parent_call()`, `BaseProperty`, `bound_property`, and `invocation_context_check`
- Fixed `cache()` to cast float limits to `int` for `functools.lru_cache` compatibility

**Discovered:**
- `parent_call` captures `func.__name__` at definition time, so wrapper `__name__` reassignments after `parent_call(...)` do not affect MRO lookup
- `bound_property.__get__` does not accept extra `*args/**kw`, so `parent_call` forwarding to such parents must use a `BaseProperty` subclass with a `call` method instead
- `functools.lru_cache` in Python 3.12 rejects `float` for `maxsize`, requiring an `int` cast

**Decisions:**
- Keep runtime behavior identical; annotations are PEP 484 compliant and use `from __future__ import annotations`

**Open:**
- None

**Modified files:**
- `src/kain/properties/primitives.py`
- `tests/test_primitives.py`

### [2026-04-03T14:10:00+03:00]
**Completed:**
- Added comprehensive type annotations to `src/kain/properties/class_property.py` for IDE inference:
  - `Generic[T, R]` + `@overload` on `__new__` and `__get__` for both `class_property` and `mixed_property`
  - Return type `R | Awaitable[R]` on `call()` to reflect coroutine wrapping
  - Added `from __future__ import annotations`
- Created `tests/test_class_property.py` with 36 granular unit tests covering:
  - `class_property`: instance/class access, inheritance, no caching, `get_node`, `call`, async wrapping, `with_parent`, mixin owner resolution, `is_data`
  - `mixed_property`: instance/class access, falsy-instance bug, inheritance, `get_node`, `call`, async wrapping, `with_parent`, mixin behavior, `is_data`

**Discovered:**
- `class_property` and `mixed_property` are non-data descriptors (`is_data is False`), so they do not override instance `__dict__` entries
- `mixed_property` intentionally uses `instance or klass`, meaning falsy instances receive the class — this is relied upon by consumers
- `ensure_future` triggers a DeprecationWarning in Python 3.12 when there is no active event loop

**Decisions:**
- Preserved the `instance or klass` behavior in `mixed_property.__get__` exactly as-is

**Open:**
- None

**Modified files:**
- `src/kain/properties/class_property.py`
- `tests/test_class_property.py`

### [2026-04-07T18:15:00Z]
**Completed:**
- Added comprehensive type annotations and granular unit tests for the entire `src/kain/properties/cached/` scope
- Typed `klass.py`: `CustomCallbackMixin`, `class_parent_cached_property`, `class_cached_property` with `Generic[T, R]` and `__new__` overloads (including `is_actual` for TTL/custom callbacks)
- Typed `instance.py`: `cached_property` with `__new__` overloads and `is_actual` support
- Typed `mixed.py`: `mixed_parent_cached_property`, `mixed_cached_property` with `Generic[T, R]` and `__new__` overloads
- Updated `pre.py` and `post.py` to use the new generic bases
- Created `tests/test_cached.py` with 62 granular unit tests covering:
  - `CustomCallbackMixin` (`by`, `expired_by`, `ttl`, invalid inputs)
  - `class_parent_cached_property` / `class_cached_property` (owner caching, `.here`, subclass separation, `with_parent`, async, TTL)
  - `cached_property` (instance caching, class-access rejection, `with_parent`, async future caching)
  - `mixed_parent_cached_property` / `mixed_cached_property` (dual cache, `.here`, `with_parent`)
  - `pre_parent_cached_property` / `pre_cached_property` (skips instance caching)
  - `post_parent_cached_property` / `post_cached_property` (skips class caching)
- Fixed `TypeError` from `__new__` overloads missing `is_actual` keyword argument (used by `CustomCallbackMixin.by()` and `.ttl()`)
- Full suite now at **788 passing** tests

**Discovered:**
- `__new__` overloads on cached descriptors must explicitly accept `is_actual: Any = ...` because `CustomCallbackMixin.by()` and `.ttl()` return `partial(cls, is_actual=callback)`; otherwise decoration raises `TypeError`
- `class_cached_property.expired_by` and `.by` compare as bound methods, so identity check fails even though they wrap the same underlying function — must compare `__func__` instead
- `pre_parent_cached_property.__set__` skips instances entirely, so `__instance_memoized__` may not exist after a manual set

**Decisions:**
- Kept runtime behavior identical; all changes are type annotations and tests
- Used `Callable[[type[T]], R]` and `Callable[[T], R]` overloads where appropriate for class vs instance access typing

**Open:**
- None

**Modified files:**
- `src/kain/properties/cached/klass.py`
- `src/kain/properties/cached/instance.py`
- `src/kain/properties/cached/mixed.py`
- `src/kain/properties/cached/pre.py`
- `src/kain/properties/cached/post.py`
- `tests/test_cached.py` (new, 62 tests)

---

## Archive
<!-- Old sessions summarized here when history.md exceeds 200 lines -->


### [2026-04-07T18:57:00Z]
**Completed:**
- Removed legacy `src/kain/descriptors.py`; cleaned `src/kain/__init__.py` exports; deleted `tests/test_descriptors.py` and `tests/test_descriptors_extended.py`
- Repaired bulk-edit syntax errors in `tests/test_internals.py` and `tests/test_internals_extended.py` (missing `def` lines after `@pytest.mark.parametrize`, unmatched parentheses, bad docstring replacements)
- Fixed broken extended tests across `test_classes.py`, `test_internals.py`, `test_internals_extended.py`, `test_importer.py` — corrected operator-chaining assertions (`is` with `==`), wrong exception expectations, and mocked-module tests that bypassed `sys.modules`
- Patched two implementation bugs in `src/kain/internals.py`:
  - `is_subclass` now catches `TypeError` from `issubclass()` and returns `False` gracefully for non-class types (TypeVar, non-runtime Protocols, NewType)
  - `get_mro` now maps types through `str` when `glue` is provided but `func` is not, preventing `TypeError: sequence item 0: expected str instance, type found`
- Added new test files: `tests/test_kain_init.py` (public API verification), `tests/test_pin_here.py` (90 tests for `.here` aliases on cached descriptors)
- Full suite now at **1464 passing** tests

**Discovered:**
- `class_cached_property.here` caches on the *owner* parent class (via `get_owner`), so subclasses share one cache — this differs from plain `class_cached_property`
- `Missing.__eq__` always returns `False`, so `m == m is False` parses as chained comparison and fails; must use `(m == m) is False`
- `get_attr(A, "__init__", default="nope")` returns `"nope"` because `_get_attribute_from_inheritance` only searches `__dict__`, not inherited attributes
- `is_mapping(dict)` is `True` because `issubclass(dict, dict)` is true
- `who_is(weakref.ref)` returns `weakref.ReferenceType`, not `weakref.ref`

**Decisions:**
- Kept graceful-failure behavior in `is_subclass` rather than raising on exotic typing constructs

**Open:**
- None

**Modified files:**
- `src/kain/__init__.py`
- `src/kain/internals.py`
- `tests/test_classes.py`
- `tests/test_internals.py`
- `tests/test_internals_extended.py`
- `tests/test_importer.py`
- `tests/test_kain_init.py` (new)
- `tests/test_pin_here.py` (new)


### [2026-04-07T19:10:00Z]
**Completed:**
- Conducted a full docstring review across the entire `src/kain/` codebase and applied fixes through parallel subagents
- Added missing module-level docstrings to `src/kain/__init__.py` and `src/kain/internals.py`
- Added missing method docstrings to `classes.py` (Missing/Singleton dunders), `primitives.py`, `class_property.py`, and all `properties/cached/*.py` files
- Completed incomplete docstrings with `Args`/`Returns`/`Raises` sections in `internals.py`, `importer.py`, `monkey.py`, `signals.py`, `primitives.py`, and cached descriptors
- Converted `quit_at` and nested `sleep` docstrings in `signals.py` from NumPy-style to Google-style for consistency
- Fixed multiple typos and grammar issues:
  - `is_from_primivite` → `is_from_primitive` (function and `Is.Primivite` → `Is.Primitive`)
  - `"isn't exists"` → `"does not exist"` (importer.py and proxy_to.py error messages)
  - `"all arguments is None"` → `"all arguments are None"`
  - `"expire must be positive number"` → `"expire must be a positive number"`
  - `"looks like as non-instance invokation"` → `"looks like a non-instance invocation"`
  - `"instance just-replace-descriptor"` → `"instance-bound descriptor"`
- Translated Russian comments in `unique` overloads to English
- Fixed `proxy_to.py` error messages and corrected the misleading `pre` parameter description
- Updated tests to match new error messages and renamed symbols (`test_internals.py`, `test_importer.py`, `test_proxy_to.py`, `test_cached.py`, `test_properties.py`, `test_primitives.py`, `test_kain_init.py`, `test_properties_init.py`)
- Full suite remains at **1464 passing** tests

**Discovered:**
- `kain.__all__` actually exports 18 names (not 14), including `cache`, `class_property`, `mixed_property`, `pin` from `kain.properties`
- `kain.properties.__all__` exports 20 names (not 19), including `cache`

**Decisions:**
- Standardized on Google-style docstring sections (`Args`, `Returns`, `Raises`, `Example`) across the package, replacing NumPy-style sections where they appeared

**Open:**
- None

**Modified files:**
- `src/kain/__init__.py`
- `src/kain/classes.py`
- `src/kain/internals.py`
- `src/kain/importer.py`
- `src/kain/monkey.py`
- `src/kain/signals.py`
- `src/kain/properties/primitives.py`
- `src/kain/properties/class_property.py`
- `src/kain/properties/proxy_to.py`
- `src/kain/properties/cached/instance.py`
- `src/kain/properties/cached/klass.py`
- `src/kain/properties/cached/mixed.py`
- `src/kain/properties/cached/pre.py`
- `src/kain/properties/cached/post.py`
- `tests/test_cached.py`
- `tests/test_importer.py`
- `tests/test_internals.py`
- `tests/test_primitives.py`
- `tests/test_properties.py`
- `tests/test_proxy_to.py`
- `tests/test_kain_init.py`
- `tests/test_properties_init.py`


### [2026-04-07T19:16:00Z]
**Completed:**
- Standardized **all** docstrings in `src/kain/` to Google style (`Args:`, `Returns:`, `Raises:`, `Example:`)
- Converted remaining NumPy-style docstrings in:
  - `src/kain/properties/primitives.py` (`cache`, `extract_wrapped`, `parent_call`, `invocation_context_check`, `BaseProperty.with_parent`, `bound_property.__get__`)
  - `src/kain/properties/proxy_to.py` (`proxy_to` function)
  - `src/kain/properties/cached/klass.py` (`CustomCallbackMixin.ttl`, `class_parent_cached_property.__init__`)
- Fixed the actual runtime string in `bound_property.title` from `"instance just-replace descriptor"` to `"instance-bound descriptor"`
- Verified no NumPy-style headers remain anywhere in `src/kain/`
- Full suite stays at **1464 passing** tests

**Discovered:**
- `bound_property.title` return string still contained `"just-replace"` in the implementation even though the docstring/test had been updated earlier

**Decisions:**
- Adopted Google-style docstrings project-wide for consistency

**Open:**
- None

**Modified files:**
- `src/kain/properties/primitives.py`
- `src/kain/properties/proxy_to.py`
- `src/kain/properties/cached/klass.py`
