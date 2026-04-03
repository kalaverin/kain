---
# Machine Index
version: 1
schema: knowledge-graph
last_updated: 2026-04-07T13:18:30Z
graph_hash: "c6f43a34"
---

<!-- Protocol: ~/.config/kimi/skills/knowledge-protocol/SKILL.md (commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7, modified: 2026-04-07T12:03:05Z) -->
<!-- The following section is a FULL COPY of ~/.config/kimi/skills/knowledge-protocol/SKILL.md
     Protocol commit: 8407a3fffae7e8a6a45e80fb73eeded8078dafa7
     Protocol modified: 2026-04-07T12:03:05Z -->
---
name: knowledge-protocol
description: Protocol for maintaining codebase knowledge — via MCP MemPalace (preferred) OR .agent/knowledge.md (fallback)
---

# Knowledge Protocol

Accumulated codebase knowledge for AI agents. **Primary storage:** MCP MemPalace server. **Fallback:** `.agent/knowledge.md` when MCP unavailable.

## File Structure

```
repo-root/
├── mempalace.yaml          # Optional: wing/room taxonomy
└── .agent/
    └── knowledge.md        # Fallback when MCP unavailable
```

## Storage Selection Priority

| Priority | Storage | Condition |
|----------|---------|-----------|
| 1 | **MCP MemPalace** | MCP `mempalace` server available |
| 2 | **CLI MemPalace** | MCP unavailable BUT `mempalace.yaml` exists → use `mempalace` CLI |
| 3 | **Local file** | No MCP and no `mempalace.yaml` → use `.agent/knowledge.md` |

**Never split knowledge** — use exactly one storage system per session.

## Read on Session Start

### Step 1: Attempt MCP Mode
1. **Call `mempalace_status`**:
   - **Success**: MCP mode activated
   - **Failure**: Go to Step 2 (File Mode)
2. If `.agent/knowledge.md` has content:
   - **Migrate** to MCP: for each YAML block, call `mempalace_add_drawer`
   - Truncate file to stub after successful migration
3. Acknowledge: "Knowledge: MCP active, migrated X entries"

### Step 2: CLI Mode (MCP unavailable, mempalace.yaml exists)
1. Check if `mempalace.yaml` exists in repo root:
   - **Yes**: CLI mode activated
   - **No**: Go to Step 3 (File Mode)
2. **On session start**: Shell `mempalace wake-up [--wing <wing>]` → inject context
3. **For search**: Shell `mempalace search "query"`
4. Acknowledge: "Knowledge: CLI mode (mempalace.yaml), wing: X"

### Step 3: File Mode (MCP unavailable, no mempalace.yaml)
1. Read `.agent/knowledge.md` entirely
2. Parse YAML blocks — restore mental model
3. Acknowledge: "Knowledge: file mode, loaded X components, Y gotchas"

## Write Triggers (MANDATORY)

### MCP Mode (when `mempalace_status` succeeded)

| Trigger | Command |
|---------|---------|
| Component discovered | `mempalace_kg_add` + `mempalace_add_drawer(room="components")` |
| Gotcha found | `mempalace_add_drawer(room="gotchas")` |
| Insight discovered | `mempalace_add_drawer(room="insights")` |
| Investigation concluded | `mempalace_add_drawer(room="investigations")` |
| Data flow traced | `mempalace_add_drawer(room="architecture")` |

### CLI Mode (mempalace.yaml exists, MCP unavailable)
Read-only mode via CLI:
- **Search**: `mempalace search "query" [--wing WING] [--room ROOM]`
- **Wake-up**: `mempalace wake-up [--wing WING]` (on session start)

Writes go to `.agent/knowledge.md` (file mode) until MCP available.

### File Mode (no mempalace.yaml, MCP unavailable)
Use traditional YAML block updates in `.agent/knowledge.md`.

## Migration Procedure

**From file to MCP (one-time):**

```yaml
migration_steps:
  1_verify: "Call mempalace_status to confirm MCP availability"
  2_cli_check: "If MCP unavailable, check for mempalace.yaml → use CLI mode"
  3_read: "Read .agent/knowledge.md entirely"
  4_migrate:
    - For each YAML block → mempalace_add_drawer(wing=..., room=..., content=...)
    - For relationships → mempalace_kg_add(subject=..., predicate=..., object=...)
  5_cleanup: "Truncate knowledge.md to stub"
  6_verify: "Call mempalace_search to confirm migration"
```

## File Mode Schema (YAML)

**Only when MCP unavailable:**

```yaml
dependency_graph:
  components:
    component_name:
      type: [service|repository|router|model|utility]
      path: file_path
      depends_on: [list_of_components]
      provides: [list_of_capabilities]
      critical: true|false

class_hierarchy:
  classes:
    ClassName:
      extends: ParentClass
      mixins: [Mixin1, Mixin2]
      implements: [Interface1]
      abstract: true|false
      file: path/to/file.py

gotchas:
  - id: unique_id
    severity: [critical|high|medium|low]
    category: [behavior|performance|security|environment]
    title: "Short description"
    description: "Detailed explanation"
    location: "Where it occurs"
    discovered: "YYYY-MM-DDTHH:mm:ssZ"

insights:
  - id: insight_001
    type: [pattern|connection|performance|security|design]
    title: "Brief description"
    description: "Detailed insight"
    confidence: [confirmed|likely|speculative]
    discovered: "YYYY-MM-DDTHH:mm:ssZ"

investigation_log:
  - id: INV-001
    topic: "What was investigated"
    status: [ongoing|resolved|stalled]
    conclusion: "What was learned"
    discovered: "YYYY-MM-DDTHH:mm:ssZ"
    resolved: "YYYY-MM-DDTHH:mm:ssZ"
```

## Save Work / Persist State

### MCP Mode
1. All knowledge already persisted via MCP calls
2. Ensure `mempalace_diary_write` called if session ending
3. No local file operations

### CLI Mode (mempalace.yaml exists, MCP unavailable)
1. No persistent writes to palace via CLI (read-only)
2. Accumulate new knowledge to `.agent/knowledge.md` (file mode)
3. Flush pending knowledge to YAML blocks

### File Mode (no mempalace.yaml)
1. Flush pending knowledge to `.agent/knowledge.md` YAML blocks
2. Verify file written

## Cross-References

- **mempalace-protocol**: MCP server commands, AAAK format
- **memory-protocol**: Session history (always local file)
- **decision-protocol**: Architectural decisions
- **task-protocol**: Work tracking

## Implementation Notes

1. **Check MCP availability** each session via `mempalace_status`
2. **Graceful fallback** — MCP failure → automatically switch to file mode
3. **No caching** — check MCP availability each session
<!-- END OF PROTOCOL COPY -->

# Dependency Graph
```yaml
dependency_graph:
  components:
    kain.classes:
      exports: [Missing, Nothing, Singleton]
      dependents: [kain.descriptors, kain.internals, kain.properties.primitives, kain.properties.cached.klass, kain.properties.proxy_to, kain.signals]
    kain.internals:
      exports: [Is, Who, get_attr, get_owner, iter_inheritance, iter_stack, to_ascii, to_bytes, unique]
      dependents: [kain.classes, kain.descriptors, kain.importer, kain.monkey, kain.properties.primitives, kain.properties.class_property, kain.properties.proxy_to, kain.properties.cached.*, kain.signals]
    kain.importer:
      exports: [add_path, optional, required, sort]
      dependents: [kain.monkey]
    kain.descriptors:
      exports: [cache, class_property, mixed_property, pin]
      dependents: [kain.__init__]
      note: Legacy module; superseded by kain.properties package
    kain.monkey:
      exports: [Monkey]
      dependents: [kain.__init__]
    kain.signals:
      exports: [on_quit, quit_at]
      dependents: [kain.__init__]
    kain.properties.primitives:
      exports: [BaseProperty, bound_property, PropertyError, ContextFaultError, ReadOnlyError, AttributeException, cache, extract_wrapped, parent_call]
      dependents: [kain.properties.class_property, kain.properties.proxy_to, kain.properties.cached.*, kain.properties.__init__]
    kain.properties.class_property:
      exports: [class_property, mixed_property]
      dependents: [kain.properties.__init__]
    kain.properties.proxy_to:
      exports: [proxy_to]
      dependents: [kain.properties.__init__]
    kain.properties.cached.klass:
      exports: [class_cached_property, class_parent_cached_property, CustomCallbackMixin]
      dependents: [kain.properties.cached.instance, kain.properties.cached.mixed, kain.properties.cached.pre, kain.properties.cached.post, kain.properties.__init__]
    kain.properties.cached.instance:
      exports: [cached_property]
      dependents: [kain.properties.__init__]
    kain.properties.cached.mixed:
      exports: [mixed_cached_property, mixed_parent_cached_property]
      dependents: [kain.properties.cached.pre, kain.properties.cached.post, kain.properties.__init__]
    kain.properties.cached.pre:
      exports: [pre_cached_property, pre_parent_cached_property]
      dependents: [kain.properties.__init__]
    kain.properties.cached.post:
      exports: [post_cached_property, post_parent_cached_property]
      dependents: [kain.properties.__init__]
    kain.properties.__init__:
      exports: [pin, cached_property, class_cached_property, mixed_cached_property, pre_cached_property, post_cached_property, class_property, mixed_property, proxy_to, bound_property, error classes]
      dependents: [kain.__init__]
  edges:
    - [kain.properties.primitives, kain.properties.class_property, "BaseProperty subclassing"]
    - [kain.properties.primitives, kain.properties.cached.klass, "BaseProperty subclassing"]
    - [kain.properties.cached.klass, kain.properties.cached.instance, "class_parent_cached_property subclassing"]
    - [kain.properties.cached.klass, kain.properties.cached.mixed, "class_cached_property subclassing"]
    - [kain.properties.cached.mixed, kain.properties.cached.pre, "mixed_parent_cached_property subclassing"]
    - [kain.properties.cached.mixed, kain.properties.cached.post, "mixed_parent_cached_property subclassing"]
    - [kain.internals, kain.properties.proxy_to, "get_attr, get_owner, Is, Who"]
    - [kain.classes, kain.signals, "Singleton metaclass"]
```

# Class Hierarchy
```yaml
class_hierarchy:
  classes:
    kain.classes.Missing:
      bases: [object]
      note: Sentinel object; always falsy, never equal to anything
    kain.classes.Singleton:
      bases: [type]
      note: Metaclass caching first constructor call
    kain.internals.Is:
      bases: [dataclass]
      note: Namespace of type-check predicates
    kain.internals.Who:
      bases: [dataclass]
      note: Namespace of introspection/formatting helpers
    kain.monkey.Monkey:
      bases: [object]
      note: Namespace class for monkey-patching helpers
    kain.signals.on_quit:
      bases: [object]
      metaclass: Singleton
      note: Graceful shutdown orchestrator
    kain.properties.primitives.BaseProperty:
      bases: [object]
      note: Abstract base for all kain descriptors; provides with_parent, name, title, header
    kain.properties.primitives.bound_property:
      bases: [BaseProperty]
      note: Instance-only non-caching descriptor
    kain.properties.class_property.class_property:
      bases: [BaseProperty]
      note: Class-only descriptor; get_owner resolves defining class
    kain.properties.class_property.mixed_property:
      bases: [BaseProperty]
      note: Mixed instance/class descriptor
    kain.properties.cached.klass.class_cached_property:
      bases: [BaseProperty]
      note: Class-level cached descriptor; stores in __class_memoized__
    kain.properties.cached.klass.class_parent_cached_property:
      bases: [class_cached_property]
      note: Caches on owning class via get_owner
    kain.properties.cached.instance.cached_property:
      bases: [class_parent_cached_property]
      note: Instance-level cached descriptor; stores in __instance_memoized__
    kain.properties.cached.mixed.mixed_cached_property:
      bases: [class_cached_property]
      note: Mixed-level cached descriptor
    kain.properties.cached.mixed.mixed_parent_cached_property:
      bases: [mixed_cached_property, class_parent_cached_property]
      note: Mixed cached with parent-aware resolution
    kain.properties.cached.pre.pre_parent_cached_property:
      bases: [mixed_parent_cached_property]
      note: Only caches on class access
    kain.properties.cached.pre.pre_cached_property:
      bases: [mixed_cached_property]
      note: Only caches on class access (non-parent)
    kain.properties.cached.post.post_parent_cached_property:
      bases: [mixed_parent_cached_property]
      note: Only caches on instance access
    kain.properties.cached.post.post_cached_property:
      bases: [mixed_cached_property]
      note: Only caches on instance access (non-parent)
    kain.properties.__init__.pin:
      bases: [bound_property]
      note: User-facing namespace decorator exposing native, cls, any, pre, post variants
  interfaces: {}
```

# Data Flow Map
```yaml
data_flow:
  flows:
    pin_descriptor_access:
      path: instance/class -> __get__ -> call -> get_node -> function -> cache/store -> return
      notes: Async functions wrapped with ensure_future
    parent_call_override:
      path: child descriptor -> parent_call -> get_attr (MRO) -> extract_wrapped -> parent value -> child function(node, parent_value, ...)
    on_quit_teardown:
      path: signal/exception/atexit -> teardown -> callbacks -> restore_original_handlers
    proxy_to_wrapper:
      path: class access -> wrapper -> pivot resolution -> getter -> optional pre-processing -> return
    importer_required:
      path: dotted path -> get_module -> import_module -> get_child -> cached result
```

# Database Schema Knowledge
```yaml
database:
  tables: {}
  indexes: []
  relationships: []
```

# Configuration Registry
```yaml
config:
  sources:
    pyproject.toml: [project metadata, build-system, dependency-groups, tool.* configs]
    justfile: [development commands, packaging commands]
    Makefile: [install, lint, test shortcuts]
    mise.toml: [tool versions]
    etc/lint/ruff.toml: [ruff lint rules]
    etc/lint/mypy.toml: [type checker config]
    etc/lint/pyright.json: [basedpyright config]
    etc/pre-commit.yaml: [pre-commit hooks]
  secrets:
    - .env (UV_INDEX_PRIVATE_PASSWORD, etc.)
```

# Testing Matrix
```yaml
testing:
  strategies:
    unit:
      runner: pytest
      command: make test (or uv run pytest)
      coverage: pytest-cov with --cov app (note: references "app" not "src/kain")
  fixtures: {}
  mock_rules: []
```

# Gotchas & Quirks
```yaml
gotchas:
  - id: G1
    description: "Two overlapping descriptor implementations exist: descriptors.py (legacy) and properties/ (modern). Both are exported from kain.__init__.py."
    discovered: 2026-04-03
  - id: G2
    description: "Makefile test target uses --cov app but source directory is src/kain/. Coverage may report unexpectedly."
    discovered: 2026-04-03
  - id: G3
    description: "parent_call uses index=func.__name__ not in Is.classOf(node).__dict__ which relies on bool being subclass of int."
    discovered: 2026-04-03
  - id: G4
    description: "bound_property and pin reject coroutine functions at decoration time; async properties must use pin.native (cached_property)."
    discovered: 2026-04-03
  - id: G5
    description: "on_quit is a Singleton metaclass instance that replaces sys.excepthook and signal handlers on import."
    discovered: 2026-04-03
  - id: G6
    description: "pin.pre / pin.post still create empty __instance_memoized__ / __class_memoized__ dicts even when the value is not cached; only the specific key is omitted."
    discovered: 2026-04-02
  - id: G7
    description: "class_property passes the accessed class (e.g. Bar), not the owner class (Foo), because ClassProperty.get_node is dead code for plain (non-cached) descriptors."
    discovered: 2026-04-02
  - id: G8
    description: "mixed_property has a known behavior where falsy instances (__bool__ returning False) receive the class instead of themselves due to `instance or klass` — external libraries depend on this."
    discovered: 2026-04-02
  - id: G9
    description: ".here changes cache storage location, not the class passed to the function. Without .here: cache on accessed class. With .here: cache on defining/parent class."
    discovered: 2026-04-02
  - id: G10
    description: "invocation_context_check in src/kain/properties/primitives.py is dead code — defined but never used by any class in the properties package (only used in descriptors.py)."
    discovered: 2026-04-03
  - id: G11
    description: "properties/cached/klass.py fixed a latent TTL bug compared to descriptors.py: it uses isinstance(value, float) instead of a truthiness check, so timestamp 0.0 is not mistaken for falsy."
    discovered: 2026-04-03
  - id: G12
    description: "Monkey.patch with a tuple (node, name) calls required(node, name), which works only when node is a module or object with hasattr(node, name) — not for plain dicts."
    discovered: 2026-04-03
  - id: G13
    description: "Monkey.wrap(original) without an explicit `name` almost always fails because Who.Name(node) != func.__name__ and required(node, 'wrapper') raises ImportError."
    discovered: 2026-04-03
  - id: G14
    description: "sys.excepthook is not self.exceptions_hooks_proxy is a classic Python bound-method identity trap: every attribute access on a bound method produces a new wrapper object, so `is` comparison always fails."
    discovered: 2026-04-03
  - id: G15
    description: "RuntimeError('boom') == RuntimeError('boom') can be False in CPython because exception equality is identity-based unless __eq__ is overridden."
    discovered: 2026-04-03
  - id: G16
    description: "proxy_to has three bind modes; safe=True only blocks public attributes (not starting with _); private attributes bypass collision check entirely."
    discovered: 2026-04-03
  - id: G17
    description: "is_subclass handles types.UnionType and typing.Union correctly but does not handle Literal[1, 2] (raises TypeError from issubclass)."
    discovered: 2026-04-03
  - id: G18
    description: "unique with `key` de-duplicates by key value, so multiple items sharing the same key are collapsed to the first occurrence."
    discovered: 2026-04-03
  - id: G19
    description: "pretty_module(os.path.join) returns posixpath on macOS, not os.path, because os.path.join is defined in the posixpath module."
    discovered: 2026-04-03
```

# Insights & Patterns
```yaml
insights:
  - id: I1
    description: "Zero-runtime-dependency philosophy: every module uses only stdlib."
    discovered: 2026-04-03
  - id: I2
    description: "Descriptor system is highly stratified: primitives -> class/mixed -> cached (klass/instance/mixed/pre/post) -> parent variants."
    discovered: 2026-04-03
  - id: I3
    description: "Is and Who are implemented as dataclasses full of callable attributes, acting as ergonomic namespaces."
    discovered: 2026-04-03
  - id: I4
    description: "proxy_to uses dynamic wrapper generation with bound_property as the default descriptor factory, enabling zero-boilerplate attribute forwarding."
    discovered: 2026-04-03
  - id: I5
    description: "Protocol with __call__ overloads is the correct pattern for classes that act as decorators with chained attribute access (e.g. pin.cls.here.ttl())."
    discovered: 2026-04-02
  - id: I6
    description: "__new__ overloads work better than function-based decorator approaches because they preserve the class identity for with_parent access and enable proper type inference."
    discovered: 2026-04-02
```

# Investigation Log
```yaml
investigation_log: []
```

# Uncertainty Registry
```yaml
unknowns:
  - id: U1
    question: "Is descriptors.py planned for deprecation or will it continue to be maintained alongside properties/?"
    context: Both modules export similar classes and both are in __all__.
    status: open
```

# Deprecated Knowledge
```yaml
deprecated: []
```

# Performance Baselines
```yaml
performance:
  benchmarks: []
```

# External Dependencies
```yaml
third_party:
  runtime: []
  development:
    - pytest
    - ruff
    - black
    - basedmypy
    - basedpyright
    - bandit
    - refurb
    - vulture
    - pre-commit
    - uv
```

# Health Check Endpoints
```yaml
health: []
```

# Critical Paths & Failure Domains
```yaml
critical_paths:
  - "kain.properties.primitives.extract_wrapped is central to parent_call; adding new descriptor types requires updating it"
single_points_of_failure:
  - "on_quit singleton modifies global sys.excepthook and signal handlers; misuse can break process-wide exception handling"
```

# Auto-Update Rules
```yaml
maintenance:
  auto_update_triggers:
    - file_changed: "src/kain/properties/**/*.py"
      update_section: [class_hierarchy, dependency_graph]
    - file_changed: "src/kain/*.py"
      update_section: [dependency_graph, insights]
    - file_changed: "pyproject.toml"
      update_section: [config, third_party]
    - file_changed: "tests/**/*.py"
      update_section: testing
  
  validation_rules:
    - rule: "All BaseProperty subclasses must have entry in class_hierarchy"
      severity: warning
    - rule: "All inter-module imports must be in dependency_graph"
      severity: warning
  
  cleanup:
    remove_entries_older_than: "90 days"
```

---

# Usage Instructions for Agents

## When starting session:
1. Read this file entirely - it's structured for fast parsing
2. Check `investigation_log` for relevant solved problems
3. Note `unknowns` that might affect current task
4. Review `critical_paths` if changing infrastructure

## When discovering new knowledge:
1. Identify correct YAML section
2. Append/update with timestamp
3. If replacing old knowledge → move to `deprecated`
4. If investigation completes → update status, add conclusion

## Auto-write triggers (NO user prompt needed):
- New class discovered → class_hierarchy
- New dependency found → dependency_graph
- Gotcha encountered → gotchas
- Insight realized → insights
- Investigation done → investigation_log
