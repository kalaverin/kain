# kain

[![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD-green)](LICENSE)

> Minimalist Python utility library. Zero runtime dependencies.

`kain` is a set of helper tools: introspection, descriptors, dynamic imports, monkey-patching, signal handling. Everything is built on `typing` and the standard library — no external packages.

---

## Installation

Via [uv](https://docs.astral.sh/uv/):

```bash
uv add kain
```

---

## Contents

- [Feature overview](#feature-overview)
- [How to use](#how-to-use)
  - [`Is` — type predicates](#is--type-predicates)
  - [`Who` — introspection and names](#who--introspection-and-names)
  - [`required` / `optional` — dynamic imports](#required--optional--dynamic-imports)
  - [`class_property` / `mixed_property` — descriptors](#class_property--mixed_property--descriptors)
  - [`pin` — cached properties](#pin--cached-properties)
  - [`on_quit` / `quit_at` — signals](#on_quit--quit_at--signals)
  - [Utilities](#utilities)
- [License](#license)

---

## Feature overview

| Module | Purpose |
|--------|---------|
| `kain.Is` | Type predicates: `Is.Class`, `Is.callable`, `Is.collection`, `Is.mapping`, `Is.subclass`, ... |
| `kain.Who` | Object name formatting: `Who.Name`, `Who.Addr`, `Who.Is`, `Who.Module`, `Who.Args` |
| `kain.importer` | Dynamic imports: `required()`, `optional()`, `add_path()`, `get_path()` |
| `kain.properties` | Descriptors: `class_property`, `mixed_property`, `pin`, cached variants |
| `kain.signals` | Graceful shutdown: `on_quit`, `quit_at` |
| `kain.classes` | Sentinels: `Missing`, `Nothing`, `Singleton` |
| `kain.internals` | Utilities: `to_ascii`, `to_bytes`, `unique`, `iter_inheritance` |

---

## How to use

### `Is` — type predicates

`Is` is a namespace module of predicates. All members are callables returning `bool` (or a type for `Is.classOf`).

```python
from kain import Is

Is.Class(str)           # True
Is.callable(print)      # True
Is.collection([1, 2])   # True
Is.mapping({"a": 1})    # True
Is.primitive(42)        # True
Is.subclass(int, object)  # True

# Special
Is.tty                  # True if stdin/stdout/stderr are TTYs
Is.Builtin              # alias for is_from_builtin
```

### `Who` — introspection and names

`Who` builds human-readable strings for objects — fully-qualified names, memory addresses, call arguments.

```python
from kain import Who

Who.Name(str)           # 'builtins.str'
Who.Addr(str)           # 'builtins.str#7f8b2c4d1e00'  # with hex id
Who.Is(str)             # 'builtins.str (type)'
Who.Module(str)         # 'builtins'

# Call arguments
Who.Args(1, "hello", flag=True)  # "1, 'hello', flag=True"

# Short name
Who.Name(str, full=False)        # 'str'
```

### `required` / `optional` — dynamic imports

Import by string name with different strictness levels.

```python
from kain import required, optional

# Strict — raises ImportError if not found
os_path = required("os.path")
join = required("os.path.join")

# Optional — falls back to default if package is missing
natsorted = optional("natsort.natsorted", default=sorted)
# If natsort is not installed, returns built-in sorted
```

Adding paths to `sys.path`:

```python
from kain import add_path

add_path("../src")      # Path("../src").resolve() is added to sys.path
```

### `class_property` / `mixed_property` — descriptors

Analogues of `property`, but for the class (`class_property`) and both levels (`mixed_property`).

```python
from kain import class_property, mixed_property

class Config:
    _name = "default"

    @class_property
    def name(cls):
        return cls._name

    @mixed_property
    def greet(cls_or_self):
        who = "class" if isinstance(cls_or_self, type) else "instance"
        return f"hello from {who}"

Config.name        # 'default'
Config().greet     # 'hello from instance'
Config.greet       # 'hello from class'
```

### `pin` — cached properties

`pin` is the unified entry point for all cached descriptors. It exposes aliases for different variants:

```python
from kain import pin

class Data:
    @pin
    def expensive(self):
        return sum(range(10_000_000))

    @pin.cls
    def class_level(cls):
        return f"cache on {cls.__name__}"

    @pin.any
    def mixed(cls_or_self):
        return "works on both"

d = Data()
d.expensive        # computed once, then read from __dict__
Data.class_level   # cache on class
```

Under the hood, `pin` uses the `kain.properties.cached` descriptor family:

| Descriptor | Cache on | Access level |
|-----------|----------|--------------|
| `cached_property` | instance | instance only |
| `class_cached_property` | class | class only |
| `mixed_cached_property` | class | class + instance |
| `pre_cached_property` | class | instance (value stored on class) |
| `post_cached_property` | class | instance (lazy evaluation) |

All support inheritance via `with_parent`:

```python
from kain.properties import class_property

class Base:
    @class_property
    def label(cls):
        return "base"

class Derived(Base):
    @class_property.with_parent
    def label(cls, parent_value):
        return f"derived:{parent_value}"

Derived.label      # 'derived:base'
```

### `proxy_to` — attribute proxying

Class decorator that forwards method calls to a pivot attribute.

```python
from kain import proxy_to

@proxy_to("_inner", "read", "write", "close")
class Wrapper:
    def __init__(self, inner):
        self._inner = inner

# Wrapper.read / Wrapper.write / Wrapper.close now proxy
# to _inner.read / _inner.write / _inner.close
```

### `Monkey` — monkey-patch

```python
from kain import Monkey

# Replace an attribute on a module or object
Monkey.patch((os.path, "join"), lambda *a: "patched!")

# Wrap an existing callable
original = Monkey.wrap(len, lambda orig, x: orig(x) * 2)

# Bind a new method to a class
Monkey.bind(SomeClass, "new_method", lambda self: 42)

# Decorator: suppress specified exceptions
class Parser:
    @Monkey.expect(ValueError)
    def parse_int(cls, s):
        return int(s)

Parser.parse_int("not-a-number")   # None, exception swallowed
```

### `on_quit` / `quit_at` — signals

```python
from kain import on_quit, quit_at

# Register a callback for graceful shutdown
@on_quit().schedule
def cleanup():
    print("Cleaning up before exit...")

# Auto-reload on file change (dev mode)
checker = quit_at()
while checker.sleep(2.5):
    pass  # main loop; exits when script mtime changes
```

`on_quit` is a singleton that intercepts `SIGINT`, `SIGTERM`, `SIGQUIT`, and `atexit`. All registered callbacks are executed exactly once.

### Utilities

```python
from kain import to_ascii, to_bytes, unique

# Encoding / decoding with default charset
to_bytes("hello")              # b'hello'
to_ascii(b"hello")             # 'hello'

# Unique elements with key and filtering
items = [1, 2, 2, 3, 1]
list(unique(items))            # [1, 2, 3]

# MRO iteration with filters
from kain.internals import iter_inheritance
list(iter_inheritance("hello", exclude_stdlib=True))
```

---

## License

BSD License. See [LICENSE](LICENSE).
