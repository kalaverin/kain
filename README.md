---
title: kain
description: Runtime introspection, dynamic imports, and monkey-patching utilities for Python
---

[ref: #kain]

# kain

`kain` is a small Python toolbox for runtime introspection, dynamic imports, and controlled monkey-patching.

It gives you several public entry points:

- `Is` — type and shape predicates.
- `Who` — introspection helpers for names, modules, and inheritance.
- `Monkey` — runtime patching, binding, and exception suppression.
- `required`, `optional`, `add_path` — dynamic imports and `sys.path` management.
- `to_ascii`, `to_bytes`, `unique` — small text and iteration helpers.

[ref: #installation]

## Installation

Install with `uv`:

```bash
uv add kain
```

Or with any PEP 517-compatible tool:

```bash
pip install kain
```

[ref: #is]

## Type and shape checks with `Is`

`Is` is a namespace of predicates for classes, collections, primitives, mappings, and more.

```python
from kain import Is

assert Is.Class(int) is True
assert Is.Class(42) is False
assert Is.collection([1, 2, 3]) is True
assert Is.mapping({"a": 1}) is True
assert Is.primitive(42) is True
assert Is.subclass(42, int | str) is True
```

[ref: #who]

## Introspection with `Who`

`Who` gives you readable names, source files, argument lists, and inheritance chains.

```python
from kain import Who

assert Who.Is(42) == "int"
assert Who.Name(str) == "str"
assert Who.Cast(42) == "(int)42"
assert Who.Args(1, 2, a=3) == "'1', '2', a=3"

class A:
    pass

class B(A):
    pass

assert A in Who.Inheritance(B)
assert Who.Inheritance(B, glue=" -> ") == "__main__.A"
```

[ref: #importer]

## Dynamic imports with `required`, `optional`, and `add_path`

`required` imports by dotted path and raises on failure; `optional` returns a default instead.
`add_path` resolves a directory or file path and appends it to `sys.path`.

```python
from kain import add_path, optional, required

assert required("os.path.join") is __import__("os").path.join
assert optional("nonexistent_package_xyz") is None
assert optional("nonexistent_package_xyz", default="fallback") == "fallback"

# Add a directory or a file's parent directory to sys.path.
add_path(".")
```

[ref: #monkey]

## Runtime patching with `Monkey`

`Monkey` provides helpers for suppressing exceptions, attaching functions to objects, and replacing attributes while preserving the original in `Monkey.mapping`.

```python
from kain import Monkey

assert Parser.parse("not-a-number") is None
assert Parser.parse("42") == 42

node = type("Node", (), {})()


@Monkey.bind(node)
def greet(name: str) -> str:
    return f"hi {name}"


assert node.greet("world") == "hi world"
```

[ref: #text-and-unique]

## Text coercion and deduplication with `to_ascii`, `to_bytes`, and `unique`

`to_ascii` and `to_bytes` coerce between `str` and `bytes` with an optional charset.
`unique` yields distinct items from an iterable, optionally by a key function.

```python
from kain import to_ascii, to_bytes, unique

assert to_ascii(b"hello") == "hello"
assert to_bytes("hello") == b"hello"
assert to_ascii("é".encode("utf-8"), charset="utf-8") == "é"

assert list(unique([1, 2, 2, 3, 1])) == [1, 2, 3]
assert list(unique(["a", "A", "b"], key=str.lower)) == ["a", "b"]
```

[ref: #full-example]

## Full example

```python
from kain import Is, Monkey, Who, add_path, optional, required, to_ascii, unique


class Config:
    def port(cls, raw: str) -> int | None:
        return int(raw)


assert Config.port("8080") == 8080

assert Is.primitive("hello") is True
assert Who.Is(Config.port).endswith("Config.port")

join = required("os.path.join")
assert callable(join)

parser = optional("nonexistent_parser", default=str)
assert parser is str

add_path(".")

assert to_ascii(b"kain") == "kain"
assert list(unique(["x", "x", "y"])) == ["x", "y"]
```
