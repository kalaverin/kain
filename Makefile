.PHONY: check default help install lint test

help:
	@just default

install:
	@mise trust --yes mise.toml
	@mise install

lint:
	@uv run --quiet \
	  pre-commit run \
	--config etc/pre-commit.yaml \
	--all

check:
	@uv run --quiet \
		pre-commit run \
	--config etc/pre-commit.yaml \
	--all

stubs:
	@make clean-stubs || true
	@uv run --quiet righttyper \
	  --verbose \
	--python-version 3.12 \
		--generate-stubs \
		--overwrite \
		--replace-dict \
		--target-overhead 15 \
		--use-top-pct 90 \
		--srcdir src/zx/core \
		src/zx/core/__init__.py
		src/zx/core/Is.py \
		src/zx/core/Who.py \
		src/zx/core/importer.py \
		src/zx/core/monkey.py \
		src/zx/core/signals.py \
		|| true

	@cat righttyper.log righttyper.out || true
	@rm righttyper.log righttyper.out || true

	@rm src/zx/core/classes.pyi.bak 2>/dev/null || true
	@rm src/zx/core/importer.pyi.bak 2>/dev/null || true
	@rm src/zx/core/internals.pyi.bak 2>/dev/null || true
	@rm src/zx/core/_is.pyi.bak 2>/dev/null || true
	@rm src/zx/core/monkey.pyi.bak 2>/dev/null || true
	@rm src/zx/core/signals.pyi.bak 2>/dev/null || true
	@rm src/zx/core/who.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/__init__.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/cached/instance.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/cached/klass.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/cached/mixed.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/cached/post.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/cached/pre.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/class_property.pyi.bak 2>/dev/null || true
	@rm src/zx/core/properties/primitives.pyi.bak 2>/dev/null || true

clean-stubs:
	@rm src/zx/core/classes.pyi 2>/dev/null || true
	@rm src/zx/core/importer.pyi 2>/dev/null || true
	@rm src/zx/core/internals.pyi 2>/dev/null || true
	@rm src/zx/core/_is.pyi 2>/dev/null || true
	@rm src/zx/core/monkey.pyi 2>/dev/null || true
	@rm src/zx/core/signals.pyi 2>/dev/null || true
	@rm src/zx/core/who.pyi 2>/dev/null || true
	@rm src/zx/core/properties/__init__.pyi 2>/dev/null || true
	@rm src/zx/core/properties/cached/instance.pyi 2>/dev/null || true
	@rm src/zx/core/properties/cached/klass.pyi 2>/dev/null || true
	@rm src/zx/core/properties/cached/mixed.pyi 2>/dev/null || true
	@rm src/zx/core/properties/cached/post.pyi 2>/dev/null || true
	@rm src/zx/core/properties/cached/pre.pyi 2>/dev/null || true
	@rm src/zx/core/properties/class_property.pyi 2>/dev/null || true
	@rm src/zx/core/properties/primitives.pyi 2>/dev/null || true

test:
	@PYTHONASYNCIODEBUG=1 \
	uv run --quiet \
	pytest \
		-rs \
		-svvv \
		--cov src \
		--cov-report term-missing

publish:
	@rm -rf dist/ || true
	@uv build
	@uv run uv-publish --repo kain
	@rm -rf dist/ || true

%:
	@just $@

.DEFAULT_GOAL := default
