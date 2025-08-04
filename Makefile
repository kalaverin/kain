env:
	@make install
	@make sync

install:
	@mise trust --yes .mise.toml && mise install

sync:
	@uv venv --refresh
	@uv sync
	@make freeze

freeze:
	@uv lock
	@uv pip list --format=json > packages.json
	@uv pip compile \
		--output-file packages.txt \
		--generate-hashes pyproject.toml \
		--quiet

lint:
	@uv sync --group linting

	@uv run --quiet vulture \
		--min-confidence 66 \
		'src/' 'tests/'

	@uv run --quiet black \
		--check \
		--diff \
		'src/' 'tests/'

	@uv run --quiet mypy \
		--config-file etc/lint/mypy.toml \
		'src/' 'tests/'

	@uv run --quiet flakeheaven lint \
		--config etc/lint/flakeheaven.toml \
		'src/' 'tests/'

	@uv run --quiet ruff check \
		'src/' 'tests/'

	@uv run --quiet bandit \
		--quiet \
		--recursive \
		--severity-level all \
		--confidence-level all \
		--configfile pyproject.toml \
		'src/'

	@uv run --quiet bandit \
		--quiet \
		--recursive \
		--severity-level all \
		--confidence-level all \
		--configfile pyproject.toml \
		--skip B101,B105,B106 \
		'tests/'

	@uv run --quiet yamllint \
		--format parsable \
		--config-file etc/lint/yamllint.yaml \
		.

format:
	@uv sync --group linting

	@uv run --quiet black 'src/' 'tests/'

	@uv run --quiet ruff check \
		--fix \
		'src/' 'tests/'

	@uv run --quiet yamlfix \
		--exclude '.venv/' \
		.

	@uv run --quiet pre-commit run \
		--config etc/pre-commit.yaml \
		--show-diff-on-failure \
		--color always \
		--all

	@make lint

upgrade:
	@uv sync \
		--upgrade \
		--group development \
		--group linting \
		--group testing

	@uv lock --upgrade
	@make freeze
	@uv pip list

publish:
	@rm -rf dist/ || true
	@uv build
	@uvx uv-publish --repo kain
	@rm -rf dist/ || true

.DEFAULT_GOAL := format
