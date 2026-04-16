.PHONY: install
install:
	uv sync

.PHONY: ci
ci:
	uv sync --locked --all-extras --dev

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: lint-fix
lint-fix:
	uv run ruff check . --fix

.PHONY: format
format:
	uv run ruff format . --check

.PHONY: format-fix
format-fix:
	uv run ruff format .

.PHONY: typecheck
typecheck:
	uv run ty check .

.PHONY: test
test:
	uv run pytest -v

.PHONY: test-coverage
test-coverage:
	uv run coverage run -m pytest
	uv run coverage report -m
	uv run coverage html