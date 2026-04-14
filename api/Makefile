.PHONY: install
install:
	uv sync

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format . --check

.PHONY: format-fix
format-fix:
	uv run ruff format .

.PHONY: typecheck
typecheck:
	uv run mypy .

.PHONY: test
test:
	uv run pytest -v

.PHONY: test-coverage
test-coverage:
	uv run coverage run -m pytest
	uv run coverage report -m
	uv run coverage html