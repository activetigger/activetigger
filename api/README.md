# API

This project is a FastAPI-based API using modern Python tooling for dependency management, code quality, and testing.

## Installation

Make sure you have [uv](https://docs.astral.sh/uv/) installed.

Install dependencies:

```sh
$> make install
```

For a full environment (CI, dev, extras):

```sh
$> make ci
```

## Tooling

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [Ruff](https://docs.astral.sh/ruff/) — linting & formatting
- [ty](https://docs.astral.sh/ty/) — type checking
- [Pytest](https://docs.pytest.org/en/stable/) — testing
- [Coverage.py](https://coverage.readthedocs.io/en/7.13.5/) — code coverage
- [uv](https://docs.astral.sh/uv/) — dependency management & execution

### Linting

Check for issues:

```sh
$> make lint
```

Auto-fix issues:

```sh
$> make lint-fix
```

### Formatting

Check formatting:

```sh
$> make format
```

Apply formatting:

```sh
$> make format-fix
```

### Type Checking

Run type checks:

```sh
$> make typecheck
```

### Tests

Tests are run locally and configure to use a sqlite database instance and the `test-data` folder.

#### Run tests

Run tests:

```sh
$> make test
```

Run tests with coverage :

```sh
$> make test-coverage
```

This generates:

- a terminal report
- an HTML report in `htmlcov`

#### Test Best Practices

- Each test must be independent, meaning it should not rely on the state or outcome of another test (e.g., Test A creates a project and Test B reuses it).
- When creating data (such as users or projects), always ensure uniqueness to avoid collisions and make tests repeatable. A common approach is to suffix identifiers with the current timestamp: `project_name = f"Test-{int(time.time())}"`
- This ensures that tests are idempotent and can be re-run safely without requiring manual cleanup of the SQLite database.
- Prefer generating fresh data per test rather than sharing state across tests. This improves reliability and makes debugging easier.
- The API is typed, so each endpoint has an input model. Please use it in your tests when calling the API.
  Example:

```python
data = NewUserModel(
    username=username,
    password="l3tm31n!",
    contact="activetigger@yopmail.com",
    status="active",
)

r = client.post(
    "/api/users/create",
    json=data.model_dump(),
    headers=superuser_header,
)
```

If the API input model changes, your tests will immediately fail, so you will be aware of it.
