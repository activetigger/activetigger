# CI/CD

The project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/check-main.yml`.

## Triggers

The workflow runs on:
- Push to `main` or `dev` branches
- Pull requests targeting `main` or `dev`
- Manual trigger (`workflow_dispatch`)

## Jobs

The two jobs run in parallel.

### `check-api` (Python backend)

Runs in the `./api` directory.

1. **Install uv** — Sets up the [uv](https://docs.astral.sh/uv/) package manager with caching enabled.
2. **Set up Python 3.13** — Installs Python 3.13 (pinned to stay within the `>= 3.11, < 3.14` constraint).
3. **Install the project** — Installs all dependencies from the lockfile (`uv sync --locked --all-extras --dev`).
4. **Lint check** — Runs [ruff](https://docs.astral.sh/ruff/) to check for linting errors.
5. **Type check** — Runs [ty](https://docs.astral.sh/ty/) for static type checking.

### `check-frontend` (React/TypeScript frontend)

Runs in the `./frontend` directory.

1. **Install Node 24** — Sets up Node.js.
2. **Cache node_modules** — Caches dependencies keyed on `package-lock.json` hash.
3. **Install dependencies** — Runs `npm ci` for a clean install.
4. **Lint** — Runs ESLint with Prettier integration (`npm run lint`).
5. **Build** — Runs TypeScript compilation and Vite build (`npm run build`).
