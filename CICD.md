# CI/CD

The project uses GitHub Actions. Workflows live in `.github/workflows/`:

| Workflow | File | Trigger | Purpose |
| --- | --- | --- | --- |
| [Check code sanity](#check-code-sanity) | `check-main.yml` | push / PR on `main`, `dev`; manual | Lint, type-check, and test the backend and frontend |
| [Build the frontend](#build-the-frontend) | `build-react-app.yml` | push / PR on `deployed`; manual | Build the React app and deploy it to GitHub Pages |
| [Build and push Docker images](#build-and-push-docker-images) | `build-docker-images.yml` | manual only | Build the API and frontend Docker images and push them to Docker Hub |

## Check code sanity

`check-main.yml` — runs on every push and pull request to `main` or `dev`, and can be triggered manually. Two jobs run in parallel.

### `check-api` (Python backend)

Runs in the `./api` directory.

1. **Install uv** — Sets up the [uv](https://docs.astral.sh/uv/) package manager with caching enabled.
2. **Set up Python 3.13** — Pinned to stay within the `>= 3.11, < 3.14` constraint.
3. **Install the project** — `make ci` (installs all dependencies from the lockfile).
4. **Format check** — `make format` (ruff formatting).
5. **Lint check** — `make lint` ([ruff](https://docs.astral.sh/ruff/)).
6. **Type check** — `make typecheck` ([ty](https://docs.astral.sh/ty/)).
7. **Tests** — `make test-coverage` (pytest with coverage).

### `check-frontend` (React/TypeScript frontend)

Runs in the `./frontend` directory.

1. **Install Node 24** — Sets up Node.js.
2. **Cache node_modules** — Keyed on the `package-lock.json` hash.
3. **Install dependencies** — `npm ci` for a clean install.
4. **Lint** — ESLint with Prettier integration (`npm run lint`).
5. **Build** — TypeScript compilation and Vite build (`npm run build`).

## Build the frontend

`build-react-app.yml` — runs on push and pull request to the `deployed` branch, and can be triggered manually. This is the production deployment path for the frontend served through GitHub Pages.

1. **Build** — `npm i && npm run build -- --base=/` in `./frontend`, with `VITE_API_URL` taken from the repository **variable** `VITE_API_URL` (Settings > Secrets and variables > Actions > Variables). This is the URL of the production API the deployed frontend talks to.
2. **Deploy** — Pushes `frontend/dist` to the `gh-pages` branch (via `JamesIves/github-pages-deploy-action`), which GitHub Pages serves.

Configuration:

- Repository variable `VITE_API_URL` must point to the production API endpoint.
- GitHub Pages must be configured to serve from the `gh-pages` branch.
- The workflow needs no secrets beyond the automatic `GITHUB_TOKEN` (it has `contents: write` permission to push to `gh-pages`).

## Build and push Docker images

`build-docker-images.yml` — **manual trigger only** (`workflow_dispatch`). Builds the two Docker images used by the Kubernetes/Helm deployment (see `charts/README.md`) and pushes them to Docker Hub:

- `activetigger/activetigger-api` — built from `docker/api/Dockerfile`;
- `activetigger/activetigger-frontend` — built from `docker/frontend/Dockerfile`.

Both are built from the repository root as context (the root `.dockerignore` keeps local data out) for `linux/amd64`, matching the Kubernetes cluster nodes. The two images build in parallel via a job matrix, with GitHub Actions layer caching per image.

### One-time configuration

1. On [hub.docker.com](https://hub.docker.com), with an account that has push access to the `activetigger` organization: Account Settings > **Personal access tokens** > generate a token with **Read & Write** scope.
2. In the GitHub repository: Settings > Secrets and variables > Actions > add two **repository secrets**:
   - `DOCKERHUB_USERNAME` — the Docker Hub username;
   - `DOCKERHUB_TOKEN` — the access token from step 1.

### Running it

From the GitHub UI: **Actions** tab > "Build and push Docker images" > **Run workflow**, then choose:

- **tag** — the image tag to push (default `prototype`, the tag the Helm chart pulls). Use another tag (e.g. `test`) to publish without affecting deployments.
- **image** — `both` (default), `api`, or `frontend` to rebuild only one side.

From the command line:

```bash
gh workflow run build-docker-images.yml -f tag=prototype -f image=both
```

Note: the "Run workflow" button only appears in the Actions tab once the workflow file exists on the default branch (`main`). Before that, trigger it on a feature branch with `gh workflow run build-docker-images.yml --ref <branch>`.
