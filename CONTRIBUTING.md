# Contributing to ActiveTigger

Thank you for your interest in contributing to ActiveTigger! To keep the project organized and avoid wasted effort, please follow the workflow below. We are a scientific software, so we would like to take some thoughts on what we put on our code.

## 1. Start with an issue

**Let us know what you want to do !**

Before writing any code, open a GitHub issue describing what you want to change and why. This lets maintainers and other contributors discuss the approach, suggest alternatives, and confirm the change is welcome before you invest time on it. 

- **Bug reports**: describe the problem, how to reproduce it, and what you expected instead.
- **Feature requests**: explain the use case and how the feature would work from a user's perspective.
- **Refactors / improvements**: explain what is wrong with the current code and what your proposed change improves.

## 2. One pull request, one concern

Each pull request should address a **single issue**. Do not bundle unrelated changes (e.g. a new feature + a refactor + a typo fix) into the same PR. Small, focused PRs are easier to review and less likely to introduce regressions.

If your work naturally splits into independent steps, submit them as separate PRs.

## 3. Fork and branch

1. Fork the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes.
4. Commit with clear, descriptive messages.
5. Push to your fork and open a pull request against `main`.

Reference the issue number in the PR description (e.g. "Closes #42").

## 4. Code guidelines

- **Backend** (Python / FastAPI): follow the existing code style. Use type hints where the surrounding code does.
- **Frontend** (React / TypeScript): follow the existing conventions in `frontend/src/`.
- Do not introduce new dependencies without discussing it in the issue first.
- Make sure the application starts and the feature you changed works before submitting.

## 5. Review process

- A maintainer will review your PR and may request changes.
- Please respond to review comments in a timely manner.
- Once approved, a maintainer will merge the PR.

## 6. Reporting bugs

When reporting a bug, include:

- Steps to reproduce the problem.
- Expected vs. actual behavior.
- Your environment (OS, browser, Docker or local install, Python version).
- Relevant logs or screenshots if available.

## Questions?

If you are unsure whether a change is appropriate, feel free to open an issue to ask. We are happy to discuss ideas before any code is written.
