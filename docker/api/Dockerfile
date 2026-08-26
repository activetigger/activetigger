ARG PYTHON_VERSION="3.11"
FROM python:${PYTHON_VERSION}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd python --gid ${GROUP_ID} \
    && useradd -ms /bin/bash python --uid ${USER_ID} --gid ${GROUP_ID}

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/home/python/venv
ENV VIRTUAL_ENV=/home/python/venv
ENV MODE=prod
ENV API_PORT=4000
ENV DATA_PATH=/data
# CPU/GPU is a runtime choice: entrypoint.sh installs the matching torch extra
# (uv sync --extra cpu|gpu), so the same image serves both deployments.
ENV CPU_ONLY=true
# Inert unless the container runtime exposes NVIDIA devices/drivers (convention for NVIDIA drivers)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /api

# Compose bind-mounts the live checkout over /api and /entrypoint.sh, shadowing
# these baked-in copies; Kubernetes runs them as-is.
COPY --chown=python:python api/ /api/
COPY docker/api/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p /data/projects/static /data/models \
    && chown -R python:python /data

# No USER directive: under compose the container starts as root so
# entrypoint.sh can chown the named volumes, then drops to ${USER_ID} via gosu.
# Under Kubernetes the pod securityContext starts it directly as uid 1000 and
# the root block in entrypoint.sh is skipped.

EXPOSE 4000

ENTRYPOINT ["/entrypoint.sh"]
