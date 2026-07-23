ARG PYTHON_VERSION="3.11"
FROM python:${PYTHON_VERSION}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd python --gid $GROUP_ID
RUN useradd -ms /bin/bash python --uid $USER_ID --gid $GROUP_ID

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# No USER directive: the container starts as root so entrypoint.sh can chown
# the named volumes, then drops to ${USER_ID} via gosu.

ENV UV_PROJECT_ENVIRONMENT=/home/python/venv
ENV VIRTUAL_ENV=/home/python/venv
