#!/bin/bash
# init.sh
#
# This script sets up a virtual environment with uv and Python 3.11,
# installs required packages, prepares a configuration file, and launches the server.
#
# Usage: ./init.sh [-p PORT]
#   -p PORT : Optional flag to specify the port number (default is 5000)

# Default port
PORT=5000

# Install dependencies with uv
echo "Installing dependencies with uv..."
cd activetigger/api || { echo "Directory activetigger/api not found"; exit 1; }
uv sync

# Check for a config.yaml file in the api directory
if [ ! -f config.yaml ]; then
    if [ -f config.yaml.sample ]; then
        echo "No config.yaml found. Copying config.yaml.sample to config.yaml..."
        cp config.yaml.sample config.yaml
        echo "You can now edit activetigger/api/config.yaml to modify paths for static files and database."
    else
        echo "No config.yaml.sample found in activetigger/api. Please ensure your configuration is set as needed."
    fi
fi

# Launch the server
echo "Launching the server on port $PORT..."
uv run python -m activetigger -p "$PORT"