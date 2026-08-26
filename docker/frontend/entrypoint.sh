#!/bin/sh
set -e

# Fail fast if the compose overlay declared a required MODE and .env disagrees.
if [ -n "$EXPECTED_MODE" ] && [ "$EXPECTED_MODE" != "$MODE" ]; then
  echo "ERROR: this compose overlay expects MODE=$EXPECTED_MODE but MODE=$MODE." >&2
  echo "       Edit docker/.env and set MODE=$EXPECTED_MODE (see DEPLOY.md)." >&2
  exit 1
fi

if [ "$MODE" = "dev" ]; then
  echo "/!\\ Mode is set to DEV /!\\"
else
  echo "/!\\ Mode is set to PRODUCTION /!\\"
fi
echo "(i) Npm version is $(npm -v)"
echo "(i) Node version is $(node -v)"

echo
echo " ~"
echo " ~ Install dependencies"
echo " ~"
echo

cd /frontend
npm install

if [ "$MODE" = "dev" ]; then
  echo
  echo " ~"
  echo " ~ Start frontend dev"
  echo " ~"
  echo
  npm run dev
else
  echo
  echo " ~"
  echo " ~ Build frontend"
  echo " ~"
  echo
  npm run build
fi
