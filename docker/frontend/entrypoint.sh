#!/bin/sh
set -e

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
