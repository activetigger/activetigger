#!/usr/bin/env bash
# Run all tests/checks for the frontend and the API.
# Usage: ./run-tests.sh

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=()

run_step() {
    local name="$1"
    local dir="$2"
    shift 2
    echo ""
    echo "==> $name"
    echo "    (in $dir: $*)"
    if ( cd "$dir" && "$@" ); then
        echo "    OK: $name"
    else
        echo "    FAIL: $name"
        FAILED+=("$name")
    fi
}

run_step "frontend lint"   "$ROOT_DIR/frontend" npm run lint
run_step "frontend build"  "$ROOT_DIR/frontend" npm run build
run_step "api lint"        "$ROOT_DIR/api"     make lint
run_step "api format"      "$ROOT_DIR/api"     make format
run_step "api typecheck"   "$ROOT_DIR/api"     make typecheck
run_step "api test"        "$ROOT_DIR/api"     make test

echo ""
echo "================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All checks passed."
    exit 0
else
    echo "Failed steps:"
    for step in "${FAILED[@]}"; do
        echo "  - $step"
    done
    exit 1
fi
