#!/usr/bin/env bash
# Run from project root: ./scripts/dev.sh [test|server]
set -e
cd "$(dirname "$0")/.."

case "${1:-server}" in
  test)
    pytest tests/ --ignore=tests/test_backtranslate.py -q
    ;;
  server)
    uvicorn app.main:app --reload --port "${PORT:-8001}"
    ;;
  *)
    echo "Usage: ./scripts/dev.sh [test|server]"
    exit 1
    ;;
esac
