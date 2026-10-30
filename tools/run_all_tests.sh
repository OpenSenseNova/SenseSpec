#!/bin/bash
# Run the retained DSpark, DSpark metrics and DFlash2 tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

exec python -m pytest tests/ "$@"
