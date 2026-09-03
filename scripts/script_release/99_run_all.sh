#!/usr/bin/env bash
# Compatibility alias for the public root entry point.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${repo_root}/run_alps.sh" "$@"
