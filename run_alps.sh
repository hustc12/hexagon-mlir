#!/usr/bin/env bash
# Public zero-argument entry point for the complete ALPS evaluation suite.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${repo_root}/scripts/script_release/99_run_all.sh"
