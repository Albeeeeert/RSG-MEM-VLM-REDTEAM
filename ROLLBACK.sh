#!/usr/bin/env bash
set -euo pipefail

BASELINE_COMMIT="2c78457b898708f477e9c8514e1155e554edd852"
TARGET="${1:-$(pwd)}"
TARGET="$(cd "$TARGET" && pwd)"

git -C "$TARGET" rev-parse --is-inside-work-tree >/dev/null
git -C "$TARGET" cat-file -e "${BASELINE_COMMIT}^{commit}"
git -C "$TARGET" reset --hard "$BASELINE_COMMIT" >/dev/null

if [[ -n "$(git -C "$TARGET" status --porcelain)" ]]; then
  printf 'ROLLBACK_ERROR target=%s restored_commit=%s status=dirty\n' "$TARGET" "$BASELINE_COMMIT" >&2
  exit 1
fi
if [[ -d "$TARGET/src/rsg_mem" ]]; then
  printf 'ROLLBACK_ERROR target=%s implementation_present=true\n' "$TARGET" >&2
  exit 1
fi
printf 'ROLLBACK_OK target=%s restored_commit=%s status=clean implementation_present=false\n' "$TARGET" "$BASELINE_COMMIT"
