#!/bin/bash
# Repoint hardcoded cluster paths to your own scratch.
# Usage: tools/retarget.sh /iopsstor/scratch/cscs/<youruser>
set -e
NEW="${1:?usage: retarget.sh <your-scratch-root>}"
OLD="/iopsstor/scratch/cscs/mrohania"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
grep -rl "$OLD" "$ROOT" --include='*.sh' --include='*.yaml' --include='*.py' 2>/dev/null \
  | while read -r f; do sed -i.bak "s|$OLD|$NEW|g" "$f" && rm -f "$f.bak"; done
echo "retargeted $OLD -> $NEW"
