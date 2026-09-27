#!/usr/bin/env bash
# Evaluation matrix: run several configurations back to back and report each. Edit CONFIGS as needed.
#   scripts/rulebook_matrix.sh [GAMES] [JOBS]      e.g. scripts/rulebook_matrix.sh all 4
# Each entry: "<tag>|<mode>|<minutes>|<level_actions>|<extra config json>"
set -u
cd "$(dirname "$0")/.."
GAMES=${1:-all}; JOBS=${2:-4}
CONFIGS=(
  "mx-choose-15m|choose|15|300|{}"
  "mx-choose-think-15m|choose|15|300|{\"review_think\":\"always\"}"
  "mx-coder-15m|coder|15|300|{}"
  "mx-choose-nothink-15m|choose|15|300|{\"init_think\":false}"
)
for c in "${CONFIGS[@]}"; do
    IFS='|' read -r tag mode minutes la extra <<< "$c"
    cfg=/tmp/rulebook-matrix-$tag.json; echo "$extra" > "$cfg"
    .venv/bin/python scripts/run_rulebook.py --games "$GAMES" --mode "$mode" --minutes "$minutes" --level-actions "$la" --jobs "$JOBS" --config "$cfg" --tag "$tag" > "/tmp/rulebook-$tag.out" 2>&1
    .venv/bin/python scripts/rulebook_eval.py --tag "$tag" --name "$tag" | sed -n 3,5p
done
