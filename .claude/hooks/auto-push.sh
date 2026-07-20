#!/bin/bash
# Auto-commit + push after Edit/Write/NotebookEdit inside this repo.
# Never blocks Claude Code's flow: all failure paths print a warning and exit 0.

REPO="/Users/ryanwang/Desktop/ISEF"
LOG="$REPO/.claude/hooks/auto-push.log"

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"

# Only act on edits inside this repo.
case "$file_path" in
  "$REPO"/*) ;;
  *) exit 0 ;;
esac

cd "$REPO" || exit 0

warn() {
  echo "{\"systemMessage\": \"auto-push: $1\"}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"
  exit 0
}

if [ -z "$(git status --porcelain)" ]; then
  exit 0
fi

if ! git fetch origin main >>"$LOG" 2>&1; then
  warn "git fetch failed, skipped auto-push (changes stay local/uncommitted)"
fi

if ! git pull --rebase origin main >>"$LOG" 2>&1; then
  git rebase --abort >>"$LOG" 2>&1
  warn "rebase against origin/main failed (conflict), skipped auto-push - resolve manually"
fi

changed="$(git status --porcelain | awk '{print $2}' | xargs -n1 basename 2>/dev/null | paste -sd, -)"
timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

git add -A >>"$LOG" 2>&1

if ! git commit -m "Auto-commit: ${changed:-changes} (${timestamp})" >>"$LOG" 2>&1; then
  warn "git commit failed, changes left staged - check manually"
fi

if ! git push origin main >>"$LOG" 2>&1; then
  # Remote moved again between rebase and push - retry once.
  if git pull --rebase origin main >>"$LOG" 2>&1 && git push origin main >>"$LOG" 2>&1; then
    exit 0
  fi
  warn "git push failed after retry - commit is local only, push manually"
fi

exit 0
