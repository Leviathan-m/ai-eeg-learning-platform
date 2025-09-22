#!/usr/bin/env bash
set -euo pipefail

MESSAGE=${1:-"chore: automated update"}
BRANCH=${2:-"master"}

# Move to repo root regardless of current directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$REPO_ROOT" ]]; then
  echo "Not inside a git repository." >&2
  exit 1
fi
cd "$REPO_ROOT"

# Ensure git user is configured
if ! git config user.name >/dev/null; then
  git config user.name "automation-bot"
fi
if ! git config user.email >/dev/null; then
  git config user.email "automation@example.com"
fi

# Stage and commit if needed
git add -A
if ! git diff --cached --quiet; then
  git commit -m "$MESSAGE"
fi

# Token push if provided
if [[ -n "${GITHUB_PUSH_TOKEN:-}" && -n "${GITHUB_REPO_URL:-}" ]]; then
  AUTHED_URL=${GITHUB_REPO_URL/https:\/\//https://$GITHUB_PUSH_TOKEN@}
  git remote remove authed >/dev/null 2>&1 || true
  git remote add authed "$AUTHED_URL"
  git push authed "$BRANCH"
else
  git push origin "$BRANCH"
fi

echo "Pushed to branch '$BRANCH' successfully."


